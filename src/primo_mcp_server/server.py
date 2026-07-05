"""FastMCP server exposing Primo library search tools."""

from __future__ import annotations

import functools
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import httpx
from mcp.server.fastmcp import Context, FastMCP

from primo_mcp_server.citations import format_citation
from primo_mcp_server.client import PrimoAPIError, PrimoClient
from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.exporters import export_bibtex, export_csv, export_ris
from primo_mcp_server.formatter import (
    format_record_detail,
    format_search_results,
    format_suggestions,
)
from primo_mcp_server.librarians import (
    format_librarian_directory,
    format_librarian_recommendations,
    load_librarian_directory_cached,
    looks_like_identifier,
)
from primo_mcp_server.policy import PRIMO_SEARCH_DESCRIPTION, SERVER_INSTRUCTIONS
from primo_mcp_server.query import QueryClause
from primo_mcp_server.recommendation import (
    RecommendationOutcome,
    recommend_with_fallback,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Create a shared httpx client for the server lifetime."""
    config = PrimoConfig()
    async with httpx.AsyncClient(
        base_url=config.base_url,
        timeout=config.request_timeout,
        headers={"User-Agent": config.user_agent},
    ) as http_client:
        client = PrimoClient(http_client, config)
        yield {"client": client, "config": config}


mcp = FastMCP(
    "primo",
    instructions=SERVER_INSTRUCTIONS,
    lifespan=app_lifespan,
)


def _tool_error_boundary(action: str):
    """Uniform error boundary for MCP tools.

    Primo API failures return their caller-facing message ("Error {action}:
    ..."); anything else is a bug, so the traceback is logged before the
    short message goes back to the caller -- without the log, unexpected
    errors were invisible one-liners.
    """

    def decorate(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except PrimoAPIError as e:
                return f"Error {action}: {e}"
            except Exception as e:
                logger.exception("Unexpected error in %s", func.__name__)
                return f"Unexpected error: {e}"

        return wrapper

    return decorate


def _get_client(ctx: Context) -> PrimoClient:
    """Extract the PrimoClient from the lifespan context."""
    return ctx.request_context.lifespan_context["client"]


def _get_config(ctx: Context) -> PrimoConfig:
    """Extract the PrimoConfig from the lifespan context."""
    return ctx.request_context.lifespan_context["config"]


async def _format_recommendations_for_records(
    config: PrimoConfig,
    query: str,
    records,
    *,
    limit: int = 2,
    embedding_timeout: float | None = None,
) -> str:
    """Load configured profiles and format validated recommendations.

    The ranking itself lives in ``recommendation.recommend_with_fallback``
    (shared with the offline evaluation harness); this helper adds the
    identifier skip, directory loading, and MCP-facing formatting.

    Identifier-shaped queries (DOI, ISBN, ISSN, record ids) skip both paths:
    embedding a DOI produces noise and keyword-matching one is meaningless.
    """
    if looks_like_identifier(query):
        return format_librarian_recommendations(
            [],
            query,
            skip_reason=(
                "The query looks like a record identifier (DOI, ISBN, ISSN, "
                "or record ID), so librarian recommendations were skipped."
            ),
        )

    directory, message, specificity = load_librarian_directory_cached(
        config.librarians_file
    )
    if message or directory is None:
        return format_librarian_recommendations(
            [],
            query,
            configuration_message=message,
        )

    outcome = await recommend_with_fallback(
        directory,
        query,
        records,
        config,
        limit=limit,
        specificity=specificity,
        embedding_timeout=embedding_timeout,
    )
    _log_recommendation_outcome(config, query, outcome)
    return format_librarian_recommendations(
        outcome.matches,
        query,
        semantic_error=outcome.semantic_error,
        semantic_skipped=outcome.semantic_skipped,
        near_misses=outcome.near_misses,
    )


def _log_recommendation_outcome(
    config: PrimoConfig, query: str, outcome: RecommendationOutcome
) -> None:
    """Append one JSONL line per recommendation outcome (opt-in).

    The log exists to close the tuning loop: the golden eval set can only
    grow from real queries, and without a record of what matched (or
    near-missed) at what score, every mis-routed live query is lost. Logged
    only at the server layer so the offline eval harness never logs, and
    fail-silent so an unwritable path can never break a recommendation.
    """
    if not config.recommend_log_file:
        return

    def entry_for(match) -> dict:
        return {
            "id": match.librarian.id,
            "score": match.score,
            "terms": match.matched_terms,
            "fields": match.evidence_fields,
        }

    entry: dict = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "status": "matched" if outcome.matches else "no_match",
        "matches": [entry_for(match) for match in outcome.matches],
        "near_misses": [entry_for(near) for near in outcome.near_misses],
    }
    if outcome.semantic_error:
        entry["semantic_error"] = outcome.semantic_error
    if outcome.semantic_skipped:
        entry["semantic_skipped"] = outcome.semantic_skipped
    try:
        path = Path(config.recommend_log_file).expanduser()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Could not write recommendation log: %s", e)


# ---------------------------------------------------------------------------
# Tool 1: primo_search
# ---------------------------------------------------------------------------

@mcp.tool(description=PRIMO_SEARCH_DESCRIPTION)
@_tool_error_boundary("searching Primo")
async def primo_search(
    ctx: Context,
    query: str,
    field: str = "any",
    scope: str = "everything",
    sort_by: str = "rank",
    limit: int = 10,
    offset: int = 0,
    resource_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    peer_reviewed: bool | None = None,
    include_unavailable: bool | None = None,
    recommend_librarians: bool = True,
    librarian_limit: int = 2,
    clauses: list[QueryClause] | None = None,
    facet_filters: dict[str, str] | None = None,
    facet_exclusions: dict[str, str] | None = None,
) -> str:
    """Search Singapore Management University Library via Primo.

    The caller-facing scope and zero-result retry policy plus the full
    argument reference live in policy.PRIMO_SEARCH_DESCRIPTION, which is
    served as this tool's description.
    """
    client = _get_client(ctx)
    config = _get_config(ctx)
    response = await client.search(
        query=query,
        field=field,
        scope=scope,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
        resource_type=resource_type,
        date_from=date_from,
        date_to=date_to,
        peer_reviewed=peer_reviewed,
        include_unavailable=include_unavailable,
        clauses=clauses,
        facet_filters=facet_filters,
        facet_exclusions=facet_exclusions,
    )
    result = format_search_results(
        response,
        query,
        offset,
        config=config,
        field=field,
        scope=scope,
        sort_by=sort_by,
        resource_type=resource_type,
        date_from=date_from,
        date_to=date_to,
        peer_reviewed=peer_reviewed,
        include_unavailable=include_unavailable,
        clauses=clauses,
    )
    if (
        recommend_librarians
        and config.inline_librarian_recommendations
        # Identifier lookups (DOI, ISBN, record ids) get no inline
        # recommendation section at all rather than a "skipped" notice.
        and not looks_like_identifier(query)
    ):
        result += "\n\n" + await _format_recommendations_for_records(
            config,
            query,
            response.records,
            limit=librarian_limit,
            # Inline recommendations ride on every ordinary search, so a
            # slow embedding call gets a tighter budget than the explicit
            # primo_recommend_librarians tool.
            embedding_timeout=config.embedding_inline_timeout,
        )
    return result


# ---------------------------------------------------------------------------
# Tool 2: primo_get_record
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("fetching record")
async def primo_get_record(ctx: Context, record_id: str) -> str:
    """Get full details for a single library record.

    Use the record ID from primo_search results to fetch complete metadata
    including abstract, all authors, subjects, identifiers, and availability.

    Args:
        record_id: The Primo record ID (from search results, e.g. "alma991234567890" or "cdi_crossref_primary_10_1234").

    Returns:
        Full record details including title, authors, abstract, identifiers, and availability.
    """
    client = _get_client(ctx)
    config = _get_config(ctx)
    record = await client.get_record(record_id)
    if record is None:
        return (
            f'Record "{record_id}" not found. '
            "It may have been removed, or the ID may be incorrect. "
            "Try searching again with primo_search."
        )
    return format_record_detail(record, config=config)


# ---------------------------------------------------------------------------
# Tool 3: primo_suggest
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("getting suggestions")
async def primo_suggest(ctx: Context, query: str) -> str:
    """Get autocomplete suggestions for a search term.

    Useful for refining searches, checking subject headings, or exploring
    related terms before running a full search.

    Args:
        query: Partial search term (e.g. "entrepre" or "machine lear").

    Returns:
        List of suggested search terms.
    """
    client = _get_client(ctx)
    suggestions = await client.suggest(query)
    return format_suggestions(suggestions, query)


# ---------------------------------------------------------------------------
# Tool 4: primo_recommend_librarians
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("recommending librarians")
async def primo_recommend_librarians(
    ctx: Context,
    query: str,
    record_ids: list[str] | None = None,
    field: str = "any",
    scope: str = "everything",
    sort_by: str = "rank",
    offset: int = 0,
    search_limit: int = 5,
    resource_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    peer_reviewed: bool | None = None,
    include_unavailable: bool | None = None,
    limit: int = 2,
) -> str:
    """Recommend configured SMU librarian help for a Primo query or records.

    Recommendations are validated against the configured JSON profile
    directory. The server returns only configured librarian names; callers
    must not invent or substitute librarian recommendations. Callers should
    include the "Recommended librarian help:" section when summarising results.

    Args:
        query: User research topic or Primo search query.
        record_ids: Optional Primo record IDs to use as metadata evidence.
            When omitted, a small Primo search is run for context.
        field: Search field used when record_ids are omitted.
        scope: Search scope used when record_ids are omitted.
        sort_by: Sort order used when record_ids are omitted.
        offset: Search offset used when record_ids are omitted.
        search_limit: Number of Primo records to inspect when searching.
            Defaults to 5 and is capped by the Primo client.
        resource_type: Optional Primo resource type filter.
        date_from: Optional start year filter in YYYY format.
        date_to: Optional end year filter in YYYY format.
        peer_reviewed: Set to true to inspect only peer-reviewed items.
        include_unavailable: Set to true to include CDI records without full
            text access when searching for context.
        limit: Number of recommendations to return. Defaults to 2 and is
            capped at 3.

    Returns:
        Validated librarian recommendations, configuration guidance, or a
        no-recommendation message when matches are weak.
    """
    client = _get_client(ctx)
    config = _get_config(ctx)

    if record_ids:
        records = await client.get_records(record_ids)
    else:
        response = await client.search(
            query=query,
            field=field,
            scope=scope,
            sort_by=sort_by,
            limit=search_limit,
            offset=offset,
            resource_type=resource_type,
            date_from=date_from,
            date_to=date_to,
            peer_reviewed=peer_reviewed,
            include_unavailable=include_unavailable,
            # Records are only metadata evidence here; the facet summary
            # would be an unused second request.
            include_facets=False,
        )
        records = response.records

    return await _format_recommendations_for_records(
        config,
        query,
        records,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Tool 5: primo_list_librarians
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("listing librarians")
async def primo_list_librarians(ctx: Context) -> str:
    """List every configured SMU librarian profile.

    Use this when librarian recommendation returns no match but the user
    still wants a contact, or when the user asks who the librarians are and
    what they cover. The list is the complete configured directory: only
    these names may be presented; never invent or substitute names.

    Returns:
        All configured librarian profiles with title, contact, schools,
        best-for areas, and a sample of subjects, or configuration guidance
        when no directory is configured.
    """
    config = _get_config(ctx)
    directory, message, _ = load_librarian_directory_cached(
        config.librarians_file
    )
    if message or directory is None:
        return f"Librarian directory unavailable: {message}"
    return format_librarian_directory(directory)


# ---------------------------------------------------------------------------
# Tool 6: primo_cite
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("fetching records for citation")
async def primo_cite(
    ctx: Context,
    record_ids: list[str],
    style: str = "apa7",
) -> str:
    """Generate formatted citations for library records.

    Args:
        record_ids: List of Primo record IDs to cite.
        style: Citation style -- "apa7" (default), "harvard", "chicago", "ieee", "vancouver".

    Returns:
        Formatted citations. Note: always verify generated citations before submission.
    """
    valid_styles = {"apa7", "harvard", "chicago", "ieee", "vancouver"}
    style = style.strip().lower()
    if style not in valid_styles:
        return f'Invalid citation style "{style}". Use one of: {", ".join(sorted(valid_styles))}'

    client = _get_client(ctx)
    records = await client.get_records(record_ids)

    if not records:
        return "No records found for the provided IDs."

    citations = [format_citation(record, style) for record in records]

    result = "\n\n".join(citations)
    result += "\n\n-- Note: verify citations before submission. Automated formatting may not cover all edge cases."
    return result


# ---------------------------------------------------------------------------
# Tool 7: primo_export
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("fetching records for export")
async def primo_export(
    ctx: Context,
    record_ids: list[str],
    format: str = "bibtex",
) -> str:
    """Export library records to reference manager formats.

    Args:
        record_ids: List of Primo record IDs to export.
        format: Export format -- "bibtex" (default), "ris", "csv".

    Returns:
        Formatted export data ready for import into reference managers (Zotero, Mendeley, EndNote).
    """
    valid_formats = {"bibtex", "ris", "csv"}
    format = format.strip().lower()
    if format not in valid_formats:
        return f'Invalid format "{format}". Use one of: {", ".join(sorted(valid_formats))}'

    client = _get_client(ctx)
    records = await client.get_records(record_ids)

    if not records:
        return "No records found for the provided IDs."

    if format == "bibtex":
        return export_bibtex(records)
    elif format == "ris":
        return export_ris(records)
    else:
        return export_csv(records)
