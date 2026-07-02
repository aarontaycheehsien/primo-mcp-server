"""FastMCP server exposing Primo library search tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from mcp.server.fastmcp import Context, FastMCP

from primo_mcp_server.client import PrimoAPIError, PrimoClient
from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.formatter import (
    format_record_detail,
    format_search_results,
    format_suggestions,
)
from primo_mcp_server.librarian_embeddings import semantic_fallback
from primo_mcp_server.librarians import (
    _MAX_RECOMMENDATIONS,
    format_librarian_recommendations,
    load_librarian_directory_cached,
    looks_like_identifier,
    recommend_librarians as rank_librarians,
)


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
    instructions=(
        "Search Singapore Management University Library catalogue records, "
        "articles, databases, books, videos, and holdings via the Ex Libris "
        "Primo discovery API. "
        "Scope selection policy: when asked to search the catalogue, call "
        "primo_search with scope='catalogue' first; if that returns no "
        "results and the user did not ask for catalogue-only results, retry "
        "with scope='everything' and say that you widened the search. "
        "For books, databases, and videos, default to scope='catalogue'. "
        "For articles, default to scope='everything'. For confirmation "
        "requests about whether the library has, owns, subscribes to, or "
        "provides access to a title, use Primo as the evidence source and "
        "do not use websites, LibGuides, or general web pages unless the "
        "user explicitly asks for web confirmation. "
        "Use primo_search for queries, primo_get_record for full details, "
        "primo_suggest for autocomplete, primo_recommend_librarians for "
        "validated librarian recommendations, primo_cite for citations, "
        "and primo_export for BibTeX/RIS/CSV export. Librarian "
        "recommendations are limited to configured profile IDs; do not "
        "invent or substitute names."
    ),
    lifespan=app_lifespan,
)


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

    Deterministic keyword matching runs first. The semantic path runs when
    keywords find nothing OR when the best keyword score falls below the
    second-guess threshold, so a marginal keyword win (one generic stemmed
    term) cannot suppress a strong semantic match. Keyword matches stay
    primary and are never displaced; passing semantic candidates for other
    librarians are appended within the limit. Embedding cost is still paid
    only when keywords are weak or absent.

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

    matches = rank_librarians(
        directory,
        query,
        records,
        limit=limit,
        min_score=config.librarian_min_score,
        specificity=specificity,
    )
    semantic_error: str | None = None
    semantic_skipped: str | None = None
    best_keyword_score = matches[0].score if matches else 0.0
    if config.librarian_semantic_fallback and (
        not matches
        or best_keyword_score < config.librarian_semantic_second_guess_score
    ):
        semantic = await semantic_fallback(
            directory,
            query,
            records,
            config,
            limit=limit,
            timeout=embedding_timeout,
        )
        semantic_error = semantic.error
        semantic_skipped = semantic.skipped
        keyword_ids = {match.librarian.id for match in matches}
        matches = (
            matches
            + [
                match
                for match in semantic.matches
                if match.librarian.id not in keyword_ids
            ]
        )[: min(max(1, limit), _MAX_RECOMMENDATIONS)]
    return format_librarian_recommendations(
        matches,
        query,
        semantic_error=semantic_error,
        semantic_skipped=semantic_skipped,
    )


# ---------------------------------------------------------------------------
# Tool 1: primo_search
# ---------------------------------------------------------------------------

@mcp.tool()
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
) -> str:
    """Search Singapore Management University Library via Primo.

    Scope selection policy for callers:
    - When asked to search the catalogue, use scope="catalogue" first. If
      that returns no results and the user did not ask for catalogue-only
      results, retry with scope="everything" and say that the search was
      widened.
    - For books, databases, and videos, default to scope="catalogue".
    - For articles, default to scope="everything".
    - For confirmation requests about whether the library has, owns,
      subscribes to, or provides access to a title, use Primo as the
      evidence source. Do not rely on websites, LibGuides, or general web
      pages unless the user explicitly asks for web confirmation.

    Args:
        query: Search terms (e.g. "machine learning entrepreneurship").
        field: Search field -- "any" (default), "title", "creator", "sub" (subject), "isbn", "issn", "oclcnum".
        scope: "everything" for local catalogue + subscribed databases, "catalogue" for local only, "books_videos" for the books/videos scope.
        sort_by: "rank" (relevance, default), "date" (newest first), "title" (alphabetical).
        limit: Number of results to return (1-50, default 10).
        offset: Pagination offset (default 0). Use to get the next page of results.
        resource_type: Filter by type -- "books", "articles", "journals", "databases", "videos", "dissertations", "conference_proceedings".
        date_from: Start year filter (YYYY format, e.g. "2020").
        date_to: End year filter (YYYY format, e.g. "2025").
        peer_reviewed: Set to true to show only peer-reviewed items.
        include_unavailable: Set to true to also include article-index (CDI)
            records the library has NO full text access to (Primo's
            "expanded" search). Default (false) restricts results to
            accessible material, which is what holdings and access
            confirmation requires. Only set true when the user explicitly
            wants to discover material beyond the library's collection,
            e.g. for interlibrary loan or comprehensive literature mapping.
        recommend_librarians: Set to false to suppress inline librarian
            recommendations for this search. Inline recommendations also
            require PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS=true. When shown,
            callers should include the bottom "Recommended librarian help:" section
            when summarising Primo results.
        librarian_limit: Number of librarian recommendations to include
            inline. Defaults to 2 and is capped at 3.

    Returns:
        Formatted search results with title, authors, year, identifiers,
        availability, and any bottom "Recommended librarian help:" section.
    """
    try:
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
    except PrimoAPIError as e:
        return f"Error searching Primo: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool 2: primo_get_record
# ---------------------------------------------------------------------------

@mcp.tool()
async def primo_get_record(ctx: Context, record_id: str) -> str:
    """Get full details for a single library record.

    Use the record ID from primo_search results to fetch complete metadata
    including abstract, all authors, subjects, identifiers, and availability.

    Args:
        record_id: The Primo record ID (from search results, e.g. "alma991234567890" or "cdi_crossref_primary_10_1234").

    Returns:
        Full record details including title, authors, abstract, identifiers, and availability.
    """
    try:
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
    except PrimoAPIError as e:
        return f"Error fetching record: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool 3: primo_suggest
# ---------------------------------------------------------------------------

@mcp.tool()
async def primo_suggest(ctx: Context, query: str) -> str:
    """Get autocomplete suggestions for a search term.

    Useful for refining searches, checking subject headings, or exploring
    related terms before running a full search.

    Args:
        query: Partial search term (e.g. "entrepre" or "machine lear").

    Returns:
        List of suggested search terms.
    """
    try:
        client = _get_client(ctx)
        suggestions = await client.suggest(query)
        return format_suggestions(suggestions, query)
    except PrimoAPIError as e:
        return f"Error getting suggestions: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool 4: primo_recommend_librarians
# ---------------------------------------------------------------------------

@mcp.tool()
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
    try:
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
            )
            records = response.records

        return await _format_recommendations_for_records(
            config,
            query,
            records,
            limit=limit,
        )
    except PrimoAPIError as e:
        return f"Error recommending librarians: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool 5: primo_cite
# ---------------------------------------------------------------------------

@mcp.tool()
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
    try:
        from primo_mcp_server.citations import format_citation

        valid_styles = {"apa7", "harvard", "chicago", "ieee", "vancouver"}
        style = style.strip().lower()
        if style not in valid_styles:
            return f'Invalid citation style "{style}". Use one of: {", ".join(sorted(valid_styles))}'

        client = _get_client(ctx)
        records = await client.get_records(record_ids)

        if not records:
            return "No records found for the provided IDs."

        citations = []
        for record in records:
            citations.append(format_citation(record, style))

        result = "\n\n".join(citations)
        result += "\n\n-- Note: verify citations before submission. Automated formatting may not cover all edge cases."
        return result
    except PrimoAPIError as e:
        return f"Error fetching records for citation: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool 6: primo_export
# ---------------------------------------------------------------------------

@mcp.tool()
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
    try:
        from primo_mcp_server.exporters import export_bibtex, export_csv, export_ris

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
    except PrimoAPIError as e:
        return f"Error fetching records for export: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
