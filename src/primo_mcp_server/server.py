"""FastMCP server exposing Primo library search tools."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal

import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent

from primo_mcp_server.citations import format_citation
from primo_mcp_server.client import PrimoAPIError, PrimoClient
from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.exporters import export_bibtex, export_csv, export_ris
from primo_mcp_server.formatter import (
    build_search_url,
    format_record_detail,
    format_search_results,
    format_suggestions,
    search_query_label,
)
from primo_mcp_server.librarian_embeddings import warm_up_local_embedder
from primo_mcp_server.librarian_llm import (
    LibrarianChoice,
    Reasoner,
    sampling_reasoner,
    validate_choices,
)
from primo_mcp_server.librarians import (
    _MAX_RECOMMENDATIONS,
    LibrarianMatch,
    format_librarian_directory,
    format_librarian_recommendations,
    is_llm_match,
    is_semantic_match,
    load_librarian_directory_cached,
    looks_like_identifier,
)
from primo_mcp_server.policy import (
    PRIMO_SEARCH_DESCRIPTION,
    SEARCH_TRANSPARENCY_CALLER_ACTION,
    SERVER_INSTRUCTIONS,
)
from primo_mcp_server.query import QueryClause
from primo_mcp_server.rag_guard import (
    RagSessionStore,
    extract_citation_ids,
    format_no_citations_failure,
    format_retrieve_response,
    format_validation_failure,
    format_validation_success,
    validate_ids,
)
from primo_mcp_server.recommendation import (
    RecommendationOutcome,
    recommend_with_fallback,
)

logger = logging.getLogger(__name__)


@dataclass
class FormattedRecommendation:
    """A librarian outcome suitable for both text and structured MCP output."""

    status: Literal["matched", "no_match", "unavailable", "skipped"]
    text: str
    matches: list[LibrarianMatch]
    near_misses: tuple[LibrarianMatch, ...] = ()
    # True when the caller-reasoned tier asked this caller for a routing
    # decision. Status stays no_match -- the request is a task, not a
    # recommendation -- so this is what distinguishes "nothing to do" from
    # "something is owed" for a caller reading metadata.
    routing_request_pending: bool = False

    @property
    def caller_action(self) -> str | None:
        """Action a caller must take before answering the user."""
        if self.status == "matched":
            return "include_in_user_response_with_evidence"
        if self.routing_request_pending:
            return "submit_librarian_choice_or_state_no_match"
        return None


def _match_payload(match: LibrarianMatch, *, anonymous: bool = False) -> dict:
    """Serialise a configured librarian and the evidence supporting the match.

    ``anonymous`` drops the identity fields, leaving the id and the
    evidence. It is set while a routing decision is pending: the text
    withholds names so that primo_submit_librarian_choice is the only way
    to obtain one, and metadata a caller reads just as easily would
    otherwise hand back what the text declined to give.
    """
    librarian = match.librarian
    semantic = is_semantic_match(match)
    reasoned = is_llm_match(match)
    if semantic:
        match_type = "semantic"
    elif reasoned:
        match_type = "llm"
    else:
        match_type = "keyword"
    evidence: dict[str, object] = {
        "match_type": match_type,
        "score": match.score,
        "matched_terms": match.matched_terms,
        "evidence_fields": match.evidence_fields,
    }
    if semantic:
        evidence["cosine_similarity"] = match.score
    if reasoned:
        # Named so a caller cannot mistake it for a calibrated measurement.
        evidence["self_reported_confidence"] = match.score
        evidence["reasoning"] = (
            match.matched_terms[0] if match.matched_terms else ""
        )

    if anonymous:
        return {
            "id": librarian.id,
            "title": librarian.title,
            "evidence": evidence,
        }

    return {
        "id": librarian.id,
        "name": librarian.name,
        "title": librarian.title,
        "email": librarian.email,
        "url": librarian.url,
        "evidence": evidence,
    }


def _transparency_payload(
    query_label: str, result_count: int, search_url: str | None
) -> dict:
    """Machine-readable twin of the "Queries attempted:" text block.

    Covers this call only; the caller is told (in prose and here) to
    combine it across every primo_search call of the turn.
    """
    return {
        "caller_action": SEARCH_TRANSPARENCY_CALLER_ACTION,
        "queries_attempted": [
            {
                "query": query_label,
                "results": result_count,
                "url": search_url,
            }
        ],
        "total_results": result_count,
    }


def _search_tool_result(
    text: str,
    recommendation: FormattedRecommendation | None = None,
    transparency: dict | None = None,
) -> CallToolResult:
    """Return readable text plus explicit, evidence-bearing MCP metadata."""
    structured: dict[str, object] = {"result": text}
    if transparency is not None:
        structured["search_transparency"] = transparency
    if recommendation is not None:
        structured.update(
            {
                "librarian_status": recommendation.status,
                "caller_action": recommendation.caller_action,
                "librarian_recommendations": [
                    _match_payload(match) for match in recommendation.matches
                ],
                "closest_configured_contacts": [
                    _match_payload(
                        match, anonymous=recommendation.routing_request_pending
                    )
                    for match in recommendation.near_misses
                ],
            }
        )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
    )


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Create a shared httpx client for the server lifetime.

    When the semantic fallback runs against a local embedding runtime, a
    background warm-up loads the model at startup so the first search does
    not pay the multi-second model load inside its tight inline budget.
    """
    config = PrimoConfig()
    warmup = asyncio.create_task(warm_up_local_embedder(config))
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers={"User-Agent": config.user_agent},
        ) as http_client:
            client = PrimoClient(http_client, config)
            yield {"client": client, "config": config}
    finally:
        if not warmup.done():
            warmup.cancel()


mcp = FastMCP(
    "primo",
    instructions=SERVER_INSTRUCTIONS,
    lifespan=app_lifespan,
)

# Pinned retrievals for the RAG citation guardrail. Module-level so the
# retrieve and validate tool calls of one conversation share state for the
# lifetime of this stdio process.
RAG_SESSIONS = RagSessionStore()


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


def _reasoner_for(ctx: Context, config: PrimoConfig) -> Reasoner | None:
    """Build the tier-3 reasoner when it is configured to use sampling.

    Only the server can supply one: sampling runs on the connected client's
    model, so it needs the live session. Returning None leaves llm_fallback
    on its configured HTTP endpoint.
    """
    if not config.librarian_llm_fallback:
        return None
    if config.llm_provider.strip().lower() != "sampling":
        return None
    return sampling_reasoner(
        ctx.session, config=config, related_request_id=ctx.request_id
    )


async def _format_recommendations_for_records(
    config: PrimoConfig,
    query: str,
    records,
    *,
    limit: int = 2,
    embedding_timeout: float | None = None,
    reasoner: Reasoner | None = None,
) -> FormattedRecommendation:
    """Load configured profiles and format validated recommendations.

    The ranking itself lives in ``recommendation.recommend_with_fallback``
    (shared with the offline evaluation harness); this helper adds the
    identifier skip, directory loading, and MCP-facing formatting.

    Identifier-shaped queries (DOI, ISBN, ISSN, record ids) skip both paths:
    embedding a DOI produces noise and keyword-matching one is meaningless.
    """
    if looks_like_identifier(query):
        return FormattedRecommendation(
            status="skipped",
            text=format_librarian_recommendations(
                [],
                query,
                skip_reason=(
                    "The query looks like a record identifier (DOI, ISBN, ISSN, "
                    "or record ID), so librarian recommendations were skipped."
                ),
            ),
            matches=[],
        )

    directory, message, specificity = load_librarian_directory_cached(
        config.librarians_file
    )
    if message or directory is None:
        return FormattedRecommendation(
            status="unavailable",
            text=format_librarian_recommendations(
                [],
                query,
                configuration_message=message,
            ),
            matches=[],
        )

    outcome = await recommend_with_fallback(
        directory,
        query,
        records,
        config,
        limit=limit,
        specificity=specificity,
        embedding_timeout=embedding_timeout,
        reasoner=reasoner,
    )
    _log_recommendation_outcome(config, query, outcome)
    return FormattedRecommendation(
        status="matched" if outcome.matches else "no_match",
        text=format_librarian_recommendations(
            outcome.matches,
            query,
            semantic_error=outcome.semantic_error,
            semantic_skipped=outcome.semantic_skipped,
            near_misses=outcome.near_misses,
            llm_error=outcome.llm_error,
            llm_skipped=outcome.llm_skipped,
            llm_routing_request=outcome.llm_routing_request,
        ),
        routing_request_pending=bool(outcome.llm_routing_request),
        matches=outcome.matches,
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
    if outcome.llm_error:
        entry["llm_error"] = outcome.llm_error
    if outcome.llm_skipped:
        entry["llm_skipped"] = outcome.llm_skipped
    try:
        path = Path(config.recommend_log_file).expanduser()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning("Could not write recommendation log: %s", e)


# ---------------------------------------------------------------------------
# Tool 1: primo_search
# ---------------------------------------------------------------------------

@mcp.tool(description=PRIMO_SEARCH_DESCRIPTION, structured_output=False)
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
) -> CallToolResult:
    """Search Singapore Management University Library via Primo.

    The caller-facing scope and zero-result retry policy plus the full
    argument reference live in policy.PRIMO_SEARCH_DESCRIPTION, which is
    served as this tool's description.
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
        recommendation: FormattedRecommendation | None = None
        if (
            recommend_librarians
            and config.inline_librarian_recommendations
            # Identifier lookups (DOI, ISBN, record ids) get no inline
            # recommendation section at all rather than a "skipped" notice.
            and not looks_like_identifier(query)
        ):
            recommendation = await _format_recommendations_for_records(
                config,
                query,
                response.records,
                limit=librarian_limit,
                # Inline recommendations ride on every ordinary search, so a
                # slow embedding call gets a tighter budget than the explicit
                # primo_recommend_librarians tool.
                embedding_timeout=config.embedding_inline_timeout,
                reasoner=_reasoner_for(ctx, config),
            )
            if recommendation.status == "matched":
                result = (
                    "## Required librarian referral\n\n"
                    "Caller action: You MUST include every recommended "
                    "librarian's name, title, contact, and evidence in the "
                    "user-facing response.\n\n"
                    + recommendation.text
                    + "\n\n## Primo search results\n\n"
                    + result
                )
            elif recommendation.routing_request_pending:
                # A routing request is owed work, exactly like a referral, so
                # it leads the result instead of trailing several screens of
                # hits and guidance. Appended last, it was the easiest thing
                # in the response for a caller to never reach.
                result = (
                    "## Required librarian routing decision\n\n"
                    "Caller action: keyword matching found no librarian for "
                    "this query, so the decision is yours to make. Read the "
                    "routing task below, then either call "
                    "primo_submit_librarian_choice with the profiles that "
                    "genuinely fit, or tell the user no configured librarian "
                    "covers this topic. Deciding that none fits is a valid "
                    "answer; silently skipping the decision is not.\n\n"
                    + recommendation.text
                    + "\n\n## Primo search results\n\n"
                    + result
                )
            else:
                result += "\n\n" + recommendation.text
        return _search_tool_result(
            result,
            recommendation,
            _transparency_payload(
                search_query_label(query, field, clauses),
                response.info.total,
                build_search_url(
                    query,
                    config,
                    field=field,
                    scope=scope,
                    sort_by=sort_by,
                    offset=offset,
                    resource_type=resource_type,
                    date_from=date_from,
                    date_to=date_to,
                    peer_reviewed=peer_reviewed,
                    include_unavailable=include_unavailable,
                    clauses=clauses,
                ),
            ),
        )
    except PrimoAPIError as e:
        return _search_tool_result(f"Error searching Primo: {e}")
    except Exception as e:
        logger.exception("Unexpected error in primo_search")
        return _search_tool_result(f"Unexpected error: {e}")


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
    must not invent or substitute librarian recommendations. When Status is
    matched, callers MUST include every recommended librarian's name, title,
    contact, and evidence in the user-facing response.

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

    recommendation = await _format_recommendations_for_records(
        config,
        query,
        records,
        limit=limit,
        reasoner=_reasoner_for(ctx, config),
    )
    return recommendation.text


# ---------------------------------------------------------------------------
# Tool: primo_submit_librarian_choice
# ---------------------------------------------------------------------------

@mcp.tool()
@_tool_error_boundary("validating librarian choice")
async def primo_submit_librarian_choice(
    ctx: Context,
    query: str,
    choices: list[LibrarianChoice],
) -> str:
    """Validate a reasoned librarian choice and format it for the user.

    Step 2 of the caller-reasoned routing tier. When keyword matching finds
    no librarian, the recommendation output asks the calling model to
    decide which configured profile fits; this tool is where that decision
    is checked by code before anything can be shown.

    Every rule the other matching paths obey is re-applied here: an id that
    is not in the configured directory is discarded (never rendered under a
    fabricated name), a curator deny-list still suppresses its profile, a
    choice with no reason is dropped because evidence is mandatory, and a
    confidence below the configured floor is rejected. Passing this tool is
    the ONLY way a librarian from this tier may be named to a user.

    Args:
        query: The user's research topic, as given to the search that
            produced the routing request.
        choices: One object per librarian, each with "id" (exact id from
            the configured directory), "confidence" (0-1, your own
            estimate), and "reason" (one sentence naming the expertise
            that fits). All three are required on every choice.

    Returns:
        A validated, evidence-bearing recommendation, or a rejection
        explaining which ids were not accepted and why.
    """
    config = _get_config(ctx)
    directory, message, _ = load_librarian_directory_cached(
        config.librarians_file
    )
    if message or directory is None:
        return f"Librarian directory unavailable: {message}"

    # Over the wire FastMCP coerces each choice into LibrarianChoice, but an
    # in-process caller may hand over the plain dicts the schema describes;
    # primo_search accepts both forms of QueryClause for the same reason.
    matches = validate_choices(
        [
            choice.model_dump() if isinstance(choice, LibrarianChoice) else choice
            for choice in choices or []
        ],
        directory,
        query,
        config,
        limit=_MAX_RECOMMENDATIONS,
    )
    if not matches:
        known = ", ".join(profile.id for profile in directory.librarians)
        return (
            "No submitted choice passed validation, so no librarian may be "
            f'shown for "{query}". Every choice was rejected as an unknown '
            "id, a curator-excluded profile, a missing reason, or a "
            "confidence below the configured floor "
            f"({config.librarian_llm_min_confidence}). Configured ids: "
            f"{known}. Do not name a librarian in your reply; say none "
            "covers this topic, or offer primo_list_librarians as "
            "directory information."
        )
    return format_librarian_recommendations(matches, query)


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
# Tool: primo_rag_retrieve
# ---------------------------------------------------------------------------

@mcp.tool()
async def primo_rag_retrieve(
    ctx: Context,
    query: str,
    limit: int = 5,
    field: str = "any",
    scope: str = "everything",
    resource_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    peer_reviewed: bool | None = None,
    style: str = "apa7",
) -> str:
    """Retrieve top Primo records and pin them for guarded RAG answering.

    Step 1 of the citation-guardrail pipeline. Searches SMU Primo, pins the
    top records (default 5) to a session labelled [1]-[n], and returns the
    evidence pack plus drafting rules. Draft an answer citing ONLY numeric
    [n] tags, then call primo_rag_validate with the session_id and the draft
    -- validation and reference building are done by code, not by the model.

    Args:
        query: The user's research question or search terms.
        limit: Number of records to pin (1-10, default 5).
        field: Search field -- "any" (default), "title", "creator", "sub".
        scope: "everything" (default, catalogue + subscribed databases) or "catalogue".
        resource_type: Optional Primo resource type filter (e.g. "articles").
        date_from: Optional start year filter in YYYY format.
        date_to: Optional end year filter in YYYY format.
        peer_reviewed: Set to true to retrieve only peer-reviewed items.
        style: Citation style for the code-built reference list --
            "apa7" (default), "harvard", "chicago", "ieee", "vancouver".

    Returns:
        Session id, R#-labelled source records, and drafting rules.
    """
    try:
        client = _get_client(ctx)
        limit = max(1, min(limit, 10))
        response = await client.search(
            query=query,
            field=field,
            scope=scope,
            sort_by="rank",
            limit=limit,
            offset=0,
            resource_type=resource_type,
            date_from=date_from,
            date_to=date_to,
            peer_reviewed=peer_reviewed,
            include_facets=False,
        )
        records = response.records[:limit]
        if not records:
            return (
                f'No Primo results for "{query}". No RAG session was created. '
                "Follow the zero-result policy: revise the query (broader "
                "concepts, synonyms, relaxed filters) and call "
                "primo_rag_retrieve again, up to five total attempts."
            )
        session = RAG_SESSIONS.create(query, records, style=style)
        return format_retrieve_response(session)
    except PrimoAPIError as e:
        return f"Error searching Primo: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# Tool: primo_rag_validate
# ---------------------------------------------------------------------------

@mcp.tool()
async def primo_rag_validate(
    ctx: Context,
    session_id: str,
    draft_answer: str,
) -> str:
    """Validate a drafted RAG answer's [n] citations against retrieved records.

    Steps 3-5 of the citation-guardrail pipeline, executed by deterministic
    code: extract numeric [n] tags from the draft, check every ID against the
    records pinned by primo_rag_retrieve, and -- only if all IDs are valid --
    build the reference list from the pinned records' metadata.

    On failure the response contains regeneration feedback: rewrite the
    draft citing only retrieved sources and call this tool again. On success
    it contains the final answer with a code-built reference list; present
    that verbatim, including the guardrail note.

    Args:
        session_id: The session id returned by primo_rag_retrieve.
        draft_answer: The full drafted answer containing inline [R#] tags.

    Returns:
        VALIDATION PASSED with the assembled final answer, or VALIDATION
        FAILED with feedback for regeneration.
    """
    try:
        session = RAG_SESSIONS.get(session_id)
        if session is None:
            return (
                f'Unknown RAG session "{session_id}". Sessions live only for '
                "this server process. Call primo_rag_retrieve first and use "
                "the session id it returns."
            )
        session.attempts += 1
        cited = extract_citation_ids(draft_answer)
        if not cited:
            return format_no_citations_failure(session)
        valid, invalid = validate_ids(cited, len(session.records))
        if invalid:
            return format_validation_failure(session, invalid, cited)
        return format_validation_success(session, draft_answer, valid)
    except Exception as e:
        return f"Unexpected error: {e}"


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
