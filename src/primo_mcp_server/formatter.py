"""Format Primo records into compact, LLM-friendly text output."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from primo_mcp_server.models import PrimoRecord, SearchResponse
from primo_mcp_server.query import (
    date_range_facet_value,
    normalise_resource_type,
    normalise_scope,
    normalise_search_field,
    normalise_sort_by,
)

if TYPE_CHECKING:
    from primo_mcp_server.config import PrimoConfig


_PRIMO_API_SUFFIX = "/primaws/rest/pub"


def _format_authors(creators: list[str], max_authors: int = 3) -> str:
    """Format an author list, truncating with 'et al.' if needed."""
    if not creators:
        return "Unknown author"
    if len(creators) <= max_authors:
        return "; ".join(creators)
    return "; ".join(creators[:max_authors]) + " et al."


def _format_identifiers(record: PrimoRecord) -> str:
    """Format the most useful identifier for a record."""
    parts = []
    if record.doi:
        parts.append(f"DOI: {record.doi}")
    if record.isbn:
        parts.append(f"ISBN: {record.isbn[0]}")
    if record.issn:
        parts.append(f"ISSN: {record.issn[0]}")
    return " | ".join(parts) if parts else ""


def _format_preview(values: list[str], max_items: int = 4) -> str:
    """Format a short metadata preview for search-result snippets."""
    if not values:
        return ""
    shown = values[:max_items]
    suffix = " et al." if len(values) > max_items else ""
    return "; ".join(shown) + suffix


def _format_availability(record: PrimoRecord) -> str:
    """Format availability information."""
    parts = []
    if record.fulltext_available:
        parts.append("Full text available")
    elif _record_context(record) == "PC":
        # CDI records without the fulltext flag are explicit no-access
        # cases (delivery/fulltext = no_fulltext). Physical Alma holdings
        # also lack the flag but may well be on the shelf, so only remote
        # records get this label; local ones keep the OneSearch fallback.
        parts.append("No full text access")
    if record.delivery_category:
        parts.append(record.delivery_category)
    return " | ".join(parts) if parts else "Check availability in OneSearch"


def _derive_discovery_base_url(base_url: str) -> str:
    """Derive the Primo discovery app root from a Primo API base URL."""
    trimmed = base_url.rstrip("/")
    if trimmed.lower().endswith(_PRIMO_API_SUFFIX):
        return trimmed[: -len(_PRIMO_API_SUFFIX)]
    return trimmed


def _discovery_app_base_url(config: PrimoConfig) -> str:
    """Return the discovery app base URL, ending in /discovery."""
    base = (config.discovery_base_url or _derive_discovery_base_url(config.base_url)).rstrip("/")
    if not base:
        return ""
    if base.lower().endswith("/discovery"):
        return base
    return f"{base}/discovery"


def _search_scope_params(config: PrimoConfig, scope: str) -> tuple[str, str] | None:
    """Return Primo UI tab and search_scope values for a caller scope."""
    try:
        canonical_scope = normalise_scope(scope)
    except ValueError:
        return None
    if canonical_scope == "catalogue":
        return config.tab_catalogue, config.scope_local
    if canonical_scope == "everything":
        return config.tab_everything, config.scope_combined
    if canonical_scope == "books_videos":
        return config.tab_books_videos, config.scope_books_videos
    return None


def _record_context(record: PrimoRecord) -> str:
    """Return the Primo full-display context for a local or remote record."""
    if record.context.strip().upper() == "L":
        return "L"
    if record.record_id.lower().startswith("alma"):
        return "L"

    # Local records carry sourceid "alma" and source label "Alma" exactly
    # (observed live); CDI records carry database names. A substring test
    # ("alma" in value) misclassified any remote record whose source name
    # contains those letters, e.g. "World Almanac", producing fulldisplay
    # URLs with context=L that do not resolve.
    source_values = (record.source_id, record.source_system, record.source_label)
    if any(value.strip().lower() == "alma" for value in source_values if value):
        return "L"
    return "PC"


def build_record_url(record: PrimoRecord, config: PrimoConfig | None = None) -> str | None:
    """Build a direct Primo full-display URL for a record."""
    if config is None or not record.record_id or not config.vid:
        return None

    app_base = _discovery_app_base_url(config)
    if not app_base:
        return None

    params = urlencode(
        {
            "docid": record.record_id,
            "context": _record_context(record),
            "vid": config.vid,
            "lang": config.language,
        }
    )
    return f"{app_base}/fulldisplay?{params}"


record_link = build_record_url


def build_search_url(
    query: str,
    config: PrimoConfig | None = None,
    *,
    field: str = "any",
    scope: str = "everything",
    sort_by: str = "rank",
    offset: int = 0,
    resource_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    peer_reviewed: bool | None = None,
    include_unavailable: bool | None = None,
) -> str | None:
    """Build a direct Primo UI search URL for the search response."""
    if config is None or not config.vid:
        return None

    app_base = _discovery_app_base_url(config)
    scope_params = _search_scope_params(config, scope)
    if not app_base or scope_params is None:
        return None

    try:
        field = normalise_search_field(field)
        sort_by = normalise_sort_by(sort_by)
        resource_type = normalise_resource_type(resource_type)
        date_range = date_range_facet_value(date_from, date_to)
    except ValueError:
        return None

    tab, search_scope = scope_params
    if include_unavailable is None:
        include_unavailable = config.include_unavailable
    params: list[tuple[str, str]] = [
        ("query", f"{field},contains,{query}"),
        ("tab", tab),
        ("search_scope", search_scope),
        ("vid", config.vid),
        ("lang", config.language),
        ("offset", str(max(0, offset))),
        ("pcAvailability", "true" if include_unavailable else "false"),
    ]
    if sort_by:
        params.append(("sortby", sort_by))
    if resource_type:
        params.append(("facet", f"rtype,include,{resource_type}"))
    if date_range:
        params.append(("facet", f"searchcreationdate,include,{date_range}"))
    if peer_reviewed:
        params.append(("facet", "tlevel,include,peer_reviewed"))

    return f"{app_base}/search?{urlencode(params)}"


def _markdown_link_text(text: str) -> str:
    """Escape Markdown link text while preserving Unicode metadata."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _format_query_links(
    query: str,
    field: str,
    search_url: str | None,
    *,
    has_results: bool,
) -> list[str]:
    if not search_url:
        return []
    query_label = _markdown_link_text(f"{field},contains,{query}")
    result_label = "Results found" if has_results else "No results"
    return ["Queries run:", f"- {result_label}: [{query_label}]({search_url})", ""]


def _format_title(record: PrimoRecord, config: PrimoConfig | None = None) -> str:
    """Format a record title as a link when a Primo URL can be built."""
    url = build_record_url(record, config)
    if not url:
        return record.title
    return f"[{_markdown_link_text(record.title)}]({url})"


def format_search_results(
    response: SearchResponse,
    query: str = "",
    offset: int = 0,
    config: PrimoConfig | None = None,
    field: str = "any",
    scope: str = "everything",
    sort_by: str = "rank",
    resource_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    peer_reviewed: bool | None = None,
    include_unavailable: bool | None = None,
) -> str:
    """Format search results as a compact numbered list.

    Each result is 3-5 lines: title, authors, metadata, identifiers, availability.
    """
    search_url = build_search_url(
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
    )
    try:
        query_field = normalise_search_field(field)
    except ValueError:
        query_field = field
    query_links = _format_query_links(
        query,
        query_field,
        search_url,
        has_results=bool(response.records),
    )

    if not response.records:
        lines = [f'No results found for "{query}".', ""]
        lines.extend(query_links)
        lines.extend(
            [
                "Suggestions:",
                "- Broaden your search terms",
                "- Check spelling",
                "- Try a different search field (title, creator, subject)",
                "- Remove filters (resource type, date range)",
                "",
                "Iterative search guidance:",
                "- Reason about why this query returned zero results, then call primo_search again with a revised query.",
                "- Try up to five total attempts before concluding there are no good Primo results.",
                "- For dataset or data-source requests, start retries with catalogue databases (scope=\"catalogue\", resource_type=\"databases\") before expanding to articles or books.",
                "- Consider broader concepts, synonyms, related disciplines, singular/plural variants, alternate fields, relaxed filters, permitted scope widening, direct searches for likely database names, or OR queries for close alternatives.",
                "- When summarising, combine all relevant results found across attempts and report the attempted queries.",
            ]
        )
        return "\n".join(lines)

    total = f"{response.info.total:,}"
    showing_start = offset + 1
    showing_end = offset + len(response.records)

    lines = [
        f'Found {total} results for "{query}" (showing {showing_start}-{showing_end})',
        "",
    ]
    lines.extend(query_links)

    for i, record in enumerate(response.records, start=showing_start):
        # Line 1: number, title, year, type
        type_badge = record.resource_type.replace("_", " ").title() if record.resource_type else "Unknown"
        year = record.year or "n.d."

        lines.append(f"[{i}] {_format_title(record, config)}")
        lines.append(f"    {_format_authors(record.display_authors)} | {year} | {type_badge}")

        # Line 3: journal/source + identifiers
        source_parts = []
        if record.journal_title:
            journal_info = record.journal_title
            if record.volume:
                journal_info += f", {record.volume}"
            if record.issue:
                journal_info += f"({record.issue})"
            if record.start_page:
                journal_info += f", pp. {record.start_page}"
                if record.end_page:
                    journal_info += f"-{record.end_page}"
            source_parts.append(journal_info)
        elif record.publisher:
            source_parts.append(record.publisher)

        ident = _format_identifiers(record)
        if ident:
            source_parts.append(ident)
        if source_parts:
            lines.append(f"    {' | '.join(source_parts)}")

        subject_preview = _format_preview(record.subjects)
        if subject_preview:
            lines.append(f"    Subjects: {subject_preview}")

        keyword_preview = _format_preview(record.keywords)
        if keyword_preview:
            lines.append(f"    Keywords: {keyword_preview}")

        context_parts = []
        if record.language:
            context_parts.append(f"Language: {record.language}")
        if record.source_label:
            context_parts.append(f"Source: {record.source_label}")
        if context_parts:
            lines.append(f"    {' | '.join(context_parts)}")

        # Final line: availability + peer review + record ID
        status_parts = []
        if record.peer_reviewed:
            status_parts.append("Peer-reviewed")
        status_parts.append(f"Availability: {_format_availability(record)}")
        lines.append(f"    {' | '.join(status_parts)}")
        lines.append(f"    Record ID: {record.record_id}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_record_detail(record: PrimoRecord, config: PrimoConfig | None = None) -> str:
    """Format a single record with full details."""
    lines = []

    lines.append(f"Title: {_format_title(record, config)}")
    lines.append(f"Author(s): {_format_authors(record.display_authors, max_authors=10)}")

    # Show contributors only when they add names beyond those already listed
    # as authors (avoids duplication when contributors are the author fallback).
    extra_contributors = [c for c in record.contributors if c not in record.display_authors]
    if extra_contributors:
        lines.append(f"Contributor(s): {'; '.join(extra_contributors)}")

    year = record.year or "n.d."
    lines.append(f"Year: {year}")
    lines.append(f"Type: {record.resource_type.replace('_', ' ').title() if record.resource_type else 'Unknown'}")

    if record.publisher:
        lines.append(f"Publisher: {record.publisher}")

    if record.journal_title:
        journal = record.journal_title
        if record.volume:
            journal += f", vol. {record.volume}"
        if record.issue:
            journal += f", no. {record.issue}"
        if record.start_page:
            journal += f", pp. {record.start_page}"
            if record.end_page:
                journal += f"-{record.end_page}"
        lines.append(f"Journal: {journal}")

    if record.language:
        lines.append(f"Language: {record.language}")

    # Identifiers
    if record.doi:
        lines.append(f"DOI: {record.doi}")
    if record.isbn:
        lines.append(f"ISBN: {', '.join(record.isbn)}")
    if record.issn:
        lines.append(f"ISSN: {', '.join(record.issn)}")

    if record.subjects:
        lines.append(f"Subjects: {'; '.join(record.subjects)}")
    if record.keywords:
        lines.append(f"Keywords: {'; '.join(record.keywords)}")

    lines.append(f"Peer-reviewed: {'Yes' if record.peer_reviewed else 'No'}")

    if record.description:
        # Truncate long descriptions
        desc = record.description
        if len(desc) > 500:
            desc = desc[:497] + "..."
        lines.append(f"\nDescription:\n{desc}")

    # Availability
    lines.append(f"\nAvailability: {_format_availability(record)}")
    if record.source_label:
        lines.append(f"Source: {record.source_label}")

    lines.append(f"Record ID: {record.record_id}")

    return "\n".join(lines)


def format_suggestions(suggestions: list[str], query: str) -> str:
    """Format autocomplete suggestions."""
    if not suggestions:
        return f'No suggestions found for "{query}".'

    lines = [f'Suggestions for "{query}":', ""]
    for s in suggestions:
        lines.append(f"  - {s}")
    return "\n".join(lines)
