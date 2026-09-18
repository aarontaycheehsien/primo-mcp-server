"""Single source of truth for caller-facing search policy prose.

The scope-selection policy and the zero-result retry guidance are shown to
callers in three places: the MCP server instructions, the primo_search tool
description, and the zero-result output of format_search_results. All three
are composed from the line lists here so the copies cannot drift. README.md
and AGENTS.md mirror this policy for human readers; edit them together.
"""

from __future__ import annotations

SCOPE_POLICY_LINES = [
    'When asked to search the catalogue, use scope="catalogue" first. If '
    "that returns no results and the user did not ask for catalogue-only "
    'results, retry with scope="everything" and say that the search was '
    "widened.",
    'For books, databases, and videos, default to scope="catalogue".',
    'For articles, default to scope="everything".',
    "For dataset or data-source requests, first search subscribed databases "
    'with scope="catalogue" and resource_type="databases". Only after '
    "database results are weak, irrelevant, or empty should callers expand "
    "to articles or books, and they should state that expansion.",
    "For confirmation requests about whether the library has, owns, "
    "subscribes to, or provides access to a title, use Primo as the "
    "evidence source. Do not rely on websites, LibGuides, or general web "
    "pages unless the user explicitly asks for web confirmation.",
]

ZERO_RESULT_POLICY_LINES = [
    "When a search returns zero results, reason about why the query failed "
    "and call primo_search again with revised queries up to five total "
    "attempts.",
    "Try broader concepts, synonyms, related disciplines, singular/plural "
    "variants, alternate fields, relaxed filters, scope widening where the "
    "scope policy permits, direct searches for likely database names, or "
    "OR queries for close alternatives.",
    "Combine relevant results from all attempts and report the attempted "
    "queries when summarising.",
]

# The same retry policy, phrased for the moment a search has just come back
# empty. Shown by format_search_results under "Iterative search guidance:".
ZERO_RESULT_GUIDANCE_LINES = [
    "Reason about why this query returned zero results, then call "
    "primo_search again with a revised query.",
    "Try up to five total attempts before concluding there are no good "
    "Primo results.",
    "For dataset or data-source requests, start retries with catalogue "
    'databases (scope="catalogue", resource_type="databases") before '
    "expanding to articles or books.",
    "Consider broader concepts, synonyms, related disciplines, "
    "singular/plural variants, alternate fields, relaxed filters, permitted "
    "scope widening, direct searches for likely database names, or OR "
    "queries for close alternatives.",
    "When summarising, combine all relevant results found across attempts "
    "and report the attempted queries.",
]

LIBRARIAN_POLICY_TEXT = (
    "Librarian recommendations are limited to configured profile IDs; do "
    "not invent or substitute names."
)


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


SEARCH_POLICY_TEXT = (
    "Scope selection policy for callers:\n"
    + _bullets(SCOPE_POLICY_LINES)
    + "\n\nZero-result policy for callers:\n"
    + _bullets(ZERO_RESULT_POLICY_LINES)
)

SERVER_INSTRUCTIONS = (
    "Search Singapore Management University Library catalogue records, "
    "articles, databases, books, videos, and holdings via the Ex Libris "
    "Primo discovery API.\n\n"
    + SEARCH_POLICY_TEXT
    + "\n\nUse primo_search for queries (a single query string, or compound "
    "boolean clauses for known-item and precision searches), "
    "primo_get_record for full details, primo_suggest for autocomplete, "
    "primo_recommend_librarians for validated librarian recommendations, "
    "primo_list_librarians for the full configured librarian directory, "
    "primo_cite for citations, and primo_export for BibTeX/RIS/CSV export. "
    + LIBRARIAN_POLICY_TEXT
)

PRIMO_SEARCH_DESCRIPTION = (
    "Search Singapore Management University Library via Primo.\n\n"
    + SEARCH_POLICY_TEXT
    + """

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
        records the library has NO full text access to (Primo's "expanded"
        search). Default (false) restricts results to accessible material,
        which is what holdings and access confirmation requires. Only set
        true when the user explicitly wants to discover material beyond the
        library's collection, e.g. for interlibrary loan or comprehensive
        literature mapping.
    recommend_librarians: Set to false to suppress inline librarian
        recommendations for this search. Inline recommendations also
        require PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS=true. When shown,
        callers should include the bottom "Recommended librarian help:"
        section when summarising Primo results.
    librarian_limit: Number of librarian recommendations to include
        inline. Defaults to 2 and is capped at 3.
    facet_filters: Optional facet refinements as a {facet: value} object,
        e.g. {"topic": "Economics", "lang": "eng"}. Use facet names and
        values exactly as reported in the "Result landscape" section of a
        previous search. Common facets: rtype, topic, creator, jtitle,
        lang, tlevel, library.
    facet_exclusions: Like facet_filters, but removes matching results
        (e.g. {"rtype": "reviews"} to drop book reviews).
    clauses: Optional compound boolean query. Each clause has a value,
        optional field (any, title, creator, sub, isbn, issn, oclcnum),
        optional operator (contains, exact, begins_with), and optional
        connector (AND, OR, NOT) joining it to the NEXT clause. Use for
        precision needs a single query string cannot express: known-item
        lookups (title AND creator), exact-title subscription checks
        (title exact), or genuine OR expansion across synonyms. When
        given, clauses replace query/field as the retrieval query; still
        set query to a short plain-text summary of the intent (it drives
        librarian recommendations and display).

Returns:
    Formatted search results with title, authors, year, identifiers,
    availability, shelf locations and access links where known, a "Result
    landscape" facet summary when Primo serves facets, and any bottom
    "Recommended librarian help:" section.
"""
)
