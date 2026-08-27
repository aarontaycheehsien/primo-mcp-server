# Primo MCP Server

SMU-focused MCP server for searching Singapore Management University Library
Primo catalogue, articles, databases, books, videos, and records through the Ex
Libris Primo discovery API.

This is the canonical agent guidance file for this fork.

## Architecture

- **Framework:** FastMCP (mcp.server.fastmcp)
- **Transport:** stdio
- **HTTP client:** httpx (async, connection-pooled)
- **Config:** pydantic-settings with PRIMO_ env prefix

## Key Files

- `src/primo_mcp_server/server.py` -- MCP tool definitions and lifespan
- `src/primo_mcp_server/client.py` -- Primo API HTTP client
- `src/primo_mcp_server/config.py` -- pydantic-settings configuration (PRIMO_ env prefix)
- `src/primo_mcp_server/query.py` -- scope, field, sort, and resource type alias normalisation
- `src/primo_mcp_server/models.py` -- Pydantic models for PNX response normalisation
- `src/primo_mcp_server/formatter.py` -- Compact text output for LLM context
- `src/primo_mcp_server/citations.py` -- Citation formatting (APA7, Harvard, Chicago, IEEE, Vancouver)
- `src/primo_mcp_server/exporters.py` -- BibTeX, RIS, CSV export
- `src/primo_mcp_server/librarians.py` -- Librarian directory loading and keyword recommendation matching
- `src/primo_mcp_server/librarian_embeddings.py` -- Optional Gemini embedding semantic fallback for recommendations (per-term vectors, max cosine per profile)
- `src/primo_mcp_server/calibrate_embeddings.py` -- CLI for calibrating semantic fallback thresholds
- `src/primo_mcp_server/profile_tools.py` -- Curator CLI: convert a CSV profile source to JSON and lint the directory
- `src/primo_mcp_server/recommendation.py` -- Combined keyword + semantic recommendation pipeline (shared by the server and the evaluation harness)
- `src/primo_mcp_server/evaluate_recommendations.py` -- CLI benchmark: golden labelled queries against the recommendation pipeline

## Running Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

## Configuration

Defaults are SMU (Singapore Management University). Other institutions can
override these values with environment variables, but public documentation and
agent behaviour should remain SMU-first for this fork.

- PRIMO_BASE_URL -- Primo API base URL
- PRIMO_DISCOVERY_BASE_URL -- Primo web app base URL for search and record links
- PRIMO_VID -- View ID for the institution
- PRIMO_INSTITUTION_NAME -- Display name
- PRIMO_TAB_CATALOGUE / PRIMO_SCOPE_LOCAL -- SMU catalogue search
- PRIMO_TAB_EVERYTHING / PRIMO_SCOPE_COMBINED -- SMU catalogue plus CDI search
- PRIMO_TAB_BOOKS_VIDEOS / PRIMO_SCOPE_BOOKS_VIDEOS -- SMU books/videos search

Librarian recommendations (see `config.py` for the full list, including
score thresholds, margins, timeouts, and the query token gate):

- PRIMO_LIBRARIANS_FILE -- path to the JSON librarian profile directory.
  No real profile data is bundled; local installs opt in by setting this.
- PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS -- append a "Recommended librarian
  help:" section to primo_search results (default true)
- PRIMO_LIBRARIAN_MIN_SCORE -- keyword match acceptance threshold
- PRIMO_LIBRARIAN_SEMANTIC_FALLBACK -- enable the Gemini embedding fallback
  (default false; requires an API key)
- PRIMO_EMBEDDING_API_KEY -- Gemini API key for the semantic fallback
- PRIMO_EMBEDDING_MODEL / PRIMO_EMBEDDING_API_URL -- embedding endpoint
  (defaults target gemini-embedding-001)

## Search Scope Policy

Use Primo as the evidence source for library holdings, subscriptions, and
access checks. Do not use websites, LibGuides, or general web pages as
evidence for those confirmation requests unless the user explicitly asks for
web confirmation.

When asked to search the catalogue, call `primo_search` with
`scope="catalogue"` first. If that returns no results and the user did not
ask for catalogue-only results, retry with `scope="everything"` and say that
the search was widened.

For books, databases, and videos, default to `scope="catalogue"`. For
articles, default to `scope="everything"`.

For dataset or data-source requests, start with `scope="catalogue"` and
`resource_type="databases"` to find subscribed data platforms first. Expand
to articles or books only after database results are weak, irrelevant, or
empty, and say that the search was expanded beyond databases.

## Librarian Recommendation Policy

Recommendations are validated against the configured JSON profile
directory. Only configured librarian names may be returned; never invent
or substitute names. `primo_recommend_librarians` is the explicit tool;
`primo_search` appends inline recommendations by default (suppress with
`recommend_librarians=false`); `primo_list_librarians` returns the complete
configured directory when no recommendation clears the threshold or when
the user asks who the librarians are. Deterministic keyword matching runs first,
with an optional Gemini embedding fallback when keyword matches are weak
or absent. Identifier-shaped queries (DOI, ISBN, ISSN, record IDs) skip
recommendations entirely. Recommendation counts are capped at 3.

When tuning matching weights or thresholds, run the golden-query benchmark
before and after and report the delta:

```bash
python -m primo_mcp_server.evaluate_recommendations librarian-eval.json --keyword-only
```

## Iterative Search Recovery

Treat a search as final as soon as it returns results relevant to the user's
request. If it returns no results or clearly irrelevant results, make up to
six `primo_search` calls in total, stopping early when relevant results appear:

1. Search the original query in the policy-selected scope.
2. Widen a catalogue search to `scope="everything"` when the user did not
   require catalogue-only results.
3. Call `primo_suggest` and search a plausible suggestion when one is
   available.
4. Search one high-confidence spelling correction and disclose the correction.
5. Search a simplified query, removing only filters that the agent inferred.
6. Search one close synonym or related-term variant.

Skip inapplicable steps rather than manufacturing a variant. Preserve every
explicit user constraint, including catalogue-only scope, dates, formats,
peer-review requirements, and availability requirements. For holdings,
ownership, subscription, or access confirmation, do not use synonyms or
related items as evidence for the requested title. Clearly identify any result
found through a corrected title.

Set `recommend_librarians=false` on every `primo_search` call in this workflow,
including the final search attempt. After the final search, call
`primo_recommend_librarians` exactly once. Pass the final relevant record IDs
when records were found; otherwise use the best corrected or clarified query.
Summarise the attempted queries, scope changes, corrections, and final outcome
so exact results remain distinguishable from exploratory matches.

When recommending a librarian, always give the reason for that specific
recommendation. Preserve the server-provided `Reasoning` field for every
recommended librarian; never present a librarian recommendation without it.
Render every recommended librarian in the final user-facing response with this
exact two-line template:

```text
- [Name](profile-url) - <email@example.edu>
  Reason: Server-provided reasoning.
```

Repeat the complete block for each recommendation. Do not paraphrase, exchange,
or omit one librarian's reason. The project hooks capture matched Primo tool
output and prevent Claude Code or Codex from stopping until every captured
recommendation uses this template.

## Conventions

- Australian English (en-AU)
- UTF-8-sig for CSV exports
- No contractions in prose
- ASCII-only in generated content
