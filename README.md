# Primo MCP Server

MCP server for Singapore Management University library discovery via Ex
Libris Primo. It searches SMU catalogue records, subscribed databases,
articles, books, videos, and database records through the Primo API.

This fork includes hardened scope handling, direct Primo record/search links,
and Unicode-safe handling for Chinese records.

## Features

- Search Primo catalogue, Primo Central Index, books/videos scopes, and subscribed databases
- Always return direct Primo search links and individual record links
- Get record details including title, authors, identifiers, subjects, description, source, availability, and record ID
- Preserve Chinese and other Unicode metadata in search results, details, citations, and exports
- Generate citations in APA 7th, Harvard, Chicago, IEEE, and Vancouver styles
- Export records to BibTeX, RIS, or UTF-8-sig CSV
- Reject invalid search scopes instead of silently falling back to Everything
- Recommend configured SMU librarians from search queries and Primo metadata
- Append a "Result landscape" facet summary (resource types, top subjects, creators, journals, languages, availability, publication years) so zero-result and too-many-result searches can be refined from data instead of guesswork
- Compound boolean queries: multi-clause AND/OR/NOT with contains/exact/begins_with operators for known-item lookups (title AND creator), exact-title checks, and OR expansion
- Show physical shelf locations (library, location, call number, availability status) and direct full-text access links (proxied resource links, Alma link-resolver openurl) in search results and record details

## Quick Start for SMU

Clone and install the fork:

```bash
git clone https://github.com/aarontaycheehsien/primo-mcp-server.git
cd primo-mcp-server
pip install -e .
```

Register it in Claude Code by adding this to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "primo": {
      "command": "python",
      "args": ["-m", "primo_mcp_server"]
    }
  }
}
```

Restart Claude Code. The tools will appear as `mcp__primo__primo_search`,
`mcp__primo__primo_get_record`, and related tool names.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tools

| Tool | Description |
|------|-------------|
| `primo_search` | Search Primo with field, scope, type, date, and peer-review filters |
| `primo_get_record` | Get full details for a record by Primo record ID |
| `primo_suggest` | Get autocomplete suggestions |
| `primo_recommend_librarians` | Recommend configured librarian help for a query or selected records |
| `primo_list_librarians` | List every configured librarian profile with contact and coverage |
| `primo_cite` | Generate formatted citations |
| `primo_export` | Export records as BibTeX, RIS, or CSV |

## Scope Behaviour

Use these canonical scopes:

| Scope | Covers | Common aliases |
|-------|--------|----------------|
| `catalogue` | Local catalogue records, including books, databases, and videos | `catalog`, `local`, `myinstitution`, `my_institution` |
| `everything` | Local catalogue plus Primo Central Index articles and remote records | `all`, `combined`, `myinst_and_ci`, `pci` |
| `books_videos` | Institution books/videos tab where configured | `booksvideos`, `booksandvideos`, `books/videos`, `books & videos` |

Recommended caller policy:

- For books, databases, and videos, start with `scope="catalogue"`.
- For articles, start with `scope="everything"`.
- For dataset or data-source requests, start with `scope="catalogue"` and `resource_type="databases"` to find subscribed data platforms first. Expand to articles or books only after database results are weak, irrelevant, or empty, and say that the search was expanded beyond databases.
- For catalogue searches with no results, retry with `scope="everything"` only when the user did not ask for catalogue-only results.
- For any zero-result search, reason about why the query failed and try revised `primo_search` calls up to five total attempts. Good retries may broaden an over-specific phrase, use synonyms or related concepts, try singular/plural variants, switch fields, relax filters, widen scope when permitted, search directly for likely database names, or use OR queries for close alternatives.
- When summarising an iterative search, combine all relevant results found across attempts and report the attempted queries.
- For access or subscription checks, use Primo results as the evidence source rather than websites or LibGuides.

## Compound Queries

`primo_search` accepts an optional `clauses` list that compiles to Primo's
multi-clause boolean syntax and replaces the single `query`/`field` pair as
the retrieval query (`query` should still carry a short plain-text summary;
it drives librarian recommendations and display). Each clause has a `value`,
optional `field` (`any`, `title`, `creator`, `sub`, `isbn`, `issn`,
`oclcnum`), optional `operator` (`contains`, `exact`, `begins_with`), and
optional `connector` (`AND`, `OR`, `NOT`) joining it to the next clause:

```json
{
  "query": "piketty capital",
  "clauses": [
    {"field": "title", "value": "capital", "connector": "AND"},
    {"field": "creator", "value": "piketty"}
  ]
}
```

Use compound queries for known-item lookups (title AND creator), exact-title
subscription checks, genuine OR expansion across synonyms, or NOT exclusion.
The result header links to the equivalent Primo advanced search.

## SMU Configuration

SMU is the default configuration for this fork. You can run without a `.env`
file for SMU, or create one to make the settings explicit:

```env
PRIMO_BASE_URL=https://search.library.smu.edu.sg/primaws/rest/pub
PRIMO_DISCOVERY_BASE_URL=https://search.library.smu.edu.sg/discovery
PRIMO_VID=65SMU_INST:SMU_NUI
PRIMO_INSTITUTION_NAME=SMU

PRIMO_TAB_EVERYTHING=Everything
PRIMO_TAB_CATALOGUE=Catalogue
PRIMO_TAB_BOOKS_VIDEOS=booksandvideos

PRIMO_SCOPE_COMBINED=MyInst_and_CI
PRIMO_SCOPE_LOCAL=MyInstitution
PRIMO_SCOPE_BOOKS_VIDEOS=BooksVideos

PRIMO_LANGUAGE=en
PRIMO_REQUEST_TIMEOUT=30.0
PRIMO_MAX_RESULTS_PER_REQUEST=50
PRIMO_DEFAULT_RESULTS=10
PRIMO_LIBRARIANS_FILE=C:\path\to\smu-librarians.json
PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS=true
PRIMO_LIBRARIAN_MIN_SCORE=5.0
```

## Configuration Reference

Defaults are set for SMU. Other institutions can override these values with
environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMO_BASE_URL` | `https://search.library.smu.edu.sg/primaws/rest/pub` | Primo API base URL |
| `PRIMO_DISCOVERY_BASE_URL` | Derived from `PRIMO_BASE_URL` | Primo web app base URL for record and search links |
| `PRIMO_VID` | `65SMU_INST:SMU_NUI` | Primo view ID |
| `PRIMO_INSTITUTION_CODE` | Derived from `PRIMO_VID` | Institution code for the guest JWT endpoint |
| `PRIMO_INSTITUTION_NAME` | `SMU` | Display name |
| `PRIMO_TAB_EVERYTHING` | `Everything` | Primo tab for combined local and CDI searches |
| `PRIMO_TAB_CATALOGUE` | `Catalogue` | Primo tab for local catalogue searches |
| `PRIMO_TAB_BOOKS_VIDEOS` | `booksandvideos` | Primo tab for books/videos searches |
| `PRIMO_SCOPE_COMBINED` | `MyInst_and_CI` | Primo scope for combined local and CDI searches |
| `PRIMO_SCOPE_LOCAL` | `MyInstitution` | Primo scope for local catalogue searches |
| `PRIMO_SCOPE_BOOKS_VIDEOS` | `BooksVideos` | Primo scope for books/videos searches |
| `PRIMO_REQUEST_TIMEOUT` | `30.0` | HTTP timeout in seconds |
| `PRIMO_MAX_RESULTS_PER_REQUEST` | `50` | Maximum results per search request |
| `PRIMO_DEFAULT_RESULTS` | `10` | Default results per search |
| `PRIMO_LANGUAGE` | `en` | Primo language parameter |
| `PRIMO_INCLUDE_UNAVAILABLE` | `false` | Include CDI records without full text access in search results |
| `PRIMO_SEARCH_FACETS` | `true` | Fetch the facet summary after each search and append a "Result landscape" section (facets are only served for the Everything scope; other scopes omit the section) |
| `PRIMO_LIBRARIANS_FILE` | unset | External JSON librarian directory used for recommendations |
| `PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS` | `true` | Append a bottom `Recommended librarian help:` section to `primo_search` output |
| `PRIMO_LIBRARIAN_MIN_SCORE` | `5.0` | Minimum deterministic match score required before showing a recommendation |
| `PRIMO_RECOMMEND_LOG_FILE` | unset | Opt-in JSONL log of recommendation outcomes (query, status, match/near-miss ids and scores) for triaging real queries into the golden eval set |
| `PRIMO_LIBRARIAN_SEMANTIC_FALLBACK` | `false` | Enable the embedding path used when keyword matching finds nothing or matches weakly |
| `PRIMO_EMBEDDING_API_KEY` | unset | Google Gemini API key for the semantic fallback |
| `PRIMO_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for the semantic fallback |
| `PRIMO_EMBEDDING_API_URL` | `https://generativelanguage.googleapis.com/v1beta` | Embedding API base URL |
| `PRIMO_LIBRARIAN_SEMANTIC_MIN_SIMILARITY` | `0.60` | Absolute cosine floor for a semantic recommendation |
| `PRIMO_LIBRARIAN_SEMANTIC_MARGIN` | `0.08` | Self-calibrating margin: a match must exceed the mean similarity across all profiles by this much |
| `PRIMO_LIBRARIAN_SEMANTIC_MARGIN_MIN_PROFILES` | `4` | Directory size at which the margin rule starts applying |
| `PRIMO_LIBRARIAN_SEMANTIC_MIN_TOP_GAP` | `0.05` | Below the margin's profile minimum, the top match must lead the runner-up by this cosine gap (top-1 only) |
| `PRIMO_LIBRARIAN_SEMANTIC_MIN_QUERY_TOKENS` | `2` | Skip the semantic fallback (no embedding call) for queries with fewer topical words; `1` disables the gate |
| `PRIMO_LIBRARIAN_SEMANTIC_SECOND_GUESS_SCORE` | `12.0` | Keyword scores below this are second-guessed by the semantic path (`0` = strict miss-only cascade) |
| `PRIMO_EMBEDDING_DIMENSIONS` | unset | Optional Matryoshka truncation (e.g. `768`) to cut cache size and latency |
| `PRIMO_EMBEDDING_CACHE_FILE` | next to `PRIMO_LIBRARIANS_FILE` | Where profile embeddings are cached |
| `PRIMO_EMBEDDING_TIMEOUT` | `10.0` | HTTP timeout for embedding requests in seconds |
| `PRIMO_EMBEDDING_INLINE_TIMEOUT` | `2.5` | Tighter embedding budget for inline `primo_search` recommendations |
| `PRIMO_EMBEDDING_RETRY_ATTEMPTS` | `3` | How many times an HTTP 429 is waited out and retried (never on the inline path) |
| `PRIMO_EMBEDDING_RETRY_MAX_DELAY` | `65.0` | Cap in seconds on the wait honoured from the server's `Retry-After`/`RetryInfo` advice |

See `.env.example` for a commented template.

### Semantic fallback (optional)

Keyword matching is exact (after light stemming), so a query whose wording
doesn't overlap any profile term returns no recommendation. Enabling
`PRIMO_LIBRARIAN_SEMANTIC_FALLBACK=true` adds an embedding-based path that runs
when keyword matching **finds nothing or matches only weakly** (best score
below `PRIMO_LIBRARIAN_SEMANTIC_SECOND_GUESS_SCORE`), so embeddings are
computed only when keywords are unconvincing. Keyword matches stay primary and
are never displaced; a passing semantic candidate for a different librarian is
appended within the limit. It uses Google's `gemini-embedding-001` (free tier —
get a key at <https://aistudio.google.com/apikey>). Each profile term is
embedded as its own vector and a profile scores by its best term (max cosine),
so a sharp hit on one configured topic is never averaged away by the rest of a
large profile. Terms are embedded in batched `batchEmbedContents` requests,
cached to a sidecar file keyed by term content (terms shared by several
profiles are embedded once), and recomputed only when a term, the model, or
the output dimensionality changes. The cache is written after every batch,
so a rate-limited cold rebuild keeps its progress; rate-limit responses
(HTTP 429) are waited out and retried, honouring the API's own
`Retry-After`/`RetryInfo` advice, except on the latency-bounded inline
`primo_search` path, which fails closed fast instead of sleeping.

Acceptance is self-calibrating rather than a single tuned constant, with
three regimes by directory size: with at least
`PRIMO_LIBRARIAN_SEMANTIC_MARGIN_MIN_PROFILES` profiles the top matches must
exceed the mean similarity across all profiles by
`PRIMO_LIBRARIAN_SEMANTIC_MARGIN`; smaller directories accept only the top
profile and only when it leads the runner-up by
`PRIMO_LIBRARIAN_SEMANTIC_MIN_TOP_GAP`; a single-profile directory falls back
to the absolute cosine floor alone. Queries with fewer than
`PRIMO_LIBRARIAN_SEMANTIC_MIN_QUERY_TOKENS` topical words (stopwords and
filler words don't count) skip the semantic path entirely -- short or vague
queries are where cosine similarity is least reliable, and the skip happens
before any embedding call is made. Skips are reported in the output the same
way errors are, so they are never mistaken for a genuine no-match. To set the
floor, margin, and gap empirically for your own directory, print the
similarity distribution for representative test queries:

```bash
python -m primo_mcp_server.calibrate_embeddings "systematic review screening" "GIS data for urban planning"
```

The layer fails closed — only configured profiles are ever returned, and any
embedding error degrades to the keyword outcome — but not silently: errors are
logged to stderr and surfaced in the output as a
`semantic fallback errored` note so an invalid API key is distinguishable from
a genuine no-match. Semantic matches are labelled
`Status: matched (semantic fallback)` and report their cosine similarity so
callers can reason about confidence. Identifier-shaped queries (DOIs, ISBNs,
ISSNs, Alma/CDI record ids) skip librarian recommendations entirely on both
paths.

When `PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS=true` and a configured profile
meets the score threshold, `primo_search` appends a bottom Markdown section
headed `## Recommended librarian help:`. Callers should preserve this section when
summarising Primo results.

The recommendation display uses a fixed labelled format for each matched
profile:

```text
## Recommended librarian help:

Status: matched
1. Name: [Accounting Librarian](https://library.smu.edu.sg/example-profile)
   Title: Business Research Librarian
   Contact: accounting@example.edu
   Best for: Consult for accounting datasets, WRDS, and Compustat.
   Evidence: matched terms: accounting; evidence fields: query
Recommendations are limited to configured librarian profiles; do not invent or substitute names.
```

The `Name` value is always emitted as a Markdown link. The profile `url` is
used first; if it is missing, the formatter falls back to a `mailto:` link
when an email address is configured.

When no recommendation clears the confidence threshold, the `no_match`
output still shows the closest below-threshold profiles WITH their matching
evidence, explicitly labelled as not validated -- so any librarian a caller
passes on to the user always carries evidence, and a weak candidate can
never be silently presented as a confident match. When nothing matched even
weakly, the output directs callers to `primo_list_librarians`, which
returns the complete configured directory (name, title, contact, schools,
best-for areas, and a sample of subjects) so a caller can still route the
user to a real contact without inventing one.

Librarian recommendations require an external JSON file. No real profiles are
bundled. The minimum shape is:

```json
{
  "librarians": [
    {
      "id": "accounting",
      "name": "Accounting Librarian",
      "title": "Business Research Librarian",
      "email": "accounting@example.edu",
      "url": "https://library.smu.edu.sg/example-profile",
      "subjects": ["accounting", "audit fees"],
      "keywords": ["corporate governance"],
      "aliases": ["financial reporting"],
      "best_for": ["accounting datasets", "WRDS", "Compustat"],
      "schools": ["School of Accountancy"],
      "resource_types": ["databases"],
      "notes": "Consult for accounting and audit research."
    }
  ]
}
```

### Maintaining the profile directory

The `primo-profiles` CLI keeps the JSON directory reproducible from a CSV
source and reports curation problems that weaken matching:

```bash
# Build the JSON directory from a CSV source (semicolon- or comma-separated
# multi-value cells; accepts singular or plural column headers)
python -m primo_mcp_server.profile_tools convert librarian-profile.csv librarian-profile.json

# Check the configured directory (or an explicit path) for problems:
# filler-only terms, term variants that normalise identically, terms listed
# by nearly every profile, unmatchable profiles, missing contact details,
# and deny-list terms broad enough to always fire
python -m primo_mcp_server.profile_tools lint
```

`lint` exits 0 when clean, 1 with findings, and 2 when the directory cannot
be read, so it can gate a profile-update workflow.

### Measuring recommendation accuracy

The `primo-eval` CLI benchmarks the recommendation pipeline against a golden
set of labelled queries, so tuning changes to weights, thresholds, or the
semantic path are judged by a measured delta instead of anecdote:

```bash
python -m primo_mcp_server.evaluate_recommendations librarian-eval.json --keyword-only
```

The eval file lists cases of the form
`{"query": "...", "expect": ["librarian-id"]}`; an empty `expect` means the
correct outcome is no recommendation (these cases measure false positives).
Cases can pin `records` metadata for deterministic corroboration evidence.
The report gives top-1 accuracy, hit rate within the returned list, and the
correct-rejection rate. `--keyword-only` forces the deterministic path;
without it the semantic fallback runs exactly when the server would run it.
`--min-pass-rate 0.9` turns the run into a regression gate (exit 1 below the
threshold). The eval runs the same pipeline module the server uses, so its
numbers are statements about real server behaviour.

## Usage Examples

From a Claude Code conversation:

- "Search the catalogue for books on poverty in Singapore"
- "Search Everything for peer-reviewed articles on open access citation advantage"
- "Do we have access to JSTOR?"
- "search for databases with data on cost of living"
- "Get the full details for record alma991234567890"
- "Recommend a librarian for this accounting search"
- "Generate APA7 citations for these records"
- "Export these records as BibTeX"

## Licence

MIT
