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
- For catalogue searches with no results, retry with `scope="everything"` only when the user did not ask for catalogue-only results.
- For access or subscription checks, use Primo results as the evidence source rather than websites or LibGuides.

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
| `PRIMO_LIBRARIANS_FILE` | unset | External JSON librarian directory used for recommendations |
| `PRIMO_INLINE_LIBRARIAN_RECOMMENDATIONS` | `true` | Append a bottom `Recommended librarian help:` section to `primo_search` output |
| `PRIMO_LIBRARIAN_MIN_SCORE` | `5.0` | Minimum deterministic match score required before showing a recommendation |
| `PRIMO_LIBRARIAN_SEMANTIC_FALLBACK` | `false` | Enable the embedding fallback used only when keyword matching finds no librarian |
| `PRIMO_EMBEDDING_API_KEY` | unset | Google Gemini API key for the semantic fallback |
| `PRIMO_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model for the semantic fallback |
| `PRIMO_EMBEDDING_API_URL` | `https://generativelanguage.googleapis.com/v1beta` | Embedding API base URL |
| `PRIMO_LIBRARIAN_SEMANTIC_MIN_SIMILARITY` | `0.65` | Minimum cosine similarity for a semantic recommendation |
| `PRIMO_EMBEDDING_CACHE_FILE` | next to `PRIMO_LIBRARIANS_FILE` | Where profile embeddings are cached |
| `PRIMO_EMBEDDING_TIMEOUT` | `10.0` | HTTP timeout for embedding requests in seconds |

See `.env.example` for a commented template.

### Semantic fallback (optional)

Keyword matching is exact (after light stemming), so a query whose wording
doesn't overlap any profile term returns no recommendation. Enabling
`PRIMO_LIBRARIAN_SEMANTIC_FALLBACK=true` adds an embedding-based fallback that
runs **only when keyword matching finds nothing**, so embeddings are computed
only on misses. It uses Google's `gemini-embedding-001` (free tier — get a key
at <https://aistudio.google.com/apikey>). Profile embeddings are cached to a
sidecar file and recomputed only when a profile or the model changes. The layer
fails closed: any embedding error degrades silently to the normal `no_match`
result, and only configured profiles are ever returned. Semantic matches are
labelled `Status: matched (semantic fallback)` to signal lower confidence.

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
