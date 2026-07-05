"""Configuration for the Primo MCP server."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class PrimoConfig(BaseSettings):
    """Primo API configuration.

    Defaults are set for SMU (Singapore Management University).
    Override via environment variables with the PRIMO_ prefix,
    or via a .env file in the working directory.
    """

    model_config = SettingsConfigDict(env_prefix="PRIMO_", env_file=".env")

    # Institution-specific
    base_url: str = "https://search.library.smu.edu.sg/primaws/rest/pub"
    discovery_base_url: str | None = None
    vid: str = "65SMU_INST:SMU_NUI"
    # Institution code for the guest JWT endpoint. Derived from the part of
    # vid before the colon when not set explicitly.
    institution_code: str | None = None
    institution_name: str = "SMU"
    tab_everything: str = "Everything"
    tab_catalogue: str = "Catalogue"
    tab_books_videos: str = "booksandvideos"
    scope_combined: str = "MyInst_and_CI"
    scope_local: str = "MyInstitution"
    scope_books_videos: str = "BooksVideos"

    # Operational
    request_timeout: float = 30.0
    max_results_per_request: int = 50
    default_results: int = 10
    language: str = "en"
    user_agent: str = "primo-mcp-server/0.1.0"
    # Default for the Primo pcAvailability search parameter. When False,
    # CDI (Central Discovery Index) results are restricted to material the
    # institution has full text access to; when True the search is
    # "expanded" and includes records with no access. False is the safer
    # default for holdings-confirmation queries.
    include_unavailable: bool = False
    # Fetch the facet summary (a second, cheap /facets request) after each
    # search and append a "Result landscape" section to search output so
    # callers can refine queries from data instead of guessing. Primo only
    # serves facets for the combined (Everything) scope; other scopes
    # return none and the section is simply omitted.
    search_facets: bool = True

    # Optional external JSON directory used for librarian recommendations.
    # No real profile data is bundled; local installs opt in by setting this.
    librarians_file: str | None = None
    inline_librarian_recommendations: bool = True
    librarian_min_score: float = 5.0
    # Opt-in JSONL log of recommendation outcomes (query, status, match and
    # near-miss ids with scores). Exists to close the tuning loop: real
    # queries that matched wrongly or missed can be triaged into the golden
    # eval set instead of being lost. Local file, appended per outcome;
    # write failures never affect the recommendation itself.
    recommend_log_file: str | None = None

    # Optional semantic (embedding) fallback for librarian recommendations.
    # Consulted when the deterministic keyword matcher finds no match, or when
    # its best match scores below librarian_semantic_second_guess_score.
    # Opt in by setting librarian_semantic_fallback=true and providing a
    # Gemini API key. Defaults target Google's gemini-embedding-001 free tier.
    librarian_semantic_fallback: bool = False
    # "gemini" calls Google's hosted API (rate-limited free tier); "local"
    # calls an OpenAI-compatible /embeddings endpoint such as Ollama,
    # LM Studio, or a llama.cpp server, with no quota at all. A directory of
    # up to ~30 profiles embeds once in seconds on CPU and is cached; only
    # one query embedding is computed per search, so small local models are
    # entirely sufficient. When switching providers or models, re-run
    # calibrate_embeddings: the cosine floor below was tuned for
    # gemini-embedding-001 and other models have different similarity
    # distributions (the mean+margin rule self-calibrates, the floor does not).
    embedding_provider: str = "gemini"
    embedding_api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embedding_model: str = "gemini-embedding-001"
    embedding_api_key: str | None = None
    # Settings for embedding_provider="local". The defaults target Ollama
    # running EmbeddingGemma (`ollama pull embeddinggemma`), Google's 300M
    # local embedder; any OpenAI-compatible endpoint and model works. The
    # prefixes are EmbeddingGemma's recommended task prompts and stand in
    # for Gemini's taskType parameter -- for nomic-embed-text use
    # "search_query: " and "search_document: "; set both empty when the
    # serving runtime applies its own prompt template.
    embedding_local_url: str = "http://localhost:11434/v1"
    embedding_local_model: str = "embeddinggemma"
    embedding_local_query_prefix: str = "task: search result | query: "
    embedding_local_document_prefix: str = "title: none | text: "
    # Absolute cosine sanity floor. gemini-embedding-001 is anisotropic
    # (unrelated text sits near ~0.5), so this floor alone is fragile across
    # directory sizes; the self-calibrating margin below does the real work
    # once the directory has enough profiles.
    librarian_semantic_min_similarity: float = 0.60
    # Self-calibrating acceptance margin: a profile is accepted only when its
    # similarity exceeds the mean similarity of all profiles by this margin.
    # Applied only when at least librarian_semantic_margin_min_profiles
    # profiles are scored (the mean is noise for tiny directories).
    librarian_semantic_margin: float = 0.08
    librarian_semantic_margin_min_profiles: int = 4
    # Below the margin's profile minimum, the top profile must instead lead
    # the runner-up by this cosine gap (plus clear the absolute floor), and
    # only the top-1 is returned. A single-profile directory has no relative
    # signal at all and falls back to the absolute floor alone.
    librarian_semantic_min_top_gap: float = 0.05
    # Minimum topical (non-stopword, non-filler) query tokens before the
    # semantic fallback runs. One-word and filler-only queries are where
    # cosine over bag-of-terms profiles is least reliable; skipping them
    # also avoids the embedding call entirely. 0 or 1 disables the gate.
    librarian_semantic_min_query_tokens: int = 2
    # Keyword matches scoring below this are "second-guessed": the semantic
    # path also runs and may append additional candidates. Set to 0 to only
    # run the semantic fallback on a strict keyword miss (old behaviour).
    librarian_semantic_second_guess_score: float = 12.0
    # Optional Matryoshka truncation (e.g. 768) to cut cache size and latency.
    # gemini-embedding-001 degrades little when truncated; cosine scoring
    # renormalises, so no extra normalisation step is needed. Changing this
    # invalidates cached profile embeddings.
    embedding_dimensions: int | None = None
    # Where profile embeddings are cached. Defaults to a sibling of
    # librarians_file (e.g. librarian-profile-embeddings.json).
    embedding_cache_file: str | None = None
    embedding_timeout: float = 10.0
    # Tighter budget for the inline primo_search path, so a slow embedding
    # call cannot add the full embedding_timeout to every ordinary search.
    # The explicit primo_recommend_librarians tool keeps the full budget.
    embedding_inline_timeout: float = 2.5
    # Rate-limit resilience: on HTTP 429 an embedding call waits (honouring
    # the server's Retry-After header or RetryInfo body when present, capped
    # at embedding_retry_max_delay) and retries up to this many times before
    # failing closed. Free-tier quotas replenish per minute, so a cap below
    # 60 would make honouring the server's advice pointless. Retries never
    # run on the inline primo_search path, which has a hard latency budget;
    # a search fails closed fast rather than sleeping.
    embedding_retry_attempts: int = 3
    embedding_retry_max_delay: float = 65.0
