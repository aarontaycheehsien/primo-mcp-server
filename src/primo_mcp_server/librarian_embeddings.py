"""Semantic (embedding) fallback for librarian recommendations.

This layer is consulted when the deterministic keyword matcher in
``librarians.recommend_librarians`` returns no match, or when its best match
scores below the second-guess threshold. It ranks the configured librarian
profiles by cosine similarity between a Gemini embedding of the query and
cached embeddings of each profile.

Design guarantees:
- Fails closed: any error (missing key, network failure, malformed response)
  returns an empty match list, so behaviour degrades to the keyword path's
  outcome -- the tool never errors because of this layer. Errors are logged
  to stderr and surfaced as a status so they are distinguishable from a
  genuine no-match.
- Only configured profiles are ever ranked or returned, so the
  anti-hallucination guardrail is preserved.
- Acceptance is self-calibrating: besides an absolute cosine floor, the top
  matches must exceed the mean similarity across all profiles by a margin,
  which adapts to the anisotropy of the embedding space and to directory
  size instead of trusting a single institution-tuned constant.
- Profile embeddings are cached to a sidecar file keyed by a content hash and
  the model id (plus output dimensionality), so the (paid/slow) document
  embeddings are computed once and re-used until a profile, the model, or the
  dimensionality changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple, Sequence

import httpx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarians import (
    _MAX_RECOMMENDATIONS,
    LibrarianDirectory,
    LibrarianMatch,
    LibrarianProfile,
    _content_token_count,
    is_excluded,
)
from primo_mcp_server.models import PrimoRecord

logger = logging.getLogger(__name__)

# (texts, task_type) -> one embedding vector per input text.
Embedder = Callable[[Sequence[str], str], Awaitable[list[list[float]]]]

_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"
_MAX_QUERY_CHARS = 2000
# batchEmbedContents accepts at most 100 requests per call.
_MAX_BATCH_SIZE = 100


class SemanticFallbackResult(NamedTuple):
    """Outcome of the semantic fallback.

    ``error`` carries a short, key-free description (exception class name)
    when the fallback failed, so callers can surface "semantic fallback
    errored" instead of a misleading "no match". ``skipped`` carries a reason
    when the fallback deliberately did not run (e.g. the query is too short
    to embed reliably); ``error`` and ``skipped`` are mutually exclusive.
    """

    matches: list[LibrarianMatch]
    error: str | None = None
    skipped: str | None = None


class ProfileSimilarity(NamedTuple):
    """One profile's cosine similarity to a query (for scoring and the CLI)."""

    similarity: float
    librarian: LibrarianProfile


def _profile_text(librarian: LibrarianProfile) -> str:
    """Build the topical document embedded for a librarian.

    Name and title are deliberately excluded -- they carry little topical
    signal and risk spurious matches (e.g. a query mentioning a person's
    name).
    """
    parts = [
        librarian.notes,
        *librarian.subjects,
        *librarian.aliases,
        *librarian.keywords,
        *librarian.best_for,
        *librarian.schools,
        *librarian.resource_types,
    ]
    return " | ".join(p.strip() for p in parts if p and p.strip())


def _query_text(query: str, records: list[PrimoRecord] | None) -> str:
    """Return the user query, length-bounded.

    Returned-record context is deliberately ignored for semantic fallback. It
    can contain incidental topics from search results that are not what the
    user is asking for, producing broad false-positive librarian suggestions.
    """
    return query[:_MAX_QUERY_CHARS]


def _model_key(config: PrimoConfig) -> str:
    """Cache key covering everything that changes the embedding space."""
    if config.embedding_dimensions:
        return f"{config.embedding_model}@{config.embedding_dimensions}"
    return config.embedding_model


def _hash(text: str, model_key: str) -> str:
    return hashlib.sha256(f"{model_key}\n{text}".encode("utf-8")).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cache_path(config: PrimoConfig) -> Path | None:
    if config.embedding_cache_file:
        return Path(config.embedding_cache_file).expanduser()
    if config.librarians_file:
        base = Path(config.librarians_file).expanduser()
        return base.with_name(base.stem + "-embeddings.json")
    return None


def _read_cache(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(
    path: Path | None,
    model_key: str,
    vectors: dict[str, list[float]],
    hashes: dict[str, str],
) -> None:
    if path is None:
        return
    data = {
        "model": model_key,
        "entries": {
            lib_id: {"hash": hashes.get(lib_id, ""), "vector": vec}
            for lib_id, vec in vectors.items()
        },
    }
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        # Cache is an optimisation; an unwritable path is non-fatal.
        pass


async def _gemini_embed(
    texts: Sequence[str],
    task_type: str,
    *,
    config: PrimoConfig,
    timeout: float | None = None,
) -> list[list[float]]:
    """Embed ``texts`` via the Gemini ``batchEmbedContents`` endpoint.

    All texts go out in a single request (chunked at the API's limit of 100),
    so a cold cache with a large directory costs one call instead of N
    concurrent ones that would trip free-tier rate limits. The API key is
    sent as an ``x-goog-api-key`` header rather than a URL query parameter so
    it does not leak into proxy or server logs.
    """
    if not config.embedding_api_key:
        raise RuntimeError("embedding_api_key is not configured")
    base = config.embedding_api_url.rstrip("/")
    model_path = f"models/{config.embedding_model}"
    url = f"{base}/{model_path}:batchEmbedContents"

    def request_for(text: str) -> dict:
        request: dict = {
            "model": model_path,
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        if config.embedding_dimensions:
            request["outputDimensionality"] = config.embedding_dimensions
        return request

    vectors: list[list[float]] = []
    async with httpx.AsyncClient(
        timeout=timeout if timeout is not None else config.embedding_timeout,
        headers={"x-goog-api-key": config.embedding_api_key},
    ) as client:
        for start in range(0, len(texts), _MAX_BATCH_SIZE):
            chunk = texts[start : start + _MAX_BATCH_SIZE]
            response = await client.post(
                url,
                json={"requests": [request_for(text) for text in chunk]},
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            vectors.extend(item["values"] for item in embeddings)
    return vectors


async def _load_or_build_profile_vectors(
    directory: LibrarianDirectory,
    config: PrimoConfig,
    embed: Embedder,
) -> dict[str, list[float]]:
    """Return one embedding per librarian, re-using a sidecar cache."""
    path = _cache_path(config)
    cache = _read_cache(path)
    model_key = _model_key(config)
    entries = cache.get("entries", {}) if cache.get("model") == model_key else {}

    vectors: dict[str, list[float]] = {}
    hashes: dict[str, str] = {}
    stale: list[tuple[str, str]] = []
    for librarian in directory.librarians:
        text = _profile_text(librarian)
        if not text:
            continue
        digest = _hash(text, model_key)
        hashes[librarian.id] = digest
        cached = entries.get(librarian.id)
        if cached and cached.get("hash") == digest and cached.get("vector"):
            vectors[librarian.id] = cached["vector"]
        else:
            stale.append((librarian.id, text))

    if stale:
        new_vectors = await embed([text for _, text in stale], _TASK_DOCUMENT)
        for (lib_id, _), vector in zip(stale, new_vectors):
            vectors[lib_id] = vector
        _write_cache(path, model_key, vectors, hashes)
    return vectors


async def score_profiles(
    directory: LibrarianDirectory,
    query: str,
    config: PrimoConfig,
    *,
    embedder: Embedder | None = None,
    timeout: float | None = None,
) -> list[ProfileSimilarity]:
    """Cosine similarity of every embeddable profile to ``query``, unsorted.

    Raises on embedding failure; ``semantic_fallback`` wraps this with the
    fail-closed handling, while the calibration CLI lets errors propagate.
    """
    embed = embedder or (
        lambda texts, task_type: _gemini_embed(
            texts, task_type, config=config, timeout=timeout
        )
    )
    profile_vectors = await _load_or_build_profile_vectors(directory, config, embed)
    if not profile_vectors:
        return []
    query_vector = (await embed([_query_text(query, None)], _TASK_QUERY))[0]
    return [
        ProfileSimilarity(_cosine(query_vector, vector), librarian)
        for librarian in directory.librarians
        if (vector := profile_vectors.get(librarian.id))
    ]


def _accepted(
    similarities: list[ProfileSimilarity], config: PrimoConfig
) -> list[ProfileSimilarity]:
    """Apply the acceptance rule, adapted to directory size.

    Three regimes, all above the absolute floor (which catches the degenerate
    case where the whole directory is off-topic but one profile is slightly
    less so):
    - Enough profiles for a meaningful mean: self-calibrating mean + margin.
    - Two or three profiles: the mean is noise, but relative ranking still
      informs -- accept only the top profile, and only when it leads the
      runner-up by a clear gap. Uniform similarity means the embedding
      space cannot separate the profiles, so nothing is returned.
    - One profile: no relative signal exists; the floor alone decides. This
      is the residual fixed-threshold case.
    """
    if not similarities:
        return []
    floor = config.librarian_semantic_min_similarity
    if len(similarities) >= config.librarian_semantic_margin_min_profiles:
        mean = sum(s.similarity for s in similarities) / len(similarities)
        threshold = max(floor, mean + config.librarian_semantic_margin)
        return [s for s in similarities if s.similarity >= threshold]

    ranked = sorted(similarities, key=lambda s: -s.similarity)
    top = ranked[0]
    if top.similarity < floor:
        return []
    if len(ranked) == 1:
        return [top]
    gap = top.similarity - ranked[1].similarity
    if gap >= config.librarian_semantic_min_top_gap:
        return [top]
    return []


async def semantic_fallback(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None,
    config: PrimoConfig,
    *,
    limit: int = 2,
    embedder: Embedder | None = None,
    timeout: float | None = None,
) -> SemanticFallbackResult:
    """Rank configured librarians by semantic similarity to the query.

    Returns no matches when disabled or when no profile clears the acceptance
    rule. When embedding fails the error is logged to stderr (safe under the
    stdio MCP transport) and returned in ``error`` so callers can distinguish
    "the fallback broke" from "the fallback found nothing".

    ``timeout`` overrides ``config.embedding_timeout`` for latency-sensitive
    callers such as the inline primo_search path.
    """
    if not config.librarian_semantic_fallback:
        return SemanticFallbackResult([])

    # Gate before embedding: one-word and filler-only queries are where
    # cosine over bag-of-terms profile documents is least reliable, and
    # skipping here avoids the embedding call (and its cost) entirely.
    min_tokens = config.librarian_semantic_min_query_tokens
    if min_tokens > 1 and _content_token_count(query) < min_tokens:
        return SemanticFallbackResult(
            [],
            skipped=(
                "the query has too few topical words for reliable "
                f"semantic matching (needs at least {min_tokens})"
            ),
        )

    try:
        similarities = await score_profiles(
            directory,
            _query_text(query, records),
            config,
            embedder=embedder,
            timeout=timeout,
        )
    except Exception as e:
        logger.warning(
            "Semantic librarian fallback failed (%s): %s", type(e).__name__, e
        )
        return SemanticFallbackResult([], error=type(e).__name__)

    # Curator deny-lists apply here too, or the semantic path would
    # resurrect a librarian the keyword path deliberately suppressed. Applied
    # after acceptance so excluded profiles still contribute to the mean the
    # margin rule calibrates against.
    scored = [
        s for s in _accepted(similarities, config)
        if not is_excluded(s.librarian, query)
    ]
    scored.sort(key=lambda item: (-item.similarity, item.librarian.name.casefold()))
    capped_limit = min(max(1, limit), _MAX_RECOMMENDATIONS)
    return SemanticFallbackResult(
        [
            LibrarianMatch(
                librarian=librarian,
                score=round(similarity, 4),
                matched_terms=[],
                evidence_fields=["semantic"],
            )
            for similarity, librarian in scored[:capped_limit]
        ]
    )
