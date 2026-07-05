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
- Each profile term is embedded as its own vector and a profile scores by
  its best term (max cosine). Averaging a large profile into one document
  vector dilutes every topic it lists -- a profile with 150 aliases would
  need the whole bag to resemble the query -- whereas the routing question
  is whether ANY configured topic matches.
- Term embeddings are cached to a sidecar file keyed by a content hash of
  each term and the model id (plus output dimensionality), so the
  (paid/slow) document embeddings are computed once and re-used until a
  term, the model, or the dimensionality changes. Terms shared by several
  profiles are embedded once.
"""

from __future__ import annotations

import asyncio
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
    _normalise_text,
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
    ``near_miss`` is the highest-similarity profile when the acceptance rule
    rejected everything -- evidence for the no_match output, never a match.
    """

    matches: list[LibrarianMatch]
    error: str | None = None
    skipped: str | None = None
    near_miss: LibrarianMatch | None = None


class ProfileSimilarity(NamedTuple):
    """One profile's cosine similarity to a query (for scoring and the CLI)."""

    similarity: float
    librarian: LibrarianProfile


def _profile_texts(librarian: LibrarianProfile) -> list[str]:
    """Topical text units embedded for a librarian, one vector each.

    Every configured term (and the notes prose, as one unit) becomes its own
    embedding; the profile later scores by its best term. Name and title are
    deliberately excluded -- they carry little topical signal and risk
    spurious matches (e.g. a query mentioning a person's name).

    Terms are de-duplicated by their normalised form, the same reduction the
    keyword matcher scores by: real profiles list case and plural variants
    of one concept ("Financial databases" / "financial database"), and
    embedding each variant buys near-identical vectors at real quota cost.
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
    seen: set[str] = set()
    texts: list[str] = []
    for part in parts:
        cleaned = part.strip() if part else ""
        if not cleaned:
            continue
        key = _normalise_text(cleaned) or cleaned.casefold()
        if key not in seen:
            seen.add(key)
            texts.append(cleaned)
    return texts


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


# Sidecar cache layout version. Version 2 keys entries by a content hash of
# each term text; version 1 (one document vector per profile, keyed by
# librarian id) is silently discarded and rebuilt on first use.
_CACHE_FORMAT = 2


def _read_cache(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("format") != _CACHE_FORMAT:
        return {}
    return data


def _write_cache(
    path: Path | None,
    model_key: str,
    vectors_by_hash: dict[str, list[float]],
) -> None:
    if path is None:
        return
    data = {
        "model": model_key,
        "format": _CACHE_FORMAT,
        "entries": vectors_by_hash,
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


# Indirection so tests can observe and skip real sleeps.
_sleep = asyncio.sleep


def _retry_delay_seconds(
    error: httpx.HTTPStatusError, attempt: int, cap: float
) -> float:
    """How long a 429 asks us to wait, capped; backoff when it does not say.

    Google's rate-limit responses carry the delay in the ``Retry-After``
    header and/or a ``google.rpc.RetryInfo`` detail in the JSON body (e.g.
    ``"retryDelay": "37s"``). Honouring the server's own number converges
    much faster than blind exponential backoff.
    """
    retry_after = error.response.headers.get("retry-after")
    if retry_after:
        try:
            return min(cap, max(1.0, float(retry_after)))
        except ValueError:
            pass
    try:
        details = error.response.json()["error"]["details"]
        for detail in details:
            if detail.get("@type", "").endswith("RetryInfo"):
                delay = str(detail.get("retryDelay", ""))
                if delay.endswith("s"):
                    return min(cap, max(1.0, float(delay[:-1])))
    except Exception:
        pass
    return min(cap, 5.0 * (2**attempt))


async def _embed_with_retry(
    embed: Embedder,
    texts: Sequence[str],
    task_type: str,
    config: PrimoConfig,
    *,
    retries: int,
) -> list[list[float]]:
    """Call ``embed``, waiting out HTTP 429 up to ``retries`` times.

    Only rate limiting is retried -- auth failures, malformed responses,
    and network errors still fail immediately (and closed, in the caller).
    """
    for attempt in range(max(0, retries) + 1):
        try:
            return await embed(texts, task_type)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429 or attempt >= retries:
                raise
            delay = _retry_delay_seconds(
                e, attempt, config.embedding_retry_max_delay
            )
            logger.warning(
                "Gemini embedding rate limited (429); waiting %.0fs before "
                "retry %d/%d",
                delay,
                attempt + 1,
                retries,
            )
            await _sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises


async def _load_or_build_profile_vectors(
    directory: LibrarianDirectory,
    config: PrimoConfig,
    embed: Embedder,
    *,
    retries: int = 0,
) -> dict[str, list[list[float]]]:
    """Return one embedding per profile term, re-using a sidecar cache.

    The cache is keyed by a content hash of each term text, so a term shared
    by several profiles is embedded and stored once, and editing one term on
    one profile re-embeds only that term.
    """
    path = _cache_path(config)
    cache = _read_cache(path)
    model_key = _model_key(config)
    entries = cache.get("entries", {}) if cache.get("model") == model_key else {}

    texts_by_profile = {
        librarian.id: _profile_texts(librarian)
        for librarian in directory.librarians
    }
    vectors_by_hash: dict[str, list[float]] = {}
    pending: dict[str, str] = {}
    for texts in texts_by_profile.values():
        for text in texts:
            digest = _hash(text, model_key)
            cached_vector = entries.get(digest)
            if cached_vector:
                vectors_by_hash[digest] = cached_vector
            else:
                pending.setdefault(digest, text)

    if pending:
        ordered = list(pending.items())
        # Embed chunk by chunk and persist the cache after every chunk, so a
        # mid-rebuild failure (free-tier rate limits are the common case for
        # a large directory) keeps the progress made and a retry resumes
        # from the remainder instead of restarting from zero. Rewriting only
        # the hashes in use also prunes vectors for removed terms.
        for start in range(0, len(ordered), _MAX_BATCH_SIZE):
            chunk = ordered[start : start + _MAX_BATCH_SIZE]
            new_vectors = await _embed_with_retry(
                embed,
                [text for _, text in chunk],
                _TASK_DOCUMENT,
                config,
                retries=retries,
            )
            for (digest, _), vector in zip(chunk, new_vectors):
                vectors_by_hash[digest] = vector
            _write_cache(path, model_key, vectors_by_hash)

    return {
        lib_id: [
            vectors_by_hash[digest]
            for text in texts
            if (digest := _hash(text, model_key)) in vectors_by_hash
        ]
        for lib_id, texts in texts_by_profile.items()
    }


async def score_profiles(
    directory: LibrarianDirectory,
    query: str,
    config: PrimoConfig,
    *,
    embedder: Embedder | None = None,
    timeout: float | None = None,
    retries: int | None = None,
) -> list[ProfileSimilarity]:
    """Cosine similarity of every embeddable profile to ``query``, unsorted.

    A profile's similarity is the maximum over its per-term vectors: the
    routing question is whether any configured topic matches the query, so a
    sharp hit on one term must not be averaged away by the profile's other
    topics. A stray term causing a false positive is a curation problem --
    the profile lint tool flags candidates and ``excludes`` patches them.

    ``retries`` bounds how many times an HTTP 429 is waited out (None means
    ``config.embedding_retry_attempts``); pass 0 on latency-bounded paths.

    Raises on embedding failure; ``semantic_fallback`` wraps this with the
    fail-closed handling, while the calibration CLI lets errors propagate.
    """
    if retries is None:
        retries = config.embedding_retry_attempts
    embed = embedder or (
        lambda texts, task_type: _gemini_embed(
            texts, task_type, config=config, timeout=timeout
        )
    )
    profile_vectors = await _load_or_build_profile_vectors(
        directory, config, embed, retries=retries
    )
    if not any(profile_vectors.values()):
        return []
    query_vector = (
        await _embed_with_retry(
            embed, [_query_text(query, None)], _TASK_QUERY, config, retries=retries
        )
    )[0]
    return [
        ProfileSimilarity(
            max(_cosine(query_vector, vector) for vector in vectors),
            librarian,
        )
        for librarian in directory.librarians
        if (vectors := profile_vectors.get(librarian.id))
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
    callers such as the inline primo_search path; a caller that sets it also
    opts out of 429 retry waits, since sleeping would blow the same budget
    the tight timeout protects.
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
            retries=0 if timeout is not None else None,
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

    # When the acceptance rule rejected everything, keep the closest profile
    # (with its cosine) so a no_match outcome can show WHY nothing was good
    # enough instead of discarding the evidence. Curator exclusions apply
    # here too; a near-miss must never resurrect a suppressed profile.
    near_miss: LibrarianMatch | None = None
    if not scored:
        rejected = [
            s for s in similarities if not is_excluded(s.librarian, query)
        ]
        if rejected:
            top = max(rejected, key=lambda s: s.similarity)
            near_miss = LibrarianMatch(
                librarian=top.librarian,
                score=round(top.similarity, 4),
                matched_terms=[],
                evidence_fields=["semantic"],
            )

    return SemanticFallbackResult(
        [
            LibrarianMatch(
                librarian=librarian,
                score=round(similarity, 4),
                matched_terms=[],
                evidence_fields=["semantic"],
            )
            for similarity, librarian in scored[:capped_limit]
        ],
        near_miss=near_miss,
    )
