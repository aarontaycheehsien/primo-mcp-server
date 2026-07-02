"""Semantic (embedding) fallback for librarian recommendations.

This layer is consulted *only* when the deterministic keyword matcher in
``librarians.recommend_librarians`` returns no match. It ranks the configured
librarian profiles by cosine similarity between a Gemini embedding of the
query and cached embeddings of each profile.

Design guarantees:
- Fails closed: any error (missing key, network failure, malformed response)
  returns an empty match list, so behaviour degrades exactly to the keyword
  path's no-match outcome -- the tool never errors because of this layer.
- Only configured profiles are ever ranked or returned, so the
  anti-hallucination guardrail is preserved.
- Profile embeddings are cached to a sidecar file keyed by a content hash and
  the model id, so the (paid/slow) document embeddings are computed once and
  re-used until a profile or the model changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Awaitable, Callable, Sequence

import httpx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarians import (
    _MAX_RECOMMENDATIONS,
    LibrarianDirectory,
    LibrarianMatch,
    LibrarianProfile,
)
from primo_mcp_server.models import PrimoRecord

# (texts, task_type) -> one embedding vector per input text.
Embedder = Callable[[Sequence[str], str], Awaitable[list[list[float]]]]

_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"
_MAX_QUERY_CHARS = 2000


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


def _hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


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
    model: str,
    vectors: dict[str, list[float]],
    hashes: dict[str, str],
) -> None:
    if path is None:
        return
    data = {
        "model": model,
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
    texts: Sequence[str], task_type: str, *, config: PrimoConfig
) -> list[list[float]]:
    """Embed ``texts`` via the Gemini ``embedContent`` endpoint.

    The Gemini embedding models expose a single-content ``embedContent`` method
    (the only synchronous one), so each text is embedded with its own request;
    the handful of requests per call are issued concurrently.
    """
    if not config.embedding_api_key:
        raise RuntimeError("embedding_api_key is not configured")
    base = config.embedding_api_url.rstrip("/")
    model_path = f"models/{config.embedding_model}"
    url = f"{base}/{model_path}:embedContent"

    async with httpx.AsyncClient(timeout=config.embedding_timeout) as client:

        async def embed_one(text: str) -> list[float]:
            response = await client.post(
                url,
                params={"key": config.embedding_api_key},
                json={
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                },
            )
            response.raise_for_status()
            return response.json()["embedding"]["values"]

        return list(await asyncio.gather(*(embed_one(text) for text in texts)))


async def _load_or_build_profile_vectors(
    directory: LibrarianDirectory,
    config: PrimoConfig,
    embed: Embedder,
) -> dict[str, list[float]]:
    """Return one embedding per librarian, re-using a sidecar cache."""
    path = _cache_path(config)
    cache = _read_cache(path)
    entries = cache.get("entries", {}) if cache.get("model") == config.embedding_model else {}

    vectors: dict[str, list[float]] = {}
    hashes: dict[str, str] = {}
    stale: list[tuple[str, str]] = []
    for librarian in directory.librarians:
        text = _profile_text(librarian)
        if not text:
            continue
        digest = _hash(text, config.embedding_model)
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
        _write_cache(path, config.embedding_model, vectors, hashes)
    return vectors


async def semantic_fallback(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None,
    config: PrimoConfig,
    *,
    limit: int = 2,
    embedder: Embedder | None = None,
) -> list[LibrarianMatch]:
    """Rank configured librarians by semantic similarity to the query.

    Returns an empty list when disabled, when no profile clears the similarity
    floor, or when anything goes wrong with embedding.
    """
    if not config.librarian_semantic_fallback:
        return []

    embed = embedder or (
        lambda texts, task_type: _gemini_embed(texts, task_type, config=config)
    )

    try:
        profile_vectors = await _load_or_build_profile_vectors(directory, config, embed)
        if not profile_vectors:
            return []
        query_vector = (await embed([_query_text(query, records)], _TASK_QUERY))[0]
    except Exception:
        return []

    scored: list[tuple[float, LibrarianProfile]] = []
    for librarian in directory.librarians:
        vector = profile_vectors.get(librarian.id)
        if not vector:
            continue
        similarity = _cosine(query_vector, vector)
        if similarity >= config.librarian_semantic_min_similarity:
            scored.append((similarity, librarian))

    scored.sort(key=lambda item: (-item[0], item[1].name.casefold()))
    capped_limit = min(max(1, limit), _MAX_RECOMMENDATIONS)
    return [
        LibrarianMatch(
            librarian=librarian,
            score=round(similarity, 4),
            matched_terms=[],
            evidence_fields=["semantic"],
        )
        for similarity, librarian in scored[:capped_limit]
    ]
