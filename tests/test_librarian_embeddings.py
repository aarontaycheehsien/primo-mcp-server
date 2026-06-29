"""Tests for the semantic (embedding) librarian fallback."""

from __future__ import annotations

import json

import httpx
import respx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_embeddings import _gemini_embed, semantic_fallback
from primo_mcp_server.librarians import LibrarianDirectory

# A tiny deterministic "embedding" space: one dimension per topic. A text's
# vector marks which topics it mentions, so cosine similarity recovers topical
# overlap without any network call.
_TOPICS = ["preservation", "law", "accounting", "bibliometric"]


class _FakeEmbedder:
    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, texts, task_type):
        self.calls.append((len(texts), task_type))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [1.0 if topic in lowered else 0.0 for topic in _TOPICS]
            # An orthogonal "no topic" dimension so unrelated text stays
            # dissimilar to every profile (cosine 0) rather than correlating.
            vector.append(0.0 if any(vector) else 1.0)
            vectors.append(vector)
        return vectors


def _directory() -> LibrarianDirectory:
    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "preservation",
                    "name": "Preservation Librarian",
                    "subjects": ["preservation"],
                },
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law"],
                },
                {
                    "id": "metrics",
                    "name": "Metrics Librarian",
                    "subjects": ["bibliometric"],
                },
            ]
        }
    )


def _config(tmp_path, **overrides) -> PrimoConfig:
    values = {
        "librarian_semantic_fallback": True,
        "embedding_api_key": "test-key",
        "embedding_cache_file": str(tmp_path / "embeddings.json"),
        "librarian_semantic_min_similarity": 0.5,
    }
    values.update(overrides)
    return PrimoConfig(**values)


async def test_semantic_fallback_ranks_by_similarity(tmp_path):
    matches = await semantic_fallback(
        _directory(),
        "long-term preservation of born-digital archives",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert [m.librarian.id for m in matches] == ["preservation"]
    assert matches[0].evidence_fields == ["semantic"]


async def test_semantic_fallback_disabled_returns_empty(tmp_path):
    matches = await semantic_fallback(
        _directory(),
        "preservation",
        [],
        _config(tmp_path, librarian_semantic_fallback=False),
        embedder=_FakeEmbedder(),
    )

    assert matches == []


async def test_semantic_fallback_below_threshold_returns_empty(tmp_path):
    # No topic overlap -> similarity below the floor -> honest no-match.
    matches = await semantic_fallback(
        _directory(),
        "tropical marine biology",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert matches == []


async def test_semantic_fallback_degrades_on_embedder_error(tmp_path):
    async def boom(texts, task_type):
        raise RuntimeError("embedding service down")

    matches = await semantic_fallback(
        _directory(),
        "preservation",
        [],
        _config(tmp_path),
        embedder=boom,
    )

    assert matches == []


async def test_profile_embeddings_are_cached_and_reused(tmp_path):
    config = _config(tmp_path)
    directory = _directory()

    first = _FakeEmbedder()
    await semantic_fallback(directory, "preservation", [], config, embedder=first)
    # First run embeds all profiles (one document batch) plus the query.
    assert any(task == "RETRIEVAL_DOCUMENT" for _, task in first.calls)

    second = _FakeEmbedder()
    await semantic_fallback(directory, "law", [], config, embedder=second)
    # Second run reuses the cache: only the query is embedded, no documents.
    assert all(task == "RETRIEVAL_QUERY" for _, task in second.calls)


@respx.mock
async def test_gemini_embed_uses_embed_content_endpoint():
    config = PrimoConfig(
        embedding_api_key="test-key",
        embedding_model="gemini-embedding-001",
    )
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:embedContent"
    ).mock(return_value=httpx.Response(200, json={"embedding": {"values": [0.1, 0.2, 0.3]}}))

    vectors = await _gemini_embed(["hello"], "RETRIEVAL_QUERY", config=config)

    assert vectors == [[0.1, 0.2, 0.3]]
    request = route.calls.last.request
    assert request.url.params["key"] == "test-key"
    body = json.loads(request.content)
    assert body["model"] == "models/gemini-embedding-001"
    assert body["taskType"] == "RETRIEVAL_QUERY"
    assert body["content"]["parts"][0]["text"] == "hello"


async def test_semantic_fallback_respects_limit(tmp_path):
    matches = await semantic_fallback(
        _directory(),
        "law and bibliometric preservation",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.1),
        limit=2,
        embedder=_FakeEmbedder(),
    )

    assert len(matches) == 2
