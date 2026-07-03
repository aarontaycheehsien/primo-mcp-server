"""Tests for the semantic (embedding) librarian fallback."""

from __future__ import annotations

import json

import httpx
import respx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_embeddings import (
    _gemini_embed,
    _retry_delay_seconds,
    semantic_fallback,
)
from primo_mcp_server.librarians import LibrarianDirectory
from primo_mcp_server.models import PrimoRecord


def _http_error(
    status: int = 429,
    headers: dict | None = None,
    json_body: dict | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/embed")
    if json_body is not None:
        response = httpx.Response(
            status, headers=headers, json=json_body, request=request
        )
    else:
        response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


def _record_sleeps(monkeypatch) -> list[float]:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "primo_mcp_server.librarian_embeddings._sleep", fake_sleep
    )
    return sleeps

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
    matches, error, skipped = await semantic_fallback(
        _directory(),
        "long-term preservation of born-digital archives",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert error is None
    assert skipped is None
    assert [m.librarian.id for m in matches] == ["preservation"]
    assert matches[0].evidence_fields == ["semantic"]


async def test_semantic_fallback_disabled_returns_empty(tmp_path):
    matches, error, skipped = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        _config(tmp_path, librarian_semantic_fallback=False),
        embedder=_FakeEmbedder(),
    )

    assert matches == []
    assert error is None
    assert skipped is None


async def test_semantic_fallback_below_threshold_returns_empty(tmp_path):
    # No topic overlap -> similarity below the floor -> honest no-match.
    matches, error, _ = await semantic_fallback(
        _directory(),
        "tropical marine biology",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert matches == []
    assert error is None


async def test_semantic_fallback_ignores_record_context(tmp_path):
    matches, _, _ = await semantic_fallback(
        _directory(),
        "tropical marine biology",
        [PrimoRecord(title="Digital preservation", subjects=["preservation"])],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert matches == []


async def test_semantic_fallback_degrades_on_embedder_error(tmp_path, caplog):
    async def boom(texts, task_type):
        raise RuntimeError("embedding service down")

    matches, error, _ = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        _config(tmp_path),
        embedder=boom,
    )

    # Fails closed, but the error is distinguishable from a genuine no-match
    # and is logged (to stderr under stdio transport) rather than swallowed.
    assert matches == []
    assert error == "RuntimeError"
    assert "embedding service down" in caplog.text


async def test_short_query_is_gated_before_embedding(tmp_path):
    embedder = _FakeEmbedder()

    matches, error, skipped = await semantic_fallback(
        _directory(),
        "preservation",
        [],
        _config(tmp_path),
        embedder=embedder,
    )

    # One topical word is below the default gate of two; the fallback is
    # skipped with a reason and no embedding request is ever made.
    assert matches == []
    assert error is None
    assert skipped is not None
    assert "too few topical words" in skipped
    assert embedder.calls == []


async def test_filler_only_query_is_gated(tmp_path):
    embedder = _FakeEmbedder()

    matches, _, skipped = await semantic_fallback(
        _directory(),
        "can you help me find some research support please",
        [],
        _config(tmp_path),
        embedder=embedder,
    )

    # Stopwords and filler words contribute no topical tokens.
    assert matches == []
    assert skipped is not None
    assert embedder.calls == []


async def test_query_gate_can_be_disabled(tmp_path):
    # The same one-word query that the default gate skips matches when the
    # gate is configured off.
    matches, error, skipped = await semantic_fallback(
        _directory(),
        "preservation",
        [],
        _config(tmp_path, librarian_semantic_min_query_tokens=1),
        embedder=_FakeEmbedder(),
    )

    assert error is None
    assert skipped is None
    assert [m.librarian.id for m in matches] == ["preservation"]


async def test_margin_rule_rejects_uniform_similarities(tmp_path):
    # Four profiles that all match the query equally well: nothing stands out
    # above the mean, so the self-calibrating margin refuses to pick one even
    # though every similarity clears the absolute floor.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": f"law{i}", "name": f"Law Librarian {i}", "subjects": ["law"]}
                for i in range(4)
            ]
        }
    )

    matches, error, _ = await semantic_fallback(
        directory,
        "law and legislation",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.1),
        embedder=_FakeEmbedder(),
    )

    assert matches == []
    assert error is None


async def test_margin_rule_accepts_profile_that_stands_out(tmp_path):
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "law", "name": "Law Librarian", "subjects": ["law"]},
                {"id": "a", "name": "A", "subjects": ["accounting"]},
                {"id": "b", "name": "B", "subjects": ["bibliometric"]},
                {"id": "c", "name": "C", "subjects": ["preservation"]},
            ]
        }
    )

    matches, _, _ = await semantic_fallback(
        directory,
        "law and legislation",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.1),
        embedder=_FakeEmbedder(),
    )

    assert [m.librarian.id for m in matches] == ["law"]


async def test_tiny_directory_requires_top_gap(tmp_path):
    # Below the margin's profile minimum the mean is noise, so uniform
    # similarity in a tiny directory means the space cannot separate the
    # profiles: nothing is returned even above the floor.
    matches, error, _ = await semantic_fallback(
        _directory(),
        "law and bibliometric preservation",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.1),
        limit=2,
        embedder=_FakeEmbedder(),
    )

    assert matches == []
    assert error is None


async def test_tiny_directory_accepts_clear_top_profile(tmp_path):
    # A profile that clearly leads the runner-up is accepted -- and only
    # top-1, since the gap is the evidence that justified accepting it.
    matches, _, _ = await semantic_fallback(
        _directory(),
        "preservation of digital archives",
        [],
        _config(tmp_path),
        limit=2,
        embedder=_FakeEmbedder(),
    )

    assert [m.librarian.id for m in matches] == ["preservation"]


async def test_single_profile_directory_uses_floor_only(tmp_path):
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "law", "name": "Law Librarian", "subjects": ["law"]}
            ]
        }
    )

    matches, _, _ = await semantic_fallback(
        directory,
        "law and legislation",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert [m.librarian.id for m in matches] == ["law"]


async def test_padded_profile_matches_by_its_best_term(tmp_path):
    # A profile listing many unrelated topics must still match sharply on
    # the one term that fits the query. Under the old single-document
    # embedding, the padding diluted the cosine (0.5 here) below the floor;
    # per-term scoring takes the max over terms instead.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "padded",
                    "name": "Padded Librarian",
                    "subjects": ["law", "accounting", "bibliometric", "preservation"],
                },
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law"],
                },
            ]
        }
    )

    matches, error, _ = await semantic_fallback(
        directory,
        "preservation of digital archives",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.6),
        embedder=_FakeEmbedder(),
    )

    assert error is None
    assert [m.librarian.id for m in matches] == ["padded"]


async def test_shared_terms_are_embedded_once(tmp_path):
    # Two profiles listing the same term produce one document embedding, not
    # two: the cache is keyed by term content, not by profile.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "law1", "name": "Law Librarian One", "subjects": ["law"]},
                {"id": "law2", "name": "Law Librarian Two", "subjects": ["law"]},
            ]
        }
    )
    embedder = _FakeEmbedder()

    await semantic_fallback(
        directory,
        "law and legislation",
        [],
        _config(tmp_path),
        embedder=embedder,
    )

    document_calls = [
        count for count, task in embedder.calls if task == "RETRIEVAL_DOCUMENT"
    ]
    assert document_calls == [1]


async def test_old_cache_format_is_discarded_and_rebuilt(tmp_path):
    # A version-1 sidecar (one vector per profile, keyed by librarian id) is
    # from before per-term scoring; it must be ignored, not misread.
    config = _config(tmp_path)
    (tmp_path / "embeddings.json").write_text(
        json.dumps(
            {
                "model": "gemini-embedding-001",
                "entries": {
                    "law": {"hash": "stale", "vector": [0.0, 1.0, 0.0, 0.0, 0.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    embedder = _FakeEmbedder()

    matches, error, _ = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        config,
        embedder=embedder,
    )

    assert error is None
    assert any(task == "RETRIEVAL_DOCUMENT" for _, task in embedder.calls)
    assert [m.librarian.id for m in matches] == ["preservation"]


def test_retry_delay_prefers_server_advice():
    # Retry-After header wins.
    header = _http_error(headers={"retry-after": "7"})
    assert _retry_delay_seconds(header, attempt=0, cap=65.0) == 7.0

    # Google's RetryInfo body detail is honoured when the header is absent.
    body = _http_error(
        json_body={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "37s",
                    }
                ]
            }
        }
    )
    assert _retry_delay_seconds(body, attempt=0, cap=65.0) == 37.0

    # No advice: exponential backoff, capped.
    silent = _http_error()
    assert _retry_delay_seconds(silent, attempt=0, cap=65.0) == 5.0
    assert _retry_delay_seconds(silent, attempt=4, cap=65.0) == 65.0

    # Advice above the cap is clamped.
    slow = _http_error(headers={"retry-after": "600"})
    assert _retry_delay_seconds(slow, attempt=0, cap=65.0) == 65.0


async def test_rate_limited_rebuild_waits_and_retries(tmp_path, monkeypatch):
    sleeps = _record_sleeps(monkeypatch)

    class _RateLimitedOnce(_FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.failures = 0

        async def __call__(self, texts, task_type):
            if task_type == "RETRIEVAL_DOCUMENT" and self.failures == 0:
                self.failures += 1
                raise _http_error(headers={"retry-after": "7"})
            return await super().__call__(texts, task_type)

    matches, error, _ = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        _config(tmp_path),
        embedder=_RateLimitedOnce(),
    )

    # The 429 was waited out for exactly the advised delay and the call
    # succeeded on the retry instead of failing closed.
    assert error is None
    assert [m.librarian.id for m in matches] == ["preservation"]
    assert sleeps == [7.0]


async def test_tight_timeout_opts_out_of_retry_waits(tmp_path, monkeypatch):
    sleeps = _record_sleeps(monkeypatch)

    async def always_rate_limited(texts, task_type):
        raise _http_error(headers={"retry-after": "60"})

    matches, error, _ = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        _config(tmp_path),
        embedder=always_rate_limited,
        timeout=2.5,  # the inline primo_search budget
    )

    # A latency-bounded caller fails closed immediately; sleeping 60s inside
    # an ordinary search would blow the very budget the timeout protects.
    assert matches == []
    assert error == "HTTPStatusError"
    assert sleeps == []


async def test_non_rate_limit_http_errors_are_not_retried(tmp_path, monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    calls: list[str] = []

    async def unauthorized(texts, task_type):
        calls.append(task_type)
        raise _http_error(status=401)

    matches, error, _ = await semantic_fallback(
        _directory(),
        "digital preservation of archives",
        [],
        _config(tmp_path),
        embedder=unauthorized,
    )

    # Retrying an invalid API key would never succeed; fail closed at once.
    assert matches == []
    assert error == "HTTPStatusError"
    assert sleeps == []
    assert len(calls) == 1


async def test_failed_rebuild_keeps_progress_and_resumes(tmp_path):
    # A directory with more terms than one batch: the first batch succeeds,
    # the second fails (free-tier rate limits are the common case). The
    # cache must keep the first batch so the retry embeds only the rest.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "big",
                    "name": "Big Librarian",
                    "subjects": [f"topic {i}" for i in range(120)],
                },
                {
                    "id": "preservation",
                    "name": "Preservation Librarian",
                    "subjects": ["preservation"],
                },
            ]
        }
    )
    config = _config(tmp_path)

    class _FailsAfterFirstBatch(_FakeEmbedder):
        async def __call__(self, texts, task_type):
            if task_type == "RETRIEVAL_DOCUMENT" and self.calls:
                self.calls.append((len(texts), task_type))
                raise RuntimeError("rate limited")
            return await super().__call__(texts, task_type)

    first = _FailsAfterFirstBatch()
    matches, error, _ = await semantic_fallback(
        directory,
        "digital preservation of archives",
        [],
        config,
        embedder=first,
    )
    # Fails closed for this call, but the first batch of 100 is persisted.
    assert error == "RuntimeError"
    assert matches == []

    second = _FakeEmbedder()
    matches, error, _ = await semantic_fallback(
        directory,
        "digital preservation of archives",
        [],
        config,
        embedder=second,
    )
    assert error is None
    assert [m.librarian.id for m in matches] == ["preservation"]
    document_batches = [
        count for count, task in second.calls if task == "RETRIEVAL_DOCUMENT"
    ]
    # 121 terms total, 100 already cached: the retry embeds only 21.
    assert document_batches == [21]


async def test_profile_embeddings_are_cached_and_reused(tmp_path):
    config = _config(tmp_path)
    directory = _directory()

    first = _FakeEmbedder()
    await semantic_fallback(
        directory, "digital preservation of archives", [], config, embedder=first
    )
    # First run embeds all profiles (one document batch) plus the query.
    assert any(task == "RETRIEVAL_DOCUMENT" for _, task in first.calls)

    second = _FakeEmbedder()
    await semantic_fallback(
        directory, "law and legislation", [], config, embedder=second
    )
    # Second run reuses the cache: only the query is embedded, no documents.
    assert second.calls
    assert all(task == "RETRIEVAL_QUERY" for _, task in second.calls)


async def test_changing_dimensions_invalidates_cache(tmp_path):
    directory = _directory()

    first = _FakeEmbedder()
    await semantic_fallback(
        directory,
        "digital preservation of archives",
        [],
        _config(tmp_path),
        embedder=first,
    )

    # Same directory, new output dimensionality: cached vectors are from a
    # different embedding space and must be rebuilt.
    second = _FakeEmbedder()
    await semantic_fallback(
        directory,
        "digital preservation of archives",
        [],
        _config(tmp_path, embedding_dimensions=5),
        embedder=second,
    )
    assert any(task == "RETRIEVAL_DOCUMENT" for _, task in second.calls)


@respx.mock
async def test_gemini_embed_uses_batch_endpoint_and_header_auth():
    config = PrimoConfig(
        embedding_api_key="test-key",
        embedding_model="gemini-embedding-001",
    )
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [0.1, 0.2, 0.3]},
                    {"values": [0.4, 0.5, 0.6]},
                ]
            },
        )
    )

    vectors = await _gemini_embed(["hello", "world"], "RETRIEVAL_QUERY", config=config)

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert route.call_count == 1  # one batch call, not one call per text
    request = route.calls.last.request
    # Key travels in a header, never in the URL where proxies would log it.
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key" not in request.url.params
    body = json.loads(request.content)
    assert len(body["requests"]) == 2
    for entry, text in zip(body["requests"], ["hello", "world"]):
        assert entry["model"] == "models/gemini-embedding-001"
        assert entry["taskType"] == "RETRIEVAL_QUERY"
        assert entry["content"]["parts"][0]["text"] == text
        assert "outputDimensionality" not in entry


@respx.mock
async def test_gemini_embed_passes_output_dimensionality():
    config = PrimoConfig(
        embedding_api_key="test-key",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
    )
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:batchEmbedContents"
    ).mock(
        return_value=httpx.Response(
            200, json={"embeddings": [{"values": [0.1, 0.2]}]}
        )
    )

    await _gemini_embed(["hello"], "RETRIEVAL_QUERY", config=config)

    body = json.loads(route.calls.last.request.content)
    assert body["requests"][0]["outputDimensionality"] == 768


@respx.mock
async def test_gemini_embed_chunks_large_batches():
    config = PrimoConfig(
        embedding_api_key="test-key",
        embedding_model="gemini-embedding-001",
    )

    def respond(request):
        count = len(json.loads(request.content)["requests"])
        return httpx.Response(
            200, json={"embeddings": [{"values": [0.1]}] * count}
        )

    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:batchEmbedContents"
    ).mock(side_effect=respond)

    vectors = await _gemini_embed(
        [f"text {i}" for i in range(150)], "RETRIEVAL_DOCUMENT", config=config
    )

    assert len(vectors) == 150
    assert route.call_count == 2  # 100 + 50


async def test_semantic_fallback_respects_limit(tmp_path):
    # Four profiles (margin regime) where three tie above mean + margin;
    # the limit caps how many are returned.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "law", "name": "Law Librarian", "subjects": ["law"]},
                {
                    "id": "metrics",
                    "name": "Metrics Librarian",
                    "subjects": ["bibliometric"],
                },
                {
                    "id": "preservation",
                    "name": "Preservation Librarian",
                    "subjects": ["preservation"],
                },
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "subjects": ["accounting"],
                },
            ]
        }
    )

    matches, _, _ = await semantic_fallback(
        directory,
        "law bibliometric preservation trends",
        [],
        _config(tmp_path, librarian_semantic_min_similarity=0.1),
        limit=2,
        embedder=_FakeEmbedder(),
    )

    assert len(matches) == 2


async def test_semantic_fallback_respects_curator_excludes(tmp_path):
    # A deny-listed librarian must not be resurrected by the semantic path.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law"],
                    "excludes": ["litigation"],
                },
                {
                    "id": "metrics",
                    "name": "Metrics Librarian",
                    "subjects": ["bibliometric"],
                },
            ]
        }
    )

    matches, error, _ = await semantic_fallback(
        directory,
        "law litigation strategy",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )

    assert error is None
    assert matches == []

    # The same directory still matches when no exclude term is present.
    matches, _, _ = await semantic_fallback(
        directory,
        "case law research",
        [],
        _config(tmp_path),
        embedder=_FakeEmbedder(),
    )
    assert [m.librarian.id for m in matches] == ["law"]
