"""Tests for the deterministic RAG citation guardrail."""

import re
from types import SimpleNamespace

import pytest

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.models import PrimoRecord, SearchResponse
from primo_mcp_server.rag_guard import (
    RagSessionStore,
    build_references,
    extract_citation_ids,
    find_prose_citations,
    validate_ids,
)
from primo_mcp_server.server import (
    RAG_SESSIONS,
    primo_rag_retrieve,
    primo_rag_validate,
)


def _record(i: int, **overrides) -> PrimoRecord:
    values = {
        "record_id": f"cdi_test_{i}",
        "title": f"Test Article {i}",
        "resource_type": "article",
        "creators": [f"Author{i}, Alice"],
        "creation_date": "2023",
        "journal_title": "Journal of Testing",
        "description": f"Abstract for article {i} about retrieval.",
        "doi": f"10.1234/test.{i}",
    }
    values.update(overrides)
    return PrimoRecord(**values)


# ---------------------------------------------------------------------------
# extract_citation_ids
# ---------------------------------------------------------------------------

def test_extract_single_tags_first_seen_order():
    assert extract_citation_ids("Claim [2]. Another [1]. Again [2].") == [2, 1]


def test_extract_grouped():
    assert extract_citation_ids("Claim [1, 3; 2].") == [1, 3, 2]


def test_extract_ignores_prose_and_plain_text():
    assert extract_citation_ids("Smith et al. (2020) found 1 thing.") == []


def test_extract_empty_draft():
    assert extract_citation_ids("") == []


# ---------------------------------------------------------------------------
# validate_ids
# ---------------------------------------------------------------------------

def test_validate_splits_valid_and_invalid():
    valid, invalid = validate_ids([1, 999, 2, 0], record_count=5)
    assert valid == [1, 2]
    assert invalid == [999, 0]


def test_validate_all_valid():
    valid, invalid = validate_ids([1, 5], record_count=5)
    assert valid == [1, 5]
    assert invalid == []


# ---------------------------------------------------------------------------
# build_references
# ---------------------------------------------------------------------------

def test_build_references_sorted_dedup_and_from_metadata():
    records = [_record(1), _record(2), _record(3)]
    refs = build_references([3, 1, 3], records)
    lines = refs.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[1] ")
    assert lines[1].startswith("[3] ")
    assert "Test Article 1" in lines[0]
    assert "cdi_test_3" in lines[1]
    assert "10.1234/test.1" in lines[0]


# ---------------------------------------------------------------------------
# prose citation heuristic
# ---------------------------------------------------------------------------

def test_find_prose_citations_flags_author_year():
    draft = "RAG helps (Lewis et al., 2020). Ji et al. (2023) agree [R1]."
    hits = find_prose_citations(draft)
    assert any("Lewis" in h for h in hits)
    assert any("Ji" in h for h in hits)


def test_find_prose_citations_clean_draft():
    assert find_prose_citations("RAG helps [R1] and grounds answers [R2].") == []


# ---------------------------------------------------------------------------
# session store
# ---------------------------------------------------------------------------

def test_store_create_get_and_eviction():
    store = RagSessionStore(max_sessions=2)
    s1 = store.create("q1", [_record(1)])
    s2 = store.create("q2", [_record(1)])
    s3 = store.create("q3", [_record(1)])
    assert store.get(s1.session_id) is None  # evicted FIFO
    assert store.get(s2.session_id) is s2
    assert store.get(s3.session_id) is s3
    assert store.get(" " + s3.session_id + " ") is s3  # tolerant of whitespace


# ---------------------------------------------------------------------------
# server tools end to end (fake client)
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, records: list[PrimoRecord]):
        self.records = records
        self.search_calls: list[dict] = []

    async def search(self, **kwargs) -> SearchResponse:
        self.search_calls.append(kwargs)
        return SearchResponse.model_validate(
            {"info": {"total": len(self.records)}, "records": self.records}
        )


def _fake_context(client: _FakeClient) -> SimpleNamespace:
    lifespan_context = {
        "client": client,
        "config": PrimoConfig(
            base_url="https://example.test/primaws/rest/pub", _env_file=None
        ),
    }
    return SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=lifespan_context)
    )


def _session_id(retrieve_output: str) -> str:
    m = re.search(r"RAG session created: (rag-[0-9a-f]+)", retrieve_output)
    assert m, retrieve_output
    return m.group(1)


@pytest.fixture(autouse=True)
def _clean_sessions():
    RAG_SESSIONS.clear()
    yield
    RAG_SESSIONS.clear()


async def test_retrieve_pins_top_five_and_gives_rules():
    ctx = _fake_context(_FakeClient([_record(i) for i in range(1, 6)]))
    out = await primo_rag_retrieve(ctx, query="retrieval augmented generation")
    assert "Retrieved 5 source records" in out
    assert "[1] Test Article 1" in out
    assert "[5] Test Article 5" in out
    assert "DRAFTING RULES" in out
    assert "primo_rag_validate" in out


async def test_retrieve_zero_results_creates_no_session():
    ctx = _fake_context(_FakeClient([]))
    out = await primo_rag_retrieve(ctx, query="zxqv nonsense")
    assert "No RAG session was created" in out


async def test_validate_rejects_hallucinated_id_then_accepts_fixed_draft():
    ctx = _fake_context(_FakeClient([_record(i) for i in range(1, 6)]))
    out = await primo_rag_retrieve(ctx, query="rag hallucination")
    sid = _session_id(out)

    bad = "RAG grounds answers [1]. One study found zero hallucination [999]."
    failed = await primo_rag_validate(ctx, session_id=sid, draft_answer=bad)
    assert "VALIDATION FAILED" in failed
    assert "[999]" in failed
    assert "Regenerate" in failed

    good = "RAG grounds answers [1]. LLMs hallucinate citations [2, 3]."
    passed = await primo_rag_validate(ctx, session_id=sid, draft_answer=good)
    assert "VALIDATION PASSED" in passed
    assert "References" in passed
    assert "[1] " in passed and "[2] " in passed and "[3] " in passed
    assert "Test Article 1" in passed
    assert "Retrieved but uncited: [4], [5]." in passed
    assert "does NOT verify" in passed.replace("\n", " ")


async def test_validate_requires_structured_citations():
    ctx = _fake_context(_FakeClient([_record(1)]))
    out = await primo_rag_retrieve(ctx, query="anything", limit=1)
    sid = _session_id(out)
    failed = await primo_rag_validate(
        ctx, session_id=sid, draft_answer="An answer with no tags at all."
    )
    assert "VALIDATION FAILED" in failed
    assert "no [n] citation tags" in failed


async def test_validate_unknown_session():
    ctx = _fake_context(_FakeClient([_record(1)]))
    out = await primo_rag_validate(
        ctx, session_id="rag-deadbeef", draft_answer="Hi [1]."
    )
    assert "Unknown RAG session" in out
