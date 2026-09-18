"""Tests for the offline recommendation evaluation harness."""

from __future__ import annotations

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.evaluate_recommendations import (
    EvalSet,
    _load_eval_set,
    _unknown_expect_ids,
    evaluate,
)
from primo_mcp_server.librarians import LibrarianDirectory
from primo_mcp_server.recommendation import recommend_with_fallback


def _directory() -> LibrarianDirectory:
    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "subjects": ["accounting", "audit fees"],
                    "best_for": ["accounting datasets"],
                },
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law"],
                    "aliases": ["legal research"],
                },
            ]
        }
    )


def _config(**overrides) -> PrimoConfig:
    return PrimoConfig(_env_file=None, **overrides)


def _eval_set(cases: list[dict]) -> EvalSet:
    return EvalSet.model_validate({"cases": cases})


async def test_evaluate_scores_top1_and_misses():
    report = await evaluate(
        _eval_set(
            [
                {"query": "accounting datasets for audit fees", "expect": ["accounting"]},
                {"query": "legal research on case law", "expect": ["accounting"]},
            ]
        ),
        _directory(),
        _config(),
    )

    assert [r.passed for r in report.results] == [True, False]
    assert report.results[0].path == "keyword"
    assert report.results[1].got_ids == ["law"]
    assert report.pass_rate == 0.5


async def test_evaluate_accepts_any_expected_id():
    report = await evaluate(
        _eval_set(
            [{"query": "legal research on case law", "expect": ["accounting", "law"]}]
        ),
        _directory(),
        _config(),
    )

    assert report.results[0].passed is True


async def test_evaluate_no_match_cases_measure_false_positives():
    report = await evaluate(
        _eval_set(
            [
                {"query": "tropical marine biology fieldwork", "expect": []},
                {"query": "legal research on case law", "expect": []},
            ]
        ),
        _directory(),
        _config(),
    )

    # A correct rejection passes; a recommendation on an off-topic query is
    # a false positive and fails the case.
    assert report.results[0].passed is True
    assert report.results[1].passed is False
    assert report.results[1].got_ids == ["law"]


async def test_evaluate_identifier_queries_skip_the_pipeline():
    report = await evaluate(
        _eval_set(
            [
                {"query": "10.1145/1571941.1572114", "expect": []},
                {"query": "10.1145/1571941.1572114", "expect": ["law"]},
            ]
        ),
        _directory(),
        _config(),
    )

    assert report.results[0].passed is True
    assert report.results[0].path == "identifier-skip"
    # Expecting a match on an identifier query can never pass; surfacing the
    # failure tells the curator the label is wrong.
    assert report.results[1].passed is False


async def test_evaluate_uses_case_records_as_evidence():
    # "governance" alone matches nothing, but corroborating record metadata
    # scores the accounting profile above the threshold -- the records field
    # keeps such cases deterministic without a live Primo search.
    case = {
        "query": "governance disclosures",
        "expect": ["accounting"],
        "records": [
            {
                "title": "Audit fees and accounting quality",
                "subjects": ["accounting", "audit fees"],
                "keywords": ["accounting"],
            },
            {
                "title": "Accounting and audit fees handbook",
                "subjects": ["audit fees", "accounting"],
            },
        ],
    }

    bare = await evaluate(
        _eval_set([{**case, "records": []}]), _directory(), _config()
    )
    with_records = await evaluate(_eval_set([case]), _directory(), _config())

    assert bare.results[0].passed is False
    assert with_records.results[0].passed is True


def test_unknown_expect_ids_are_reported():
    eval_set = _eval_set(
        [
            {"query": "a", "expect": ["law"]},
            {"query": "b", "expect": ["lwa", "accounting"]},
        ]
    )

    assert _unknown_expect_ids(eval_set, _directory()) == ["lwa"]


def test_load_eval_set_rejects_bad_input(tmp_path):
    missing, error = _load_eval_set(str(tmp_path / "missing.json"))
    assert missing is None and error is not None and "Cannot read" in error

    empty = tmp_path / "empty.json"
    empty.write_text('{"cases": []}', encoding="utf-8")
    loaded, error = _load_eval_set(str(empty))
    assert loaded is None and error is not None and "no cases" in error

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"cases": [{"expect": []}]}', encoding="utf-8")
    loaded, error = _load_eval_set(str(invalid))
    assert loaded is None and error is not None and "Invalid eval case" in error


async def test_recommend_with_fallback_populates_near_misses_only_on_no_match():
    from primo_mcp_server.config import PrimoConfig
    from primo_mcp_server.librarians import LibrarianDirectory
    from primo_mcp_server.recommendation import recommend_with_fallback

    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "subjects": ["accounting", "audit fees"],
                    "best_for": ["accounting datasets"],
                }
            ]
        }
    )

    # An unreachable threshold forces no_match; the scored candidate must
    # survive as a near-miss with its evidence intact.
    config = PrimoConfig(
        librarian_min_score=10_000.0,
        librarian_semantic_fallback=False,
        _env_file=None,
    )
    outcome = await recommend_with_fallback(directory, "audit fees dataset", [], config)
    assert outcome.matches == []
    assert outcome.near_misses
    assert outcome.near_misses[0].librarian.id == "accounting"
    assert outcome.near_misses[0].matched_terms

    # With the normal threshold the same query matches, and near-misses
    # stay empty so they can never shadow a validated recommendation.
    config = PrimoConfig(
        librarian_min_score=5.0,
        librarian_semantic_fallback=False,
        _env_file=None,
    )
    outcome = await recommend_with_fallback(directory, "audit fees dataset", [], config)
    assert outcome.matches
    assert outcome.near_misses == ()


async def test_semantic_near_miss_reaches_outcome(monkeypatch):
    from primo_mcp_server.config import PrimoConfig
    from primo_mcp_server.librarian_embeddings import SemanticFallbackResult
    from primo_mcp_server.librarians import LibrarianDirectory, LibrarianMatch
    from primo_mcp_server.recommendation import recommend_with_fallback

    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "gis", "name": "GIS Librarian", "subjects": ["geospatial analysis"]}
            ]
        }
    )
    near = LibrarianMatch(
        librarian=directory.librarians[0],
        score=0.44,
        evidence_fields=["semantic"],
    )

    async def fake(directory, query, records, config, *, limit=2, timeout=None, **kwargs):
        return SemanticFallbackResult([], near_miss=near)

    monkeypatch.setattr(
        "primo_mcp_server.recommendation.semantic_fallback", fake
    )
    config = PrimoConfig(
        librarian_semantic_fallback=True,
        embedding_api_key="k",
        _env_file=None,
    )
    outcome = await recommend_with_fallback(
        directory, "mapping deprivation across neighbourhoods", [], config
    )

    assert outcome.matches == []
    assert outcome.near_misses == (near,)


# ---------------------------------------------------------------------------
# Tier 3: the LLM reasoning fallback's place in the pipeline.
# ---------------------------------------------------------------------------


def _llm_directory():
    from primo_mcp_server.librarians import LibrarianDirectory

    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "subjects": ["accounting", "audit fees"],
                },
                {
                    "id": "psych",
                    "name": "Psychology Librarian",
                    "subjects": ["behavioural science"],
                },
            ]
        }
    )


def _llm_config(**overrides):
    from primo_mcp_server.config import PrimoConfig

    values = {
        "librarian_semantic_fallback": False,
        "librarian_llm_fallback": True,
        "_env_file": None,
    }
    values.update(overrides)
    return PrimoConfig(**values)


def _patch_llm(monkeypatch, result, *, spy: list | None = None):
    from primo_mcp_server import recommendation as rec

    async def fake(directory, query, records, config, *, limit=2, **kwargs):
        if spy is not None:
            spy.append(query)
        return result

    monkeypatch.setattr(rec, "llm_fallback", fake)


async def test_llm_tier_does_not_run_when_keywords_already_matched(monkeypatch):
    """Tier 3 is the expensive path: a hit on tier 1 must short-circuit it."""
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    calls: list[str] = []
    _patch_llm(monkeypatch, LlmFallbackResult([]), spy=calls)

    outcome = await recommend_with_fallback(
        _llm_directory(),
        "audit fees",
        [],
        _llm_config(librarian_min_score=1.0),
    )

    assert outcome.matches
    assert calls == []


async def test_llm_tier_supplies_matches_when_earlier_tiers_miss(monkeypatch):
    from primo_mcp_server.librarian_llm import LlmFallbackResult
    from primo_mcp_server.librarians import LibrarianMatch, is_llm_match

    directory = _llm_directory()
    reasoned = LibrarianMatch(
        librarian=directory.librarians[1],
        score=0.83,
        matched_terms=["autism is behavioural science"],
        evidence_fields=["llm"],
    )
    calls: list[str] = []
    _patch_llm(monkeypatch, LlmFallbackResult([reasoned]), spy=calls)

    outcome = await recommend_with_fallback(
        directory, "autism", [], _llm_config(librarian_min_score=10_000.0)
    )

    assert [m.librarian.id for m in outcome.matches] == ["psych"]
    assert is_llm_match(outcome.matches[0])
    assert calls == ["autism"]
    # A validated tier-3 match must not also be shadowed by near-misses.
    assert outcome.near_misses == ()


async def test_llm_tier_is_skipped_on_the_latency_sensitive_inline_path(monkeypatch):
    """Inline recommendations ride every search and cannot afford a model call."""
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    calls: list[str] = []
    _patch_llm(monkeypatch, LlmFallbackResult([]), spy=calls)

    outcome = await recommend_with_fallback(
        _llm_directory(),
        "autism",
        [],
        _llm_config(librarian_min_score=10_000.0),
        embedding_timeout=2.5,
    )

    assert calls == []
    assert outcome.llm_skipped is not None
    assert "primo_recommend_librarians" in outcome.llm_skipped


async def test_llm_tier_runs_inline_when_explicitly_opted_in(monkeypatch):
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    calls: list[str] = []
    _patch_llm(monkeypatch, LlmFallbackResult([]), spy=calls)

    await recommend_with_fallback(
        _llm_directory(),
        "autism",
        [],
        _llm_config(librarian_min_score=10_000.0, librarian_llm_inline=True),
        embedding_timeout=2.5,
    )

    assert calls == ["autism"]


async def test_llm_tier_error_reaches_the_outcome(monkeypatch):
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    _patch_llm(monkeypatch, LlmFallbackResult([], error="ConnectError"))

    # "audit fees" scores against the accounting profile but cannot clear the
    # unreachable threshold, so a near-miss exists to be preserved. A broken
    # tier 3 must not cost the caller the evidence the earlier tiers found.
    outcome = await recommend_with_fallback(
        _llm_directory(), "audit fees", [], _llm_config(librarian_min_score=10_000.0)
    )

    assert outcome.llm_error == "ConnectError"
    assert [near.librarian.id for near in outcome.near_misses] == ["accounting"]
