"""Tests for the offline recommendation evaluation harness."""

from __future__ import annotations

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.evaluate_recommendations import (
    EvalSet,
    _load_eval_set,
    _run_config,
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
        _llm_config(librarian_min_score=10_000.0, librarian_llm_inline=False),
        embedding_timeout=2.5,
    )

    assert calls == []
    assert outcome.llm_skipped is not None
    assert "primo_recommend_librarians" in outcome.llm_skipped


async def test_llm_tier_does_not_run_inline_by_default(monkeypatch):
    """Every recommendation switch ships off, including the inline one."""
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    calls: list[str] = []
    _patch_llm(monkeypatch, LlmFallbackResult([]), spy=calls)

    await recommend_with_fallback(
        _llm_directory(),
        "autism",
        [],
        _llm_config(librarian_min_score=10_000.0),
        embedding_timeout=2.5,
    )

    assert calls == []


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


# ---------------------------------------------------------------------------
# Benchmark reporting of the LLM tier.
# ---------------------------------------------------------------------------


async def test_evaluate_reports_llm_matches_as_llm_path(monkeypatch):
    from primo_mcp_server.librarian_llm import LlmFallbackResult
    from primo_mcp_server.librarians import LibrarianMatch

    directory = _llm_directory()
    reasoned = LibrarianMatch(
        librarian=directory.librarians[1],
        score=0.83,
        matched_terms=["autism is behavioural science"],
        evidence_fields=["llm"],
    )
    _patch_llm(monkeypatch, LlmFallbackResult([reasoned]))

    report = await evaluate(
        _eval_set([{"query": "autism", "expect": ["psych"]}]),
        directory,
        _llm_config(llm_provider="openai"),
    )

    assert report.results[0].path == "llm"


async def test_evaluate_carries_llm_errors_into_results(monkeypatch):
    from primo_mcp_server.librarian_llm import LlmFallbackResult

    _patch_llm(monkeypatch, LlmFallbackResult([], error="ConnectError"))

    report = await evaluate(
        _eval_set([{"query": "autism", "expect": ["psych"]}]),
        _llm_directory(),
        _llm_config(llm_provider="openai"),
    )

    assert report.results[0].llm_error == "ConnectError"


def test_keyword_only_switches_off_both_fallback_tiers():
    config, warning = _run_config(
        _llm_config(librarian_semantic_fallback=True, llm_provider="openai"),
        keyword_only=True,
    )

    assert config.librarian_semantic_fallback is False
    assert config.librarian_llm_fallback is False
    assert warning is None


def test_session_only_llm_provider_is_switched_off_with_a_warning():
    config, warning = _run_config(
        _llm_config(llm_provider="caller"), keyword_only=False
    )

    assert config.librarian_llm_fallback is False
    assert warning is not None and "caller" in warning


def test_openai_llm_provider_stays_on_for_a_full_run():
    config, warning = _run_config(
        _llm_config(llm_provider="openai"), keyword_only=False
    )

    assert config.librarian_llm_fallback is True
    assert warning is None


def test_eval_set_saved_with_byte_order_mark_loads(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        '{"cases": [{"query": "law", "expect": ["law"]}]}', encoding="utf-8-sig"
    )

    eval_set, error = _load_eval_set(str(path))

    assert error is None
    assert eval_set is not None and eval_set.cases[0].query == "law"


# ---------------------------------------------------------------------------
# Saved runs and regression comparison.
# ---------------------------------------------------------------------------


def _report(cases: list[tuple[str, list[str], list[str]]]):
    """Build an EvalReport from (query, expect, got_ids) triples."""
    from primo_mcp_server.evaluate_recommendations import (
        CaseResult,
        EvalCase,
        EvalReport,
    )

    results = []
    for query, expect, got in cases:
        passed = (bool(got) and got[0] in expect) if expect else not got
        results.append(
            CaseResult(
                case=EvalCase(query=query, expect=expect),
                got_ids=got,
                passed=passed,
                hit=passed,
                path="keyword" if got else "none",
            )
        )
    return EvalReport(results=results)


def test_compare_classifies_each_kind_of_change():
    from primo_mcp_server.evaluate_recommendations import (
        compare_results,
        results_payload,
    )

    before = results_payload(
        _report(
            [
                ("audit fees", ["accounting"], ["accounting"]),
                ("case law", ["law"], ["accounting"]),
                ("tax law", ["law"], ["law", "accounting"]),
                ("marine biology", [], []),
                ("dropped query", ["law"], ["law"]),
            ]
        )
    )
    after = _report(
        [
            ("Audit  Fees", ["accounting"], []),  # same case, folded key
            ("case law", ["law"], ["law"]),
            ("tax law", ["law", "accounting"], ["accounting"]),
            ("marine biology", [], []),
            ("new query", ["law"], ["law"]),
        ]
    )

    comparison = compare_results(before, after)

    assert comparison.newly_failing == ["Audit  Fees (was accounting, now none)"]
    assert comparison.newly_passing == ["case law (was accounting, now law)"]
    assert comparison.pick_changed == ["tax law (law -> accounting)"]
    assert comparison.added == ["new query"]
    assert comparison.removed == ["dropped query"]
    assert not comparison.unchanged


def test_identical_runs_compare_as_unchanged():
    from primo_mcp_server.evaluate_recommendations import (
        compare_results,
        results_payload,
    )

    report = _report([("audit fees", ["accounting"], ["accounting"])])

    assert compare_results(results_payload(report), report).unchanged


def test_saved_results_round_trip_through_the_cli(tmp_path, monkeypatch, capsys):
    import json
    import sys

    import pytest

    from primo_mcp_server import evaluate_recommendations as ev

    directory_path = tmp_path / "librarians.json"
    directory_path.write_text(
        json.dumps(_directory().model_dump()), encoding="utf-8"
    )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {"cases": [{"query": "accounting datasets for audit fees",
                        "expect": ["accounting"]}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ev, "PrimoConfig",
        lambda: PrimoConfig(_env_file=None, librarians_file=str(directory_path)),
    )
    baseline = tmp_path / "baseline.json"

    def run(*extra: str) -> int:
        monkeypatch.setattr(sys, "argv", ["primo-eval", str(eval_path), *extra])
        with pytest.raises(SystemExit) as exc_info:
            ev.main()
        return exc_info.value.code

    assert run("--save-results", str(baseline)) == 0
    assert json.loads(baseline.read_text())["cases"][0]["got_ids"] == ["accounting"]

    assert run("--compare", str(baseline), "--fail-on-regression") == 0
    assert "No changes." in capsys.readouterr().out

    # Relabel the case so the same output now fails: a regression.
    eval_path.write_text(
        json.dumps(
            {"cases": [{"query": "accounting datasets for audit fees",
                        "expect": ["law"]}]}
        ),
        encoding="utf-8",
    )
    assert run("--compare", str(baseline), "--fail-on-regression") == 1
    assert "Newly failing (1)" in capsys.readouterr().out


def test_report_counts_cases_with_record_evidence(capsys):
    from primo_mcp_server.evaluate_recommendations import (
        CaseResult,
        EvalCase,
        EvalReport,
        _print_report,
    )
    from primo_mcp_server.models import PrimoRecord

    report = EvalReport(
        results=[
            CaseResult(
                case=EvalCase(query="a", records=[PrimoRecord(title="t")]),
                got_ids=[], passed=True, hit=False, path="none",
            ),
            CaseResult(
                case=EvalCase(query="b"),
                got_ids=[], passed=True, hit=False, path="none",
            ),
        ]
    )

    _print_report(report, 3)

    assert "Cases with record evidence: 1/2" in capsys.readouterr().out
