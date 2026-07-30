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
