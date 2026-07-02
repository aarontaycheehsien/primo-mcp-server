"""Offline accuracy benchmark for librarian recommendations.

Runs a golden set of labelled queries through the exact recommendation
pipeline the server uses (``recommendation.recommend_with_fallback``) and
reports top-1 accuracy, hit rate within the returned list, and the
false-positive rate on queries that should return nothing. Tuning changes
to weights, thresholds, or the semantic path can then be judged by a
measured delta instead of anecdote.

Eval file shape:

    {
      "cases": [
        {
          "query": "screening tools for a systematic review",
          "expect": ["1"],
          "note": "optional curator note",
          "records": [{"title": "...", "subjects": ["..."]}]
        },
        {"query": "tropical marine biology", "expect": []}
      ]
    }

``expect`` lists the acceptable librarian ids -- a case passes when the
top recommendation is any of them. An empty ``expect`` means the correct
outcome is NO recommendation; such cases measure false positives, which
matter as much as hits. ``records`` optionally supplies fixed Primo record
metadata as corroborating evidence, keeping the benchmark deterministic
and offline instead of depending on live search results.

Usage:
    python -m primo_mcp_server.evaluate_recommendations eval.json
        [--keyword-only] [--limit 3] [--min-pass-rate 0.9]

The semantic fallback runs exactly when the server would run it (enabled,
API key configured, keyword score weak), so with semantic enabled each
weak-keyword case costs one query embedding. ``--keyword-only`` forces the
deterministic path alone. Exit codes: 0 when the pass rate meets
``--min-pass-rate`` (default 0, informational), 1 below it, 2 unusable
input.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import BaseModel, Field, ValidationError

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarians import (
    LibrarianDirectory,
    is_semantic_match,
    load_librarian_directory_cached,
    looks_like_identifier,
)
from primo_mcp_server.models import PrimoRecord
from primo_mcp_server.recommendation import recommend_with_fallback


class EvalCase(BaseModel):
    """One labelled query. Empty ``expect`` means expect no recommendation."""

    query: str
    expect: list[str] = Field(default_factory=list)
    note: str = ""
    records: list[PrimoRecord] = Field(default_factory=list)


class EvalSet(BaseModel):
    cases: list[EvalCase]


class CaseResult(BaseModel):
    case: EvalCase
    got_ids: list[str]
    passed: bool
    hit: bool
    path: str  # "identifier-skip", "keyword", "semantic", "mixed", "none"
    semantic_error: str | None = None
    semantic_skipped: str | None = None


class EvalReport(BaseModel):
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)


def _load_eval_set(path: str) -> tuple[EvalSet | None, str | None]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        return None, f"Cannot read {path}: {e}"
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in {path} at line {e.lineno}."
    try:
        eval_set = EvalSet.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        return None, f"Invalid eval case: {first['msg']} at {first['loc']}."
    if not eval_set.cases:
        return None, f"{path} contains no cases."
    return eval_set, None


def _unknown_expect_ids(
    eval_set: EvalSet, directory: LibrarianDirectory
) -> list[str]:
    """Expected ids that no profile has -- almost always a label typo.

    Left unchecked, a typo would make its case silently unpassable and
    quietly depress every future benchmark run.
    """
    known = {librarian.id for librarian in directory.librarians}
    unknown: list[str] = []
    for case in eval_set.cases:
        for expected in case.expect:
            if expected not in known and expected not in unknown:
                unknown.append(expected)
    return unknown


def _match_path(result_matches, semantic_skipped: str | None) -> str:
    if not result_matches:
        return "none"
    semantic = [is_semantic_match(match) for match in result_matches]
    if all(semantic):
        return "semantic"
    if any(semantic):
        return "mixed"
    return "keyword"


async def evaluate(
    eval_set: EvalSet,
    directory: LibrarianDirectory,
    config: PrimoConfig,
    *,
    specificity: dict[str, float] | None = None,
    limit: int = 3,
) -> EvalReport:
    """Run every case through the server's recommendation pipeline."""
    results: list[CaseResult] = []
    for case in eval_set.cases:
        if looks_like_identifier(case.query):
            got_ids: list[str] = []
            results.append(
                CaseResult(
                    case=case,
                    got_ids=got_ids,
                    passed=not case.expect,
                    hit=False,
                    path="identifier-skip",
                )
            )
            continue

        outcome = await recommend_with_fallback(
            directory,
            case.query,
            case.records,
            config,
            limit=limit,
            specificity=specificity,
        )
        got_ids = [match.librarian.id for match in outcome.matches]
        if case.expect:
            passed = bool(got_ids) and got_ids[0] in case.expect
            hit = any(got in case.expect for got in got_ids)
        else:
            passed = not got_ids
            hit = False
        results.append(
            CaseResult(
                case=case,
                got_ids=got_ids,
                passed=passed,
                hit=hit,
                path=_match_path(outcome.matches, outcome.semantic_skipped),
                semantic_error=outcome.semantic_error,
                semantic_skipped=outcome.semantic_skipped,
            )
        )
    return EvalReport(results=results)


def _print_report(report: EvalReport, limit: int) -> None:
    match_cases = [r for r in report.results if r.case.expect]
    no_match_cases = [r for r in report.results if not r.case.expect]

    failures = [r for r in report.results if not r.passed]
    if failures:
        print("Failures:")
        for result in failures:
            expected = ", ".join(result.case.expect) or "(no recommendation)"
            got = ", ".join(result.got_ids) or "(no recommendation)"
            line = (
                f'- "{result.case.query}" -> expected {expected}; '
                f"got {got} [{result.path}]"
            )
            if result.semantic_error:
                line += f" (semantic error: {result.semantic_error})"
            elif result.semantic_skipped:
                line += f" (semantic skipped: {result.semantic_skipped})"
            if result.case.note:
                line += f" -- {result.case.note}"
            print(line)
        print()

    print(f"Cases: {len(report.results)}")
    if match_cases:
        top1 = sum(r.passed for r in match_cases)
        hits = sum(r.hit for r in match_cases)
        print(
            f"Match cases: {len(match_cases)}  "
            f"top-1 accuracy: {top1}/{len(match_cases)} "
            f"({top1 / len(match_cases):.0%})  "
            f"hit@{limit}: {hits}/{len(match_cases)} "
            f"({hits / len(match_cases):.0%})"
        )
    if no_match_cases:
        rejected = sum(r.passed for r in no_match_cases)
        print(
            f"No-match cases: {len(no_match_cases)}  "
            f"correct rejections: {rejected}/{len(no_match_cases)} "
            f"({rejected / len(no_match_cases):.0%})"
        )
    semantic_errors = sum(1 for r in report.results if r.semantic_error)
    if semantic_errors:
        print(
            f"Warning: the semantic fallback errored on {semantic_errors} "
            "case(s); those cases measured the keyword path only."
        )
    print(f"Overall pass rate: {report.pass_rate:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="primo-eval",
        description="Benchmark librarian recommendations against a golden query set.",
    )
    parser.add_argument("eval_path", help="JSON file of labelled queries.")
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="Force the deterministic keyword path (no embedding calls).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Recommendations requested per query (default 3, the server cap).",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exit 1 when the overall pass rate falls below this (0-1).",
    )
    args = parser.parse_args()

    config = PrimoConfig()
    if args.keyword_only:
        config = config.model_copy(
            update={"librarian_semantic_fallback": False}
        )

    directory, message, specificity = load_librarian_directory_cached(
        config.librarians_file
    )
    if message or directory is None:
        print(message, file=sys.stderr)
        sys.exit(2)

    eval_set, error = _load_eval_set(args.eval_path)
    if error or eval_set is None:
        print(error, file=sys.stderr)
        sys.exit(2)

    unknown = _unknown_expect_ids(eval_set, directory)
    if unknown:
        print(
            "Expected id(s) not in the configured directory (label typo?): "
            + ", ".join(unknown),
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"Directory: {config.librarians_file} "
        f"({len(directory.librarians)} profiles)  "
        f"semantic fallback: {'on' if config.librarian_semantic_fallback else 'off'}"
    )
    report = asyncio.run(
        evaluate(
            eval_set,
            directory,
            config,
            specificity=specificity,
            limit=args.limit,
        )
    )
    _print_report(report, args.limit)
    sys.exit(0 if report.pass_rate >= args.min_pass_rate else 1)


if __name__ == "__main__":
    main()
