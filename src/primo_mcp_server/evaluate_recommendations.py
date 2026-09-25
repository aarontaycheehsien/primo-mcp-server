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
    _MAX_RECOMMENDATIONS,
    LibrarianDirectory,
    is_llm_match,
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
    path: str  # "identifier-skip", "keyword", "semantic", "llm", "mixed", "none"
    semantic_error: str | None = None
    semantic_skipped: str | None = None
    llm_error: str | None = None
    llm_skipped: str | None = None


class EvalReport(BaseModel):
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)


def _load_eval_set(path: str) -> tuple[EvalSet | None, str | None]:
    try:
        with open(path, encoding="utf-8-sig") as f:
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


def _match_path(result_matches) -> str:
    if not result_matches:
        return "none"
    tiers = {
        "semantic" if is_semantic_match(match)
        else "llm" if is_llm_match(match)
        else "keyword"
        for match in result_matches
    }
    return tiers.pop() if len(tiers) == 1 else "mixed"


# Providers that need a live MCP session; the offline harness has none.
_SESSION_ONLY_LLM_PROVIDERS = {"caller", "sampling"}


def _run_config(config: PrimoConfig, keyword_only: bool) -> tuple[PrimoConfig, str | None]:
    """Config for the benchmark run, plus a warning to print if any.

    ``--keyword-only`` must switch off BOTH fallback tiers, or an enabled
    LLM tier would still add matches to a supposedly keyword-only number.
    """
    if keyword_only:
        return (
            config.model_copy(
                update={
                    "librarian_semantic_fallback": False,
                    "librarian_llm_fallback": False,
                }
            ),
            None,
        )
    provider = config.llm_provider.strip().lower()
    if config.librarian_llm_fallback and provider in _SESSION_ONLY_LLM_PROVIDERS:
        return (
            config.model_copy(update={"librarian_llm_fallback": False}),
            f'Warning: the LLM fallback provider "{provider}" needs a live MCP '
            "session, so the LLM tier is off for this run. Set "
            "PRIMO_LLM_PROVIDER=openai to measure it offline.",
        )
    return config, None


def _llm_status(config: PrimoConfig) -> str:
    if not config.librarian_llm_fallback:
        return "off"
    return f"on ({config.llm_provider.strip().lower()})"


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
                path=_match_path(outcome.matches),
                semantic_error=outcome.semantic_error,
                semantic_skipped=outcome.semantic_skipped,
                llm_error=outcome.llm_error,
                llm_skipped=outcome.llm_skipped,
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
            if result.llm_error:
                line += f" (LLM error: {result.llm_error})"
            elif result.llm_skipped:
                line += f" (LLM skipped: {result.llm_skipped})"
            if result.case.note:
                line += f" -- {result.case.note}"
            print(line)
        print()

    print(f"Cases: {len(report.results)}")
    if match_cases:
        top1 = sum(r.passed for r in match_cases)
        hits = sum(r.hit for r in match_cases)
        # The pipeline caps every request, so a larger --limit returns no more.
        shown = min(max(1, limit), _MAX_RECOMMENDATIONS)
        print(
            f"Match cases: {len(match_cases)}  "
            f"top-1 accuracy: {top1}/{len(match_cases)} "
            f"({top1 / len(match_cases):.0%})  "
            f"hit@{shown}: {hits}/{len(match_cases)} "
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
    llm_errors = sum(1 for r in report.results if r.llm_error)
    if llm_errors:
        print(
            f"Warning: the LLM fallback errored on {llm_errors} case(s); "
            "those cases did not measure the LLM tier."
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

    config, warning = _run_config(PrimoConfig(), args.keyword_only)
    if warning:
        print(warning, file=sys.stderr)

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
        f"semantic fallback: {'on' if config.librarian_semantic_fallback else 'off'}  "
        f"LLM fallback: {_llm_status(config)}"
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
