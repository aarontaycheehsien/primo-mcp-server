"""Interactive triage: turn logged live queries into labelled eval cases.

The golden eval set only stays meaningful if it grows from real traffic
(AGENTS.md). This walks the opt-in recommendation log
(PRIMO_RECOMMEND_LOG_FILE) one new query at a time, shows what the server
decided and why, and asks the curator for the correct label. Nothing enters
the benchmark without a person seeing it.

Usage:
    python -m primo_mcp_server.triage_recommendations recommend-outcomes.jsonl librarian-eval.json
        [--fetch-records] [--since 2026-09-01]

At each prompt:
    Enter    accept the server's answer (its top pick, or "no librarian")
    ids      the correct librarian id(s), comma separated
    -        no librarian should be recommended
    s        skip; the query is remembered and not offered again
    q        save and quit

Every answer is saved immediately (write to a temporary file, then
replace), so quitting or a crash never loses labels. Log lines written
before the log carried record metadata have no ``records``;
``--fetch-records`` runs one live Primo search for those and freezes the
results into the case, so it stays deterministic afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.evaluate_recommendations import query_key
from primo_mcp_server.librarians import (
    LibrarianDirectory,
    load_librarian_directory_cached,
    matcher_evidence,
)

Ask = Callable[[str], str]
Write = Callable[[str], None]
RecordFetcher = Callable[[str], Awaitable[list[dict]]]

_SKIPPED_KEY = "triage_skipped"
_HELP = (
    "Enter = accept the server's answer | ids = correct librarian id(s), "
    "comma separated | - = no librarian | s = skip | q = save and quit"
)


def load_log(path: str | Path) -> tuple[list[dict], int]:
    """Parse the JSONL log, returning (entries, unusable line count)."""
    entries: list[dict] = []
    bad = 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(entry, dict) or not str(entry.get("query", "")).strip():
                bad += 1
                continue
            entries.append(entry)
    return entries, bad


def load_eval_file(path: str | Path) -> dict:
    """Load the eval file as a raw dict, or start an empty one."""
    path = Path(path)
    if not path.exists():
        return {"cases": []}
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"{path} has no 'cases' list.")
    return data


def save_eval_file(path: str | Path, data: dict) -> None:
    """Write atomically: a crash mid-write must not truncate the labels."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _entry_time(entry: dict) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(entry.get("time", "")))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def pending_entries(
    entries: list[dict], eval_data: dict, since: datetime | None = None
) -> list[dict]:
    """New queries to triage: latest entry per query, minus labelled/skipped.

    The log is append-only, so a later line for the same query is newer --
    and the one most likely to carry record metadata.
    """
    done = {
        query_key(str(case.get("query", "")))
        for case in eval_data["cases"]
        if isinstance(case, dict)
    }
    done |= {query_key(str(q)) for q in eval_data.get(_SKIPPED_KEY, [])}
    latest: dict[str, dict] = {}
    for entry in entries:
        when = _entry_time(entry)
        if since is not None and when is not None and when < since:
            continue
        key = query_key(entry["query"])
        if key in done:
            continue
        latest.pop(key, None)
        latest[key] = entry
    return list(latest.values())


def _server_ids(entry: dict) -> list[str]:
    return [str(m.get("id")) for m in entry.get("matches") or [] if m.get("id")]


def _server_summary(entry: dict) -> str:
    matches = entry.get("matches") or []
    if not matches:
        return "no match"
    top = matches[0]
    return f"{top.get('id')} (score {float(top.get('score', 0)):.1f})"


def _describe_match(match: dict, directory: LibrarianDirectory) -> str:
    names = {profile.id: profile.name for profile in directory.librarians}
    lid = str(match.get("id"))
    name = names.get(lid, "(no longer in the directory)")
    terms = ", ".join(match.get("terms") or []) or "none"
    fields = ", ".join(match.get("fields") or []) or "none"
    return (
        f"{lid} {name} (score {float(match.get('score', 0)):.1f}; "
        f"terms: {terms}; fields: {fields})"
    )


def describe_entry(
    entry: dict, directory: LibrarianDirectory, position: str, records: list
) -> list[str]:
    when = str(entry.get("time", ""))[:10] or "unknown date"
    lines = [
        "",
        f'{position} "{entry["query"]}"  ({when}, {entry.get("status", "?")}, '
        f"{len(records)} record(s) as evidence)",
    ]
    matches = entry.get("matches") or []
    if matches:
        lines.append("  Server picked:")
        lines.extend(f"    {_describe_match(m, directory)}" for m in matches)
    else:
        lines.append("  Server picked: nothing")
    near = entry.get("near_misses") or []
    if near:
        lines.append("  Near misses:")
        lines.extend(f"    {_describe_match(m, directory)}" for m in near)
    return lines


class Answer(NamedTuple):
    kind: str  # "label", "skip", "quit", "invalid"
    ids: list[str] = []
    message: str = ""


def parse_answer(text: str, default: list[str], known: set[str]) -> Answer:
    """Interpret one prompt reply. Unknown ids never become labels."""
    reply = text.strip()
    if reply.lower() == "q":
        return Answer("quit")
    if reply.lower() == "s":
        return Answer("skip")
    if reply == "-":
        return Answer("label", [])
    ids = default if reply == "" else [
        part for part in reply.replace(",", " ").split() if part
    ]
    unknown = [lid for lid in ids if lid not in known]
    if unknown:
        return Answer(
            "invalid",
            message=(
                f"Unknown id(s): {', '.join(unknown)}. "
                f"Configured ids: {', '.join(sorted(known))}."
            ),
        )
    return Answer("label", list(dict.fromkeys(ids)))


class TriageSummary(NamedTuple):
    labelled: int
    skipped: int
    remaining: int


async def triage(
    pending: list[dict],
    eval_data: dict,
    directory: LibrarianDirectory,
    save: Callable[[dict], None],
    *,
    ask: Ask | None = None,
    write: Write = print,
    fetch_records: RecordFetcher | None = None,
    today: date | None = None,
) -> TriageSummary:
    """Walk the pending queries, saving after every answer."""
    ask = ask or input
    known = {profile.id for profile in directory.librarians}
    stamp = (today or date.today()).isoformat()
    labelled = skipped = 0
    for index, entry in enumerate(pending):
        records = list(entry.get("records") or [])
        if not records and fetch_records is not None:
            try:
                records = await fetch_records(entry["query"])
            except Exception as e:  # a failed fetch must not end the session
                write(f"  (could not fetch records: {type(e).__name__}: {e})")
        for line in describe_entry(
            entry, directory, f"[{index + 1}/{len(pending)}]", records
        ):
            write(line)
        default = _server_ids(entry)[:1]
        while True:
            try:
                reply = ask("  Label> ")
            except EOFError:
                reply = "q"
            answer = parse_answer(reply, default, known)
            if answer.kind != "invalid":
                break
            write(f"  {answer.message}")
        if answer.kind == "quit":
            return TriageSummary(labelled, skipped, len(pending) - index)
        if answer.kind == "skip":
            eval_data.setdefault(_SKIPPED_KEY, []).append(entry["query"])
            skipped += 1
        else:
            case: dict = {
                "query": entry["query"],
                "expect": answer.ids,
                "note": f"triaged {stamp}; server: {_server_summary(entry)}",
            }
            if records:
                case["records"] = records
            eval_data["cases"].append(case)
            labelled += 1
        save(eval_data)
    return TriageSummary(labelled, skipped, 0)


def _parse_since(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f'Invalid --since "{value}". Use YYYY-MM-DD.'
        ) from e


async def _run(args: argparse.Namespace, config: PrimoConfig) -> int:
    directory, message, _ = load_librarian_directory_cached(config.librarians_file)
    if message or directory is None:
        print(message, file=sys.stderr)
        return 2
    try:
        entries, bad = load_log(args.log_path)
    except OSError as e:
        print(f"Cannot read {args.log_path}: {e}", file=sys.stderr)
        return 2
    try:
        eval_data = load_eval_file(args.eval_path)
    except (OSError, ValueError) as e:
        print(f"Cannot use {args.eval_path}: {e}", file=sys.stderr)
        return 2
    if bad:
        print(f"Ignored {bad} unusable log line(s).", file=sys.stderr)

    pending = pending_entries(entries, eval_data, args.since)
    if not pending:
        print("Nothing new to triage.")
        return 0
    print(f"{len(pending)} new quer{'y' if len(pending) == 1 else 'ies'} to triage.")
    print(_HELP)

    def save(data: dict) -> None:
        save_eval_file(args.eval_path, data)

    if not args.fetch_records:
        summary = await triage(pending, eval_data, directory, save)
    else:
        import httpx

        from primo_mcp_server.client import PrimoClient

        async with httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers={"User-Agent": config.user_agent},
        ) as http_client:
            client = PrimoClient(http_client, config)

            async def fetch(query: str) -> list[dict]:
                # The primo_search defaults: what an inline search would see.
                response = await client.search(
                    query=query, scope="everything", limit=10, include_facets=False
                )
                return [matcher_evidence(r) for r in response.records]

            summary = await triage(
                pending, eval_data, directory, save, fetch_records=fetch
            )

    print(
        f"\nLabelled {summary.labelled}, skipped {summary.skipped}, "
        f"{summary.remaining} left for next time. Saved to {args.eval_path}."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="primo-triage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log_path", help="Recommendation log (JSONL).")
    parser.add_argument("eval_path", help="Eval file to append labelled cases to.")
    parser.add_argument(
        "--fetch-records",
        action="store_true",
        help="Search Primo live for log lines that carry no record metadata.",
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="Only triage log entries from this date on (YYYY-MM-DD).",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args, PrimoConfig()))
    except KeyboardInterrupt:
        print("\nStopped. Every answer given so far is saved.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
