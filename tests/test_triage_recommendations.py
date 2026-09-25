"""Tests for the interactive log-to-eval triage command."""

from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone

from primo_mcp_server.evaluate_recommendations import EvalSet
from primo_mcp_server.librarians import LibrarianDirectory
from primo_mcp_server.triage_recommendations import (
    load_eval_file,
    load_log,
    main,
    parse_answer,
    pending_entries,
    save_eval_file,
    triage,
)


def _directory() -> LibrarianDirectory:
    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "5", "name": "Policy Librarian", "subjects": ["policy"]},
                {"id": "7", "name": "Law Librarian", "subjects": ["law"]},
            ]
        }
    )


def _entry(query: str, *, matched: str | None = None, time: str = "2026-09-20T10:00:00+00:00", records=None) -> dict:
    entry = {
        "time": time,
        "query": query,
        "status": "matched" if matched else "no_match",
        "matches": (
            [{"id": matched, "score": 13.5, "terms": ["policy"], "fields": ["query"]}]
            if matched
            else []
        ),
        "near_misses": [],
    }
    if records is not None:
        entry["records"] = records
    return entry


def _session(replies: list[str]):
    """A scripted terminal: canned replies in, printed lines captured."""
    queue = list(replies)
    output: list[str] = []
    saves: list[dict] = []

    def ask(_prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return ask, output.append, (lambda data: saves.append(copy.deepcopy(data))), output, saves


async def _run(pending, eval_data, replies, **kwargs):
    ask, write, save, output, saves = _session(replies)
    summary = await triage(
        pending, eval_data, _directory(), save,
        ask=ask, write=write, today=date(2026, 9, 25), **kwargs,
    )
    return summary, output, saves


# -- answers ----------------------------------------------------------------


def test_parse_answer_covers_every_reply():
    known = {"5", "7"}
    assert parse_answer("", ["5"], known) == ("label", ["5"], "")
    assert parse_answer("-", ["5"], known).ids == []
    assert parse_answer(" 7, 5 ", ["5"], known).ids == ["7", "5"]
    assert parse_answer("s", ["5"], known).kind == "skip"
    assert parse_answer("Q", ["5"], known).kind == "quit"
    invalid = parse_answer("9", ["5"], known)
    assert invalid.kind == "invalid" and "9" in invalid.message


def test_enter_on_a_default_no_longer_in_the_directory_is_rejected():
    assert parse_answer("", ["42"], {"5"}).kind == "invalid"


# -- session ----------------------------------------------------------------


async def test_enter_accepts_the_server_pick_and_saves_records():
    records = [{"title": "Housing policy", "subjects": ["Public policy"]}]
    eval_data = {"cases": []}

    summary, output, saves = await _run(
        [_entry("housing policy", matched="5", records=records)], eval_data, [""]
    )

    assert summary == (1, 0, 0)
    case = eval_data["cases"][0]
    assert case == {
        "query": "housing policy",
        "expect": ["5"],
        "note": "triaged 2026-09-25; server: 5 (score 13.5)",
        "records": records,
    }
    assert len(saves) == 1
    assert any("Policy Librarian" in line for line in output)
    # A triaged case must load as a benchmark case.
    EvalSet.model_validate(eval_data)


async def test_dash_labels_no_librarian_and_s_remembers_the_skip():
    eval_data = {"cases": []}

    summary, _, _ = await _run(
        [_entry("tropical fish", matched="5"), _entry("weather")],
        eval_data,
        ["-", "s"],
    )

    assert summary == (1, 1, 0)
    assert eval_data["cases"][0]["expect"] == []
    assert eval_data["triage_skipped"] == ["weather"]


async def test_typo_reprompts_instead_of_saving():
    eval_data = {"cases": []}

    _, output, saves = await _run([_entry("law")], eval_data, ["77", "7"])

    assert eval_data["cases"][0]["expect"] == ["7"]
    assert any("Unknown id(s): 77" in line for line in output)
    assert len(saves) == 1


async def test_quit_keeps_earlier_answers_and_counts_the_rest():
    eval_data = {"cases": []}

    summary, _, saves = await _run(
        [_entry("a", matched="5"), _entry("b"), _entry("c")], eval_data, ["", "q"]
    )

    assert summary == (1, 0, 2)
    assert [c["query"] for c in saves[-1]["cases"]] == ["a"]


async def test_end_of_input_is_treated_as_quit():
    summary, _, _ = await _run([_entry("a")], {"cases": []}, [])

    assert summary == (0, 0, 1)


async def test_records_are_fetched_only_for_entries_without_them():
    fetched: list[str] = []

    async def fetch(query: str) -> list[dict]:
        fetched.append(query)
        return [{"title": f"live {query}"}]

    eval_data = {"cases": []}
    await _run(
        [_entry("old line"), _entry("new line", records=[{"title": "logged"}])],
        eval_data,
        ["-", "-"],
        fetch_records=fetch,
    )

    assert fetched == ["old line"]
    assert eval_data["cases"][0]["records"] == [{"title": "live old line"}]
    assert eval_data["cases"][1]["records"] == [{"title": "logged"}]


async def test_failed_fetch_still_offers_the_query():
    async def fetch(query: str) -> list[dict]:
        raise RuntimeError("offline")

    eval_data = {"cases": []}
    _, output, _ = await _run([_entry("q1")], eval_data, ["-"], fetch_records=fetch)

    assert eval_data["cases"][0]["query"] == "q1"
    assert "records" not in eval_data["cases"][0]
    assert any("could not fetch records" in line for line in output)


# -- selecting what to triage ----------------------------------------------


def test_pending_keeps_latest_entry_and_drops_labelled_or_skipped():
    entries = [
        _entry("Housing Policy", time="2026-09-01T00:00:00+00:00"),
        _entry("housing  policy", matched="5", time="2026-09-10T00:00:00+00:00"),
        _entry("already labelled"),
        _entry("already skipped"),
        _entry("fresh"),
    ]
    eval_data = {
        "cases": [{"query": "Already Labelled", "expect": []}],
        "triage_skipped": ["already skipped"],
    }

    pending = pending_entries(entries, eval_data)

    assert [e["query"] for e in pending] == ["housing  policy", "fresh"]
    assert pending[0]["status"] == "matched"


def test_since_filters_older_entries():
    entries = [
        _entry("old", time="2026-08-01T00:00:00+00:00"),
        _entry("new", time="2026-09-20T00:00:00+00:00"),
    ]
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)

    assert [e["query"] for e in pending_entries(entries, {"cases": []}, since)] == ["new"]


# -- files ------------------------------------------------------------------


def test_log_loader_skips_unusable_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        json.dumps(_entry("good")) + "\n" + "not json\n" + json.dumps({"query": ""}) + "\n\n",
        encoding="utf-8",
    )

    entries, bad = load_log(path)

    assert [e["query"] for e in entries] == ["good"]
    assert bad == 2


def test_save_preserves_existing_cases_and_writes_ascii(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"cases": [{"query": "existing", "expect": ["5"], "note": "keep"}]}),
        encoding="utf-8",
    )
    data = load_eval_file(path)
    data["cases"].append({"query": "café culture", "expect": []})

    save_eval_file(path, data)

    raw = path.read_text(encoding="utf-8")
    assert raw.isascii()
    reloaded = json.loads(raw)
    assert reloaded["cases"][0] == {"query": "existing", "expect": ["5"], "note": "keep"}
    assert reloaded["cases"][1]["query"] == "café culture"
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left behind


def test_missing_eval_file_starts_empty(tmp_path):
    assert load_eval_file(tmp_path / "new.json") == {"cases": []}


def test_cli_session_end_to_end(tmp_path, monkeypatch, capsys):
    from primo_mcp_server import triage_recommendations as tr
    from primo_mcp_server.config import PrimoConfig

    directory_path = tmp_path / "librarians.json"
    directory_path.write_text(json.dumps(_directory().model_dump()), encoding="utf-8")
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(e) for e in [_entry("law", matched="7"), _entry("fish")]) + "\n",
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    monkeypatch.setattr(
        tr, "PrimoConfig",
        lambda: PrimoConfig(_env_file=None, librarians_file=str(directory_path)),
    )
    replies = iter(["", "s"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    assert main([str(log_path), str(eval_path)]) == 0

    saved = json.loads(eval_path.read_text(encoding="utf-8"))
    assert saved["cases"][0]["expect"] == ["7"]
    assert saved["triage_skipped"] == ["fish"]
    assert "Labelled 1, skipped 1, 0 left" in capsys.readouterr().out

    # A second run has nothing left to offer.
    assert main([str(log_path), str(eval_path)]) == 0
    assert "Nothing new to triage." in capsys.readouterr().out
