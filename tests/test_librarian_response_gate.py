"""Tests for the shared Claude Code/Codex librarian response hook."""

from __future__ import annotations

import json
from pathlib import Path

from hooks.librarian_response_gate import (
    Recommendation,
    extract_recommendations,
    handle_event,
)


_MATCHED_OUTPUT = """## Recommended librarian help:

Status: matched
1. Name: [Aaron Tay](https://library.example.edu/aaron)
   Title: Head, Data Services
   Contact: aarontay@example.edu
   Evidence: matched terms: systematic review; evidence fields: query
   Reasoning: Selected because systematic review expertise matched the query.
2. Name: [Bella Ratmelia](mailto:bella@example.edu)
   Title: Senior Librarian
   Contact: bella@example.edu
   Evidence: matched terms: data extraction; evidence fields: keywords
   Reasoning: Selected because data extraction expertise matched the records.
Recommendations are limited to configured librarian profiles.
"""


def _event(name: str, **overrides) -> dict:
    payload = {
        "session_id": "session-123",
        "cwd": "C:/workspace/primo",
        "hook_event_name": name,
    }
    payload.update(overrides)
    return payload


def _post_tool(tool_name: str = "mcp__primo__primo_recommend_librarians") -> dict:
    return _event(
        "PostToolUse",
        tool_name=tool_name,
        tool_response={"content": [{"type": "text", "text": _MATCHED_OUTPUT}]},
    )


def test_extract_recommendations_builds_exact_template():
    recommendations = extract_recommendations({"result": _MATCHED_OUTPUT})

    assert recommendations == [
        Recommendation(
            name="Aaron Tay",
            url="https://library.example.edu/aaron",
            email="aarontay@example.edu",
            reason="Selected because systematic review expertise matched the query.",
        ),
        Recommendation(
            name="Bella Ratmelia",
            url="mailto:bella@example.edu",
            email="bella@example.edu",
            reason="Selected because data extraction expertise matched the records.",
        ),
    ]
    assert recommendations[0].render() == (
        "- [Aaron Tay](https://library.example.edu/aaron) - "
        "<aarontay@example.edu>\n"
        "  Reason: Selected because systematic review expertise matched the query."
    )


def test_no_match_output_does_not_create_a_requirement(tmp_path):
    post = _event(
        "PostToolUse",
        tool_name="mcp__primo__primo_recommend_librarians",
        tool_response={
            "result": "## Recommended librarian help:\n\nStatus: no_match"
        },
    )

    assert handle_event(post, state_dir=tmp_path) is None
    assert (
        handle_event(
            _event("Stop", last_assistant_message="No recommendation was returned."),
            state_dir=tmp_path,
        )
        is None
    )


def test_post_tool_use_injects_the_required_blocks(tmp_path):
    output = handle_event(_post_tool(), state_dir=tmp_path)

    context = output["hookSpecificOutput"]["additionalContext"]
    assert output["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "- [Aaron Tay](https://library.example.edu/aaron)" in context
    assert "<aarontay@example.edu>" in context
    assert "  Reason: Selected because systematic review expertise" in context
    assert "- [Bella Ratmelia](mailto:bella@example.edu)" in context


def test_inline_search_recommendations_are_captured(tmp_path):
    output = handle_event(
        _post_tool("mcp__primo__primo_search"),
        state_dir=tmp_path,
    )

    assert output is not None
    assert "every recommendation" in output["hookSpecificOutput"]["additionalContext"]


def test_unrelated_tool_is_ignored(tmp_path):
    assert (
        handle_event(
            _post_tool("mcp__primo__primo_get_record"),
            state_dir=tmp_path,
        )
        is None
    )


def test_stop_blocks_when_any_template_block_is_missing(tmp_path):
    handle_event(_post_tool(), state_dir=tmp_path)
    only_one = (
        "Search results.\n\n"
        "- [Aaron Tay](https://library.example.edu/aaron) - "
        "<aarontay@example.edu>\n"
        "  Reason: Selected because systematic review expertise matched the query."
    )

    output = handle_event(
        _event("Stop", last_assistant_message=only_one),
        state_dir=tmp_path,
    )

    assert output["decision"] == "block"
    assert "- [Bella Ratmelia](mailto:bella@example.edu)" in output["reason"]
    assert "include every block below exactly" in output["reason"]


def test_stop_blocks_a_paraphrased_or_missing_reason(tmp_path):
    handle_event(_post_tool(), state_dir=tmp_path)
    message = (
        "- [Aaron Tay](https://library.example.edu/aaron) - "
        "<aarontay@example.edu>\n"
        "  Reason: Helpful for reviews.\n\n"
        "- [Bella Ratmelia](mailto:bella@example.edu) - <bella@example.edu>\n"
        "  Reason: Helpful for data."
    )

    output = handle_event(
        _event("Stop", last_assistant_message=message),
        state_dir=tmp_path,
    )

    assert output["decision"] == "block"
    assert "server-provided Reason" in output["reason"]


def test_stop_allows_complete_exact_template_and_clears_state(tmp_path):
    post_output = handle_event(_post_tool(), state_dir=tmp_path)
    required = post_output["hookSpecificOutput"]["additionalContext"].split(
        "reason:\n\n", 1
    )[1]

    assert (
        handle_event(
            _event("Stop", last_assistant_message=f"Results.\n\n{required}"),
            state_dir=tmp_path,
        )
        is None
    )
    # Successful validation removes the turn state, so a later Stop is clean.
    assert (
        handle_event(
            _event("Stop", last_assistant_message="A later response."),
            state_dir=tmp_path,
        )
        is None
    )


def test_repeated_tool_output_is_deduplicated(tmp_path):
    handle_event(_post_tool(), state_dir=tmp_path)
    output = handle_event(_post_tool(), state_dir=tmp_path)
    context = output["hookSpecificOutput"]["additionalContext"]

    assert context.count("- [Aaron Tay]") == 1
    assert context.count("- [Bella Ratmelia]") == 1


def test_new_prompt_clears_previous_recommendations(tmp_path):
    handle_event(_post_tool(), state_dir=tmp_path)
    handle_event(_event("UserPromptSubmit", prompt="New question"), state_dir=tmp_path)

    assert (
        handle_event(
            _event("Stop", last_assistant_message="No librarians in this turn."),
            state_dir=tmp_path,
        )
        is None
    )


def test_claude_and_codex_configs_register_the_same_gate():
    root = Path(__file__).parents[1]
    claude = json.loads((root / ".claude" / "settings.json").read_text())
    codex = json.loads((root / ".codex" / "hooks.json").read_text())

    for config in (claude, codex):
        hooks = config["hooks"]
        assert set(hooks) == {"UserPromptSubmit", "PostToolUse", "Stop"}
        assert hooks["PostToolUse"][0]["matcher"] == (
            "^mcp__primo__primo_(search|recommend_librarians)$"
        )
        for event in hooks.values():
            handler = event[0]["hooks"][0]
            command_text = " ".join(
                [handler.get("command", ""), *handler.get("args", [])]
            )
            assert "librarian_response_gate.py" in command_text

    codex_handlers = [
        entries[0]["hooks"][0] for entries in codex["hooks"].values()
    ]
    assert all("commandWindows" in handler for handler in codex_handlers)
