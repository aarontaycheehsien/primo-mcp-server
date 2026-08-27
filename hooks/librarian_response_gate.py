"""Enforce the user-facing librarian recommendation template.

Claude Code and Codex both send lifecycle-hook input as JSON on stdin. This
module deliberately depends only on the Python standard library so the same
script can run from either agent without activating the MCP server environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_PRIMO_TOOL_RE = re.compile(
    r"(?:^|__)primo_(?:search|recommend_librarians)$",
    re.IGNORECASE,
)
_MATCHED_STATUS_RE = re.compile(
    r"^Status:\s*matched(?:\s*\(semantic fallback\))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ENTRY_RE = re.compile(
    r"^\d+\.\s+Name:\s+\[(?P<name>[^\]\r\n]+)\]"
    r"\((?P<url>[^\r\n]+)\)\s*$"
    r"(?P<body>.*?)"
    r"(?=^\d+\.\s+Name:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CONTACT_RE = re.compile(r"^\s+Contact:\s*(?P<contact>.+?)\s*$", re.MULTILINE)
_REASONING_RE = re.compile(
    r"^\s+Reasoning:\s*(?P<reason>.+?)\s*$",
    re.MULTILINE,
)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class Recommendation:
    """The fields required in the final response template."""

    name: str
    url: str
    email: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return self.name.casefold(), self.email.casefold()

    def render(self) -> str:
        return f"- [{self.name}]({self.url}) - <{self.email}>\n  Reason: {self.reason}"


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)


def extract_recommendations(tool_response: Any) -> list[Recommendation]:
    """Extract complete matched recommendations from a hook tool response."""
    found: list[Recommendation] = []
    seen: set[tuple[str, str]] = set()

    for text in _flatten_strings(tool_response):
        if not _MATCHED_STATUS_RE.search(text):
            continue
        for match in _ENTRY_RE.finditer(text):
            body = match.group("body")
            contact_match = _CONTACT_RE.search(body)
            reason_match = _REASONING_RE.search(body)
            if not contact_match or not reason_match:
                continue
            email_match = _EMAIL_RE.search(contact_match.group("contact"))
            if not email_match:
                continue
            recommendation = Recommendation(
                name=match.group("name").strip(),
                url=match.group("url").strip(),
                email=email_match.group(0),
                reason=reason_match.group("reason").strip(),
            )
            if recommendation.key not in seen:
                seen.add(recommendation.key)
                found.append(recommendation)
    return found


def _state_root(override: Path | None = None) -> Path:
    if override is not None:
        return override
    configured = os.environ.get("PRIMO_LIBRARIAN_HOOK_STATE_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "primo-librarian-response-hook"


def _state_path(payload: dict[str, Any], state_dir: Path | None = None) -> Path:
    identity = str(
        payload.get("session_id")
        or payload.get("transcript_path")
        or payload.get("cwd")
        or "unknown-session"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return _state_root(state_dir) / f"{digest}.json"


def _load_state(path: Path) -> list[Recommendation]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Recommendation(**item) for item in data.get("recommendations", [])]
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return []


def _save_state(path: Path, recommendations: list[Recommendation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {"recommendations": [asdict(item) for item in recommendations]},
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)


def _clear_state(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _merge_recommendations(
    existing: list[Recommendation],
    incoming: list[Recommendation],
) -> list[Recommendation]:
    merged = list(existing)
    positions = {item.key: index for index, item in enumerate(merged)}
    for item in incoming:
        if item.key in positions:
            merged[positions[item.key]] = item
        else:
            positions[item.key] = len(merged)
            merged.append(item)
    return merged


def _normalise_message(message: str) -> str:
    return "\n".join(
        line.rstrip() for line in message.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def _required_blocks(recommendations: list[Recommendation]) -> str:
    return "\n\n".join(item.render() for item in recommendations)


def _post_tool_output(recommendations: list[Recommendation]) -> dict[str, Any]:
    required = _required_blocks(recommendations)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                "The Primo tool returned one or more matched librarian "
                "recommendations. In the final user-facing response, include "
                "every recommendation exactly in this template, preserving "
                "the server-provided reason:\n\n"
                f"{required}"
            ),
        }
    }


def _stop_output(recommendations: list[Recommendation]) -> dict[str, str]:
    required = _required_blocks(recommendations)
    return {
        "decision": "block",
        "reason": (
            "The response cannot finish because at least one Primo librarian "
            "recommendation is missing or does not use the required template. "
            "Revise the response and include every block below exactly, with "
            "the linked name, angle-bracketed email, and server-provided "
            f"Reason:\n\n{required}"
        ),
    }


def handle_event(
    payload: dict[str, Any],
    *,
    state_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Handle one Claude Code or Codex hook event."""
    event = str(payload.get("hook_event_name", ""))
    path = _state_path(payload, state_dir)

    if event == "UserPromptSubmit":
        _clear_state(path)
        return None

    if event == "PostToolUse":
        tool_name = str(payload.get("tool_name", ""))
        if not _PRIMO_TOOL_RE.search(tool_name):
            return None
        incoming = extract_recommendations(payload.get("tool_response"))
        if not incoming:
            return None
        merged = _merge_recommendations(_load_state(path), incoming)
        _save_state(path, merged)
        return _post_tool_output(merged)

    if event == "Stop":
        recommendations = _load_state(path)
        if not recommendations:
            return None
        message = _normalise_message(str(payload.get("last_assistant_message") or ""))
        missing = [item for item in recommendations if item.render() not in message]
        if missing:
            return _stop_output(recommendations)
        _clear_state(path)
        return None

    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise TypeError("Hook input must be a JSON object.")
        output = handle_event(payload)
        if output is not None:
            json.dump(output, sys.stdout, ensure_ascii=True)
            sys.stdout.write("\n")
        return 0
    except Exception as exc:  # Hooks should fail open rather than break the agent.
        print(f"Primo librarian response hook failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
