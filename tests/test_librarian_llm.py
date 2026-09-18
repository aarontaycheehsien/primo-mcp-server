"""Tests for the tier-3 LLM reasoning fallback.

The safety-critical assertions here are the ones about what the tier
REFUSES to surface: an id the directory does not contain, a profile a
curator excluded, and a choice that cannot say why it fits. Those are the
invariants that keep a fabricated librarian off a user's screen.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_llm import (
    build_prompt,
    llm_fallback,
    parse_choices,
)
from primo_mcp_server.librarians import LibrarianDirectory, is_llm_match
from primo_mcp_server.models import PrimoRecord


def _config(**overrides) -> PrimoConfig:
    values = {
        "base_url": "https://example.test/primaws/rest/pub",
        "librarian_llm_fallback": True,
        "llm_url": "http://localhost:11434/v1",
        "llm_model": "test-model",
        "_env_file": None,
    }
    values.update(overrides)
    return PrimoConfig(**values)


def _directory() -> LibrarianDirectory:
    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "psych",
                    "name": "Psychology Librarian",
                    "title": "Behavioural Science Librarian",
                    "email": "psych@example.edu",
                    "subjects": ["behavioural science", "wellbeing"],
                },
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law", "legal research"],
                    "excludes": ["autism"],
                },
            ]
        }
    )


def _completion(choices: list[dict]) -> str:
    return json.dumps({"choices": choices})


def _reasoner(choices: list[dict]):
    async def call(_prompt: str) -> str:
        return _completion(choices)

    return call


async def test_reasoned_choice_becomes_a_match_with_its_reasoning():
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(),
        reasoner=_reasoner(
            [
                {
                    "id": "psych",
                    "confidence": 0.82,
                    "reason": "covers behavioural science",
                }
            ]
        ),
    )

    assert [match.librarian.id for match in result.matches] == ["psych"]
    match = result.matches[0]
    assert is_llm_match(match)
    assert match.score == pytest.approx(0.82)
    assert match.matched_terms == ["covers behavioural science"]
    assert result.error is None


async def test_unknown_id_is_discarded_never_surfaced():
    """The closed-vocabulary invariant: a hallucinated id must not render."""
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(),
        reasoner=_reasoner(
            [
                {"id": "ghost", "confidence": 0.99, "reason": "invented"},
                {"id": "psych", "confidence": 0.7, "reason": "real"},
            ]
        ),
    )

    assert [match.librarian.id for match in result.matches] == ["psych"]


async def test_every_id_unknown_yields_no_matches_not_an_error():
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(),
        reasoner=_reasoner([{"id": "ghost", "confidence": 0.99, "reason": "x"}]),
    )

    assert result.matches == []
    assert result.error is None


async def test_curator_exclusions_still_apply_on_the_llm_path():
    """The law profile excludes "autism"; tier 3 cannot resurrect it."""
    result = await llm_fallback(
        _directory(),
        "autism in the courts",
        None,
        _config(),
        reasoner=_reasoner([{"id": "law", "confidence": 0.95, "reason": "legal"}]),
    )

    assert result.matches == []


async def test_choice_without_a_reason_is_dropped():
    """Evidence is mandatory for every surfaced librarian."""
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(),
        reasoner=_reasoner([{"id": "psych", "confidence": 0.9, "reason": "  "}]),
    )

    assert result.matches == []


async def test_confidence_below_the_floor_is_rejected():
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(librarian_llm_min_confidence=0.8),
        reasoner=_reasoner([{"id": "psych", "confidence": 0.5, "reason": "weak"}]),
    )

    assert result.matches == []


async def test_matches_are_capped_and_ordered_by_confidence():
    result = await llm_fallback(
        _directory(),
        "research help",
        None,
        _config(),
        reasoner=_reasoner(
            [
                {"id": "psych", "confidence": 0.7, "reason": "second"},
                {"id": "law", "confidence": 0.9, "reason": "first"},
            ]
        ),
        limit=1,
    )

    assert [match.librarian.id for match in result.matches] == ["law"]


async def test_duplicate_ids_are_collapsed():
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(),
        reasoner=_reasoner(
            [
                {"id": "psych", "confidence": 0.9, "reason": "first"},
                {"id": "psych", "confidence": 0.8, "reason": "again"},
            ]
        ),
    )

    assert [match.librarian.id for match in result.matches] == ["psych"]


async def test_fenced_json_is_recovered():
    """Small local models wrap JSON in a code fence; salvage beats failing."""

    async def call(_prompt: str) -> str:
        return (
            "Here you go:\n```json\n"
            + _completion([{"id": "psych", "confidence": 0.9, "reason": "ok"}])
            + "\n```"
        )

    result = await llm_fallback(_directory(), "autism", None, _config(), reasoner=call)

    assert [match.librarian.id for match in result.matches] == ["psych"]


async def test_unusable_output_is_reported_as_an_error_not_a_no_match():
    async def call(_prompt: str) -> str:
        return "I am not going to answer in JSON."

    result = await llm_fallback(_directory(), "autism", None, _config(), reasoner=call)

    assert result.matches == []
    assert result.error is not None


async def test_transport_failure_is_caught_and_named():
    async def call(_prompt: str) -> str:
        raise httpx.ConnectError("connection refused")

    result = await llm_fallback(_directory(), "autism", None, _config(), reasoner=call)

    assert result.matches == []
    assert result.error == "ConnectError"


async def test_disabled_tier_never_calls_the_model():
    called = False

    async def call(_prompt: str) -> str:
        nonlocal called
        called = True
        return _completion([])

    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(librarian_llm_fallback=False),
        reasoner=call,
    )

    assert result.matches == []
    assert called is False


async def test_empty_directory_is_skipped_with_a_reason():
    result = await llm_fallback(
        LibrarianDirectory(), "autism", None, _config(), reasoner=_reasoner([])
    )

    assert result.skipped is not None


def test_prompt_carries_ids_and_forbids_inventing_them():
    prompt = build_prompt(_directory(), "autism", None, limit=2)

    assert "id=psych" in prompt
    assert "id=law" in prompt
    assert "Never invent a librarian or an id" in prompt
    assert "Choose at most 2" in prompt
    assert '"autism"' in prompt


def test_prompt_labels_record_subjects_as_secondary():
    records = [PrimoRecord(record_id="a1", title="T", subjects=["Psychology"])]
    prompt = build_prompt(_directory(), "autism", records, limit=2)

    assert "Secondary context" in prompt
    assert "Psychology" in prompt


def test_parse_rejects_a_non_list_choices_field():
    with pytest.raises(ValueError):
        parse_choices(
            json.dumps({"choices": "psych"}), _directory(), "autism", _config(), limit=2
        )


@respx.mock
async def test_default_backend_posts_openai_chat_completions():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _completion(
                                [{"id": "psych", "confidence": 0.9, "reason": "fits"}]
                            )
                        }
                    }
                ]
            },
        )
    )

    result = await llm_fallback(
        _directory(), "autism", None, _config(llm_provider="openai")
    )

    assert [match.librarian.id for match in result.matches] == ["psych"]
    request = json.loads(route.calls[0].request.content)
    assert request["model"] == "test-model"
    assert request["temperature"] == 0


@respx.mock
async def test_llm_key_is_sent_only_to_the_configured_llm_host():
    """The embedding key must never travel here, and vice versa."""
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": _completion([])}}]},
        )
    )

    await llm_fallback(
        _directory(),
        "autism",
        None,
        _config(
            llm_provider="openai",
            llm_api_key="llm-secret",
            embedding_api_key="gemini-secret",
        ),
    )

    headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer llm-secret"
    assert "gemini-secret" not in str(headers)


# ---------------------------------------------------------------------------
# MCP sampling backend: the tier runs on the connected client's own model.
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for the MCP ServerSession's sampling endpoint."""

    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def create_message(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=SimpleNamespace(type="text", text=self.text)
        )


async def test_sampling_reasoner_asks_the_client_model():
    from primo_mcp_server.librarian_llm import sampling_reasoner

    session = _FakeSession(
        _completion([{"id": "psych", "confidence": 0.9, "reason": "behavioural"}])
    )
    config = _config()

    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        config,
        reasoner=sampling_reasoner(session, config=config),
    )

    assert [match.librarian.id for match in result.matches] == ["psych"]
    call = session.calls[0]
    assert call["temperature"] == 0
    # Conversation context would add noise and cost; the prompt is complete.
    assert call["include_context"] == "none"
    assert "id=psych" in call["messages"][0].content.text


async def test_sampling_refusal_degrades_to_a_tier_error():
    """A client that does not implement sampling must not break the tool."""
    from primo_mcp_server.librarian_llm import sampling_reasoner

    session = _FakeSession(error=RuntimeError("Method not found"))
    config = _config()

    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        config,
        reasoner=sampling_reasoner(session, config=config),
    )

    assert result.matches == []
    assert result.error == "RuntimeError"


async def test_non_text_sampling_content_is_an_error():
    from types import SimpleNamespace

    from primo_mcp_server.librarian_llm import sampling_reasoner

    class _ImageSession(_FakeSession):
        async def create_message(self, **kwargs):
            return SimpleNamespace(content=SimpleNamespace(type="image"))

    config = _config()
    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        config,
        reasoner=sampling_reasoner(_ImageSession(), config=config),
    )

    assert result.matches == []
    assert result.error == "ValueError"


async def test_sampling_provider_without_a_session_is_skipped_not_guessed():
    """The offline eval harness has no client session; say so explicitly."""
    result = await llm_fallback(
        _directory(), "autism", None, _config(llm_provider="sampling")
    )

    assert result.matches == []
    assert result.skipped is not None
    assert "sampling" in result.skipped


async def test_unsupported_sampling_names_the_fix_not_just_the_exception():
    """A client without sampling is a config answer, not a transient fault."""
    from mcp.shared.exceptions import McpError
    from mcp.types import METHOD_NOT_FOUND, ErrorData

    from primo_mcp_server.librarian_llm import sampling_reasoner

    session = _FakeSession(
        error=McpError(ErrorData(code=METHOD_NOT_FOUND, message="Method not found"))
    )
    config = _config()

    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        config,
        reasoner=sampling_reasoner(session, config=config),
    )

    assert result.matches == []
    assert "does not support sampling" in result.error
    assert "PRIMO_LLM_PROVIDER=openai" in result.error


async def test_other_protocol_errors_keep_their_code_and_message():
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    from primo_mcp_server.librarian_llm import sampling_reasoner

    session = _FakeSession(
        error=McpError(ErrorData(code=-32603, message="Internal error"))
    )
    config = _config()

    result = await llm_fallback(
        _directory(),
        "autism",
        None,
        config,
        reasoner=sampling_reasoner(session, config=config),
    )

    assert result.error == "McpError -32603: Internal error"
