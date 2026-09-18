"""LLM reasoning fallback for librarian recommendation (tier 3).

Keyword matching (tier 1) and embedding similarity (tier 2) both compare
*surface forms*: stemmed terms, or cosine over a bag of profile terms.
Both are blind to a query whose subject is obvious to a person but shares
no vocabulary with any profile -- "autism" against a directory whose
psychology profile says "behavioural science, wellbeing, survey data"
scores near zero on both paths, and the min-token gate skips the
embedding path for one-word queries entirely.

This tier asks a model to reason about that gap. It runs only when the
first two tiers returned nothing, so the cost and latency are paid on a
miss, never on a hit.

Three properties keep it honest:

- **Closed vocabulary.** The model may only return ids from the directory
  it was shown; unknown ids are discarded, never rendered. The
  "only configured profile IDs, never invent names" invariant is enforced
  in code here, not trusted to the prompt.
- **Curator deny-lists still apply.** ``excludes`` is re-checked after the
  model answers, so this tier cannot resurrect a profile the other tiers
  deliberately suppressed.
- **Self-reported confidence is treated as such.** The score is the
  model's own number, gated by a coarse floor. It is NOT comparable to a
  cosine and is deliberately not fed into the embedding path's
  self-calibrating mean+margin rule, which would read it as a calibrated
  measurement it is not.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Awaitable, Callable, NamedTuple, Sequence

import httpx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarians import (
    LibrarianDirectory,
    LibrarianMatch,
    LibrarianProfile,
    is_excluded,
)
from primo_mcp_server.models import PrimoRecord

logger = logging.getLogger(__name__)

# A reasoner takes a fully-built prompt and returns the raw completion text.
# Narrow on purpose: prompt construction and answer validation stay here, so
# an injected reasoner (tests, a different backend) cannot widen what this
# tier is allowed to return.
Reasoner = Callable[[str], Awaitable[str]]

_MAX_QUERY_CHARS = 2000
# Bounds the prompt: each profile contributes a line, and a directory large
# enough to exceed this wants retrieval, not a single prompt.
_MAX_PROMPT_PROFILES = 60
_MAX_TERMS_PER_FIELD = 8
_MAX_REASON_CHARS = 300

# The evidence_fields marker identifying a match from this tier, mirroring
# the embedding path's ["semantic"].
LLM_EVIDENCE_FIELD = "llm"


class LlmFallbackResult(NamedTuple):
    """Outcome of the LLM reasoning fallback.

    ``error`` carries a short, key-free description (exception class name)
    when the tier failed; ``skipped`` carries a reason when it deliberately
    did not run. They are mutually exclusive, matching the semantic tier's
    contract so the formatter can treat all three tiers alike.
    """

    matches: list[LibrarianMatch]
    error: str | None = None
    skipped: str | None = None


def _terms(values: Sequence[str]) -> str:
    return ", ".join(values[:_MAX_TERMS_PER_FIELD])


def _profile_line(index: int, profile: LibrarianProfile) -> str:
    """One compact directory line: identity plus what the profile covers."""
    parts = [f"[{index}] id={profile.id} | {profile.name}"]
    if profile.title:
        parts.append(f"title: {profile.title}")
    if profile.subjects:
        parts.append(f"subjects: {_terms(profile.subjects)}")
    if profile.keywords:
        parts.append(f"keywords: {_terms(profile.keywords)}")
    if profile.best_for:
        parts.append(f"best for: {_terms(profile.best_for)}")
    if profile.schools:
        parts.append(f"schools: {_terms(profile.schools)}")
    return " | ".join(parts)


def build_prompt(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None,
    *,
    limit: int,
) -> str:
    """Build the routing prompt.

    Catalogue subjects from the returned records are offered as secondary
    context only, and explicitly labelled as such: search results carry
    incidental topics that are not what the user asked about, and the
    embedding tier omits them for exactly that reason.
    """
    profiles = directory.librarians[:_MAX_PROMPT_PROFILES]
    lines = [
        _profile_line(i, profile) for i, profile in enumerate(profiles, start=1)
    ]
    subjects: list[str] = []
    for record in (records or [])[:5]:
        subjects.extend(record.subjects[:3])
    context = ""
    if subjects:
        context = (
            "\nSecondary context -- subject headings from the catalogue "
            "results for this query. They may contain incidental topics; "
            "weigh them below the query itself:\n"
            f"{_terms(list(dict.fromkeys(subjects)))}\n"
        )

    return (
        "You route a library user's research query to the subject "
        "librarian best placed to help.\n\n"
        "Rules:\n"
        "- Choose ONLY from the profiles listed below, by their exact id. "
        "Never invent a librarian or an id.\n"
        f"- Choose at most {limit}.\n"
        "- Choose NONE if no profile genuinely covers the query. Returning "
        "an empty list is a correct and expected answer for a query outside "
        "the directory's coverage; a wrong referral wastes the user's time "
        "and the librarian's.\n"
        "- Judge by subject expertise, not by superficial word overlap. A "
        "profile is a match when the librarian would genuinely know this "
        "literature, even if the query shares no words with the profile.\n"
        "- confidence is your own estimate from 0 to 1 that this librarian "
        "is the right referral.\n"
        "- reason must name the specific expertise that makes the profile "
        "fit, in one short sentence.\n\n"
        f"Profiles:\n" + "\n".join(lines) + "\n"
        f"{context}"
        f'\nQuery: "{query[:_MAX_QUERY_CHARS]}"\n\n'
        "Respond with JSON only, no prose and no code fence:\n"
        '{"choices": [{"id": "<exact id>", "confidence": <0-1>, '
        '"reason": "<one sentence>"}]}\n'
        'Use {"choices": []} when nothing fits.'
    )


def _extract_json(text: str) -> dict:
    """Parse a completion that should be JSON but may carry decoration.

    Small local models routinely wrap JSON in a code fence or prepend a
    sentence; salvaging that is cheaper than failing the tier, so a fenced
    or embedded object is recovered before giving up.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("completion contained no JSON object")


def parse_choices(
    raw: str,
    directory: LibrarianDirectory,
    query: str,
    config: PrimoConfig,
    *,
    limit: int,
) -> list[LibrarianMatch]:
    """Validate a completion into matches, discarding anything unverifiable.

    This is the enforcement point for the closed-vocabulary invariant: a
    returned id that is not in the directory is dropped rather than
    surfaced under a fabricated name.
    """
    payload = _extract_json(raw)
    choices = payload.get("choices") or []
    if not isinstance(choices, list):
        raise ValueError("'choices' is not a list")

    by_id = {profile.id: profile for profile in directory.librarians}
    matches: list[LibrarianMatch] = []
    seen: set[str] = set()
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        profile = by_id.get(str(choice.get("id", "")).strip())
        if profile is None:
            logger.warning(
                "LLM librarian tier returned unknown id %r; discarded",
                choice.get("id"),
            )
            continue
        if profile.id in seen or is_excluded(profile, query):
            continue
        try:
            confidence = float(choice.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < config.librarian_llm_min_confidence:
            continue
        reason = str(choice.get("reason", "")).strip()[:_MAX_REASON_CHARS]
        if not reason:
            # Evidence is mandatory for every surfaced librarian; a choice
            # that cannot say why it fits is not presentable.
            continue
        seen.add(profile.id)
        matches.append(
            LibrarianMatch(
                librarian=profile,
                score=min(confidence, 1.0),
                matched_terms=[reason],
                evidence_fields=[LLM_EVIDENCE_FIELD],
            )
        )
    matches.sort(key=lambda match: -match.score)
    return matches[:limit]


def sampling_reasoner(session, *, config: PrimoConfig, related_request_id=None) -> Reasoner:
    """Build a reasoner backed by MCP sampling.

    The server asks the connected client to run the completion on the model
    already driving the conversation, so this tier needs no API key, no
    second endpoint to keep alive, and costs the operator nothing beyond
    the client's own usage.

    Sampling is an optional part of the MCP protocol: a client is free not
    to implement it, and one that does may still decline an individual
    request. Either shows up as an exception from ``create_message``, which
    ``llm_fallback`` catches and reports as a tier error -- the same
    fail-closed path as an unreachable HTTP endpoint.
    """
    from mcp.types import SamplingMessage, TextContent

    async def call(prompt: str) -> str:
        result = await session.create_message(
            messages=[
                SamplingMessage(
                    role="user", content=TextContent(type="text", text=prompt)
                )
            ],
            max_tokens=config.llm_max_tokens,
            # The directory and query are supplied in full in the prompt;
            # conversation context would only add noise and cost.
            include_context="none",
            temperature=0,
        )
        content = result.content
        if getattr(content, "type", None) != "text":
            raise ValueError(
                f"sampling returned non-text content ({getattr(content, 'type', '?')})"
            )
        return content.text

    return call


async def _openai_chat(
    prompt: str, *, config: PrimoConfig, timeout: float | None
) -> str:
    """Call any OpenAI-compatible chat-completions endpoint.

    One code path reaches Ollama, LM Studio, vLLM, OpenAI, OpenRouter and
    Gemini's OpenAI-compatible endpoint. ``llm_api_key`` is deliberately
    separate from ``embedding_api_key`` so a configured Gemini key can
    never travel to whatever host is set here.
    """
    headers = {}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"
    url = f"{config.llm_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(
        timeout=timeout if timeout is not None else config.llm_timeout,
        headers=headers,
    ) as client:
        response = await client.post(
            url,
            json={
                "model": config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                # Deterministic: this tier feeds a benchmark harness whose
                # numbers are only meaningful if the same query routes the
                # same way twice.
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def llm_fallback(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None,
    config: PrimoConfig,
    *,
    limit: int = 2,
    reasoner: Reasoner | None = None,
    timeout: float | None = None,
) -> LlmFallbackResult:
    """Ask a model to route the query, within the configured directory.

    Fail-closed like the semantic tier: any failure is logged to stderr
    (safe under the stdio MCP transport) and returned in ``error`` so a
    caller can distinguish "the tier broke" from "the tier found nothing".
    """
    if not config.librarian_llm_fallback:
        return LlmFallbackResult([])
    if not directory.librarians:
        return LlmFallbackResult([], skipped="the librarian directory is empty")

    if reasoner is None and config.llm_provider.strip().lower() == "sampling":
        # Sampling needs the live client session, which only the server can
        # supply. Reaching here without one means a non-server caller (the
        # offline eval harness) is configured for sampling; say so rather
        # than silently falling back to an endpoint that may not exist.
        return LlmFallbackResult(
            [],
            skipped=(
                'llm_provider is "sampling", which requires the MCP client '
                "session; this caller has none"
            ),
        )

    prompt = build_prompt(directory, query, records, limit=limit)
    call = reasoner or (
        lambda text: _openai_chat(text, config=config, timeout=timeout)
    )
    try:
        raw = await call(prompt)
    except Exception as e:
        logger.warning(
            "LLM librarian fallback failed (%s): %s", type(e).__name__, e
        )
        return LlmFallbackResult([], error=type(e).__name__)

    try:
        matches = parse_choices(raw, directory, query, config, limit=limit)
    except Exception as e:
        logger.warning(
            "LLM librarian fallback returned unusable output (%s): %s",
            type(e).__name__,
            e,
        )
        return LlmFallbackResult([], error=type(e).__name__)

    return LlmFallbackResult(matches)
