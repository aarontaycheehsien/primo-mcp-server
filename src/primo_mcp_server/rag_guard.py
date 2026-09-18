"""Deterministic citation guardrails for Primo-grounded RAG answers.

Implements the code-only steps of the guarded RAG pipeline (the same
pipeline sketched in demo_rag_citation_guard.py, but backed by live Primo
retrieval and real sessions):

    1. Retrieve source records            (primo_rag_retrieve, pinned here)
    2. Draft answer with [n] citations    (the calling LLM)
    3. Extract structured citation IDs    (extract_citation_ids)
    4. Validate IDs against retrieved set (validate_ids)
       -> invalid ID? the tool returns regeneration feedback
    5. Build references from validated IDs only (build_references)

Honest limits, enforced in the wording of every validation response:
only structured [n] tags are checked, and validation proves an ID *was
retrieved*, not that the source supports the claim it is attached to.
"""

from __future__ import annotations

import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from primo_mcp_server.citations import format_citation
from primo_mcp_server.models import PrimoRecord

# Attempts are advisory: the store never refuses a validation call, but the
# feedback text changes tone once the LLM has burned this many drafts.
MAX_ATTEMPTS = 3

# Matches one citation group of bare numbers: [1] or [1, 3] or [2; 4].
_CITATION_GROUP_RE = re.compile(r"\[\s*\d+\s*(?:[,;]\s*\d+\s*)*\]")
_ID_RE = re.compile(r"\d+")

# Heuristics for prose citations that bypass the structured check, e.g.
# "(Smith et al., 2020)" or "Smith et al. (2020)". Warn-only.
_PROSE_PAREN_RE = re.compile(r"\([A-Z][^()\d]{0,60}(?:19|20)\d{2}[a-z]?\)")
_PROSE_NARRATIVE_RE = re.compile(
    r"[A-Z][A-Za-z'-]+(?: et al\.?| and [A-Z][A-Za-z'-]+)? \((?:19|20)\d{2}[a-z]?\)"
)


@dataclass
class RagSession:
    """One retrieval pinned for later citation validation."""

    session_id: str
    query: str
    records: list[PrimoRecord]
    style: str = "apa7"
    attempts: int = 0
    created_at: float = field(default_factory=time.time)


class RagSessionStore:
    """In-memory session store with FIFO eviction.

    The server runs as a long-lived stdio process, so sessions survive
    between the retrieve and validate tool calls of one conversation but
    are deliberately not persisted to disk.
    """

    def __init__(self, max_sessions: int = 20):
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, RagSession] = OrderedDict()

    def create(
        self, query: str, records: list[PrimoRecord], style: str = "apa7"
    ) -> RagSession:
        session = RagSession(
            session_id=f"rag-{secrets.token_hex(4)}",
            query=query,
            records=list(records),
            style=style,
        )
        self._sessions[session.session_id] = session
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return session

    def get(self, session_id: str) -> RagSession | None:
        return self._sessions.get(session_id.strip())

    def clear(self) -> None:
        self._sessions.clear()


# ---------------------------------------------------------------------------
# STEP 3 -- extract structured citation IDs (deterministic)
# ---------------------------------------------------------------------------

def extract_citation_ids(draft: str) -> list[int]:
    """Pull every number out of [n]-style tags, first-seen order.

    Grouped tags like [1, 3] count as one citation group but yield each
    ID; anything outside square-bracket numeric tags is invisible to this
    check.
    """
    seen: list[int] = []
    for group in _CITATION_GROUP_RE.findall(draft):
        for num in _ID_RE.findall(group):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


# ---------------------------------------------------------------------------
# STEP 4 -- validate IDs against the retrieved set (deterministic)
# ---------------------------------------------------------------------------

def validate_ids(
    cited: list[int], record_count: int
) -> tuple[list[int], list[int]]:
    """Split cited IDs into (valid, invalid) against 1..record_count."""
    valid = [c for c in cited if 1 <= c <= record_count]
    invalid = [c for c in cited if not 1 <= c <= record_count]
    return valid, invalid


# ---------------------------------------------------------------------------
# STEP 5 -- build references from validated IDs only (deterministic)
# ---------------------------------------------------------------------------

def build_references(
    valid_ids: list[int], records: list[PrimoRecord], style: str = "apa7"
) -> str:
    """Build the reference list from OUR pinned records, never LLM text.

    Each line carries the Primo record ID so the entry can be re-fetched
    with primo_get_record or cited/exported with the other tools.
    """
    lines = []
    for rid in sorted(set(valid_ids)):
        record = records[rid - 1]
        citation = format_citation(record, style)
        lines.append(f"[{rid}] {citation} [Primo record: {record.record_id}]")
    return "\n".join(lines)


def find_prose_citations(draft: str) -> list[str]:
    """Warn-only heuristic for author-year citations outside [n] tags."""
    hits = _PROSE_PAREN_RE.findall(draft) + _PROSE_NARRATIVE_RE.findall(draft)
    deduped: list[str] = []
    for hit in hits:
        if hit not in deduped:
            deduped.append(hit)
    return deduped


# ---------------------------------------------------------------------------
# MCP-facing formatting
# ---------------------------------------------------------------------------

_HONEST_LIMIT_NOTE = (
    "Guardrail note: code verified that every [n] citation points to a record "
    "retrieved from SMU Primo in this session, and the reference list was built "
    "deterministically from those records' metadata. It does NOT verify that "
    "each source supports the claim it is attached to; uncited claims and "
    "prose citations bypass this check, and errors in Primo metadata are "
    "reproduced as-is."
)


def _clip(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_sources_for_drafting(session: RagSession) -> str:
    """Render the pinned records as a numbered evidence pack."""
    blocks: list[str] = []
    for i, r in enumerate(session.records, start=1):
        lines = [f"[{i}] {r.title}"]
        details = []
        if r.display_authors:
            details.append("; ".join(r.display_authors[:6]))
        if r.year:
            details.append(r.year)
        if r.resource_type:
            details.append(r.resource_type)
        if r.peer_reviewed:
            details.append("peer reviewed")
        if details:
            lines.append("    " + " | ".join(details))
        if r.is_part_of:
            lines.append(f"    In: {r.is_part_of}")
        evidence = r.description or r.snippet
        if evidence:
            lines.append(f"    Evidence: {_clip(evidence)}")
        if r.doi:
            lines.append(f"    DOI: {r.doi}")
        lines.append(f"    Primo record: {r.record_id}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_retrieve_response(session: RagSession) -> str:
    """Session header, evidence pack, and drafting rules for the LLM."""
    n = len(session.records)
    return (
        f"RAG session created: {session.session_id}\n"
        f"Query: {session.query}\n"
        f"Retrieved {n} source record{'s' if n != 1 else ''} from SMU Primo "
        f"(labelled [1]-[{n}]).\n\n"
        f"{format_sources_for_drafting(session)}\n\n"
        "DRAFTING RULES (for the calling model):\n"
        f"1. Answer the query using ONLY the sources above ([1]-[{n}]).\n"
        "2. Cite inline with structured numeric tags: [1], [2] or grouped "
        "[1, 3]. Every factual claim needs a tag.\n"
        "3. Do NOT cite anything not listed above, do NOT use author-year "
        "prose citations, do NOT put any other numbers in square brackets, "
        "and do NOT write your own reference list -- code builds it from "
        "validated IDs.\n"
        "4. If the sources cannot answer the query, say so rather than "
        "padding with unsupported claims.\n"
        f"5. Then call primo_rag_validate with session_id='{session.session_id}' "
        "and the full draft. Do not show the draft to the user until "
        "validation passes."
    )


def format_validation_failure(
    session: RagSession, invalid: list[int], cited: list[int]
) -> str:
    """Regeneration feedback for a draft that cited unretrieved IDs."""
    n = len(session.records)
    bad = ", ".join(f"[{c}]" for c in invalid)
    lines = [
        f"VALIDATION FAILED (attempt {session.attempts} of {MAX_ATTEMPTS}) "
        f"for session {session.session_id}.",
        "",
        f"Invalid citation ID{'s' if len(invalid) != 1 else ''}: {bad}. "
        f"Only [1]-[{n}] were retrieved in this session; anything else was "
        "never retrieved and would be a fabricated citation.",
        "",
        "Regenerate the answer citing ONLY the retrieved sources, then call "
        "primo_rag_validate again with the revised draft. Do not present "
        "the answer to the user until validation passes.",
    ]
    if session.attempts >= MAX_ATTEMPTS:
        lines += [
            "",
            "Attempt budget exhausted. Either run primo_rag_retrieve again "
            "with a revised query to get better sources, or tell the user "
            "the retrieved sources could not support a fully cited answer.",
        ]
    return "\n".join(lines)


def format_no_citations_failure(session: RagSession) -> str:
    """Feedback for a draft with no structured tags at all."""
    n = len(session.records)
    return (
        f"VALIDATION FAILED (attempt {session.attempts} of {MAX_ATTEMPTS}) "
        f"for session {session.session_id}.\n\n"
        "The draft contains no [n] citation tags, so nothing can be "
        "verified. Every factual claim must carry a structured numeric tag "
        f"citing one of the retrieved sources [1]-[{n}].\n\n"
        "Regenerate the answer with inline [n] citations and call "
        "primo_rag_validate again. If the retrieved sources genuinely cannot "
        "answer the query, say that to the user instead of an uncited answer."
    )


def format_validation_success(session: RagSession, draft: str, valid: list[int]) -> str:
    """Assemble the final answer: draft + code-built references + limits."""
    n = len(session.records)
    unused = [i for i in range(1, n + 1) if i not in valid]
    references = build_references(valid, session.records, session.style)

    lines = [
        f"VALIDATION PASSED (attempt {session.attempts}) for session "
        f"{session.session_id}.",
        f"Cited sources verified against the retrieved set: "
        + ", ".join(f"[{c}]" for c in sorted(set(valid)))
        + ".",
    ]
    if unused:
        lines.append(
            "Retrieved but uncited: " + ", ".join(f"[{i}]" for i in unused) + "."
        )
    prose = find_prose_citations(draft)
    if prose:
        lines.append(
            "WARNING -- possible prose citations bypass the guardrail and were "
            "NOT verified: " + "; ".join(_clip(p, 80) for p in prose[:5]) + ". "
            "Prefer regenerating with [n] tags only."
        )
    lines += [
        "",
        "FINAL ANSWER (present to the user verbatim, including the "
        "references and the guardrail note):",
        "---",
        draft.strip(),
        "",
        "References",
        references,
        "",
        _HONEST_LIMIT_NOTE,
        "---",
    ]
    return "\n".join(lines)
