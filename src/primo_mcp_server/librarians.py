"""Librarian recommendation models and deterministic matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from primo_mcp_server.models import PrimoRecord


_MAX_RECOMMENDATIONS = 3
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SECTION_HEADING = "## Recommended librarian help:"
_RECOMMENDATION_FOOTER = (
    "Recommendations are limited to configured librarian profiles; "
    "do not invent or substitute names."
)
_UNCONFIGURED = "Not configured"


class LibrarianProfile(BaseModel):
    """A configured librarian or librarian team profile."""

    id: str
    name: str
    title: str = ""
    email: str = ""
    url: str = ""
    subjects: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    best_for: list[str] = Field(default_factory=list)
    schools: list[str] = Field(default_factory=list)
    resource_types: list[str] = Field(default_factory=list)
    notes: str = ""


class LibrarianDirectory(BaseModel):
    """External librarian directory loaded from JSON."""

    librarians: list[LibrarianProfile] = Field(default_factory=list)


class LibrarianMatch(BaseModel):
    """A validated librarian recommendation."""

    librarian: LibrarianProfile
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    evidence_fields: list[str] = Field(default_factory=list)


def _configuration_message(path: str | None = None, detail: str | None = None) -> str:
    location = (
        f'PRIMO_LIBRARIANS_FILE is set to "{path}". '
        if path
        else "Set PRIMO_LIBRARIANS_FILE to the path of a JSON file. "
    )
    suffix = f" {detail}" if detail else ""
    return (
        "Librarian recommendations are not configured. "
        + location
        + 'Expected shape: {"librarians": [{"id": "...", "name": "..."}]}.'
        + suffix
    )


def load_librarian_directory(
    path: str | Path | None,
) -> tuple[LibrarianDirectory | None, str | None]:
    """Load an external JSON librarian directory.

    Returns (directory, message). The message is populated when the directory
    cannot be loaded and is intended for MCP-facing guidance.
    """
    if path is None or str(path).strip() == "":
        return None, _configuration_message()

    resolved = Path(path).expanduser()
    try:
        with resolved.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, _configuration_message(
            str(resolved), "The file does not exist."
        )
    except PermissionError:
        return None, _configuration_message(
            str(resolved), "The file is not readable."
        )
    except json.JSONDecodeError as e:
        return None, _configuration_message(
            str(resolved), f"Invalid JSON at line {e.lineno}, column {e.colno}."
        )
    except OSError as e:
        return None, _configuration_message(str(resolved), str(e))

    try:
        directory = LibrarianDirectory.model_validate(data)
    except ValidationError as e:
        return None, _configuration_message(
            str(resolved), f"Profile validation failed: {e.errors()[0]['msg']}."
        )

    if not directory.librarians:
        return None, _configuration_message(
            str(resolved), "The directory contains no librarians."
        )
    return directory, None


def _normalise_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _contains_term(text: str, term: str) -> bool:
    term_norm = _normalise_text(term)
    if len(term_norm) < 2:
        return False
    text_norm = _normalise_text(text)
    if not text_norm:
        return False
    if " " in term_norm:
        return f" {term_norm} " in f" {text_norm} "
    return term_norm in set(text_norm.split())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _format_list(values: list[str], *, empty: str = _UNCONFIGURED) -> str:
    return ", ".join(values) if values else empty


def _format_human_list(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    if len(values) == 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _format_best_for_sentence(values: list[str]) -> str:
    if not values:
        return "No configured areas of support."
    return f"Consult for {_format_human_list(values)}."


def _profile_link_target(librarian: LibrarianProfile) -> str:
    """Return a link target so displayed names are always Markdown links."""
    if librarian.url.strip():
        return librarian.url.strip()
    if librarian.email.strip():
        return f"mailto:{librarian.email.strip()}"
    return "#"


def _format_linked_name(librarian: LibrarianProfile) -> str:
    return f"[{librarian.name}]({_profile_link_target(librarian)})"


def _format_match_evidence(match: LibrarianMatch) -> str:
    return (
        f"matched terms: {_format_list(match.matched_terms, empty='none')}; "
        f"evidence fields: {_format_list(match.evidence_fields, empty='none')}"
    )


def _record_field_texts(records: list[PrimoRecord]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {
        "title": [],
        "subjects": [],
        "keywords": [],
        "description": [],
        "resource_type": [],
        "source": [],
    }
    for record in records:
        fields["title"].append(record.title)
        fields["subjects"].extend(record.subjects)
        fields["keywords"].extend(record.keywords)
        fields["description"].extend([record.description, record.snippet])
        fields["resource_type"].append(record.resource_type)
        fields["source"].extend(
            [
                record.source_label,
                record.publisher,
                record.journal_title,
                record.is_part_of,
            ]
        )
    return {name: _unique(values) for name, values in fields.items()}


def _score_term(
    term: str,
    texts_by_field: dict[str, list[str]],
    weights: dict[str, float],
) -> tuple[float, list[str]]:
    score = 0.0
    evidence_fields: list[str] = []
    for field, texts in texts_by_field.items():
        weight = weights.get(field, 0.0)
        if weight <= 0:
            continue
        if any(_contains_term(text, term) for text in texts):
            score += weight
            evidence_fields.append(field)
    return score, evidence_fields


def recommend_librarians(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None = None,
    *,
    limit: int = 2,
    min_score: float = 5.0,
) -> list[LibrarianMatch]:
    """Rank configured librarians against a query and Primo record metadata."""
    records = records or []
    texts_by_field = {"query": [query], **_record_field_texts(records)}

    matches: list[LibrarianMatch] = []
    for librarian in directory.librarians:
        score = 0.0
        matched_terms: list[str] = []
        evidence_fields: list[str] = []

        term_groups: list[tuple[list[str], dict[str, float]]] = [
            (
                librarian.subjects,
                {
                    "query": 7.0,
                    "subjects": 8.0,
                    "keywords": 5.0,
                    "title": 3.0,
                    "description": 3.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.aliases,
                {
                    "query": 8.0,
                    "subjects": 7.0,
                    "keywords": 5.0,
                    "title": 4.0,
                    "description": 4.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.keywords,
                {
                    "query": 4.0,
                    "subjects": 4.0,
                    "keywords": 4.0,
                    "title": 2.0,
                    "description": 2.0,
                    "source": 1.0,
                },
            ),
            (
                librarian.best_for,
                {
                    "query": 8.0,
                    "subjects": 6.0,
                    "keywords": 6.0,
                    "title": 4.0,
                    "description": 4.0,
                    "resource_type": 3.0,
                    "source": 3.0,
                },
            ),
            (
                librarian.schools,
                {
                    "query": 3.0,
                    "subjects": 2.0,
                    "keywords": 2.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.resource_types,
                {
                    "query": 2.0,
                    "resource_type": 4.0,
                },
            ),
        ]

        for terms, weights in term_groups:
            for term in terms:
                term_score, term_fields = _score_term(term, texts_by_field, weights)
                if term_score <= 0:
                    continue
                score += term_score
                matched_terms.append(term)
                evidence_fields.extend(term_fields)

        if score >= min_score:
            matches.append(
                LibrarianMatch(
                    librarian=librarian,
                    score=score,
                    matched_terms=_unique(matched_terms),
                    evidence_fields=_unique(evidence_fields),
                )
            )

    matches.sort(key=lambda match: (-match.score, match.librarian.name.casefold()))
    capped_limit = min(max(1, limit), _MAX_RECOMMENDATIONS)
    return matches[:capped_limit]


def format_librarian_recommendations(
    matches: list[LibrarianMatch],
    query: str,
    *,
    configuration_message: str | None = None,
) -> str:
    """Format librarian recommendations for MCP responses."""
    if configuration_message:
        return (
            f"{_SECTION_HEADING}\n\n"
            "Status: unavailable\n"
            f"Message: Librarian recommendations unavailable: {configuration_message}"
        )

    if not matches:
        return (
            f"{_SECTION_HEADING}\n\n"
            "Status: no_match\n"
            f"Query: {query}\n"
            f'Message: No librarian recommendation met the confidence threshold for "{query}".'
        )

    lines = [_SECTION_HEADING, "", "Status: matched"]
    for i, match in enumerate(matches, start=1):
        librarian = match.librarian
        lines.append(f"{i}. Name: {_format_linked_name(librarian)}")
        lines.append(f"   Title: {librarian.title or _UNCONFIGURED}")
        lines.append(f"   Contact: {librarian.email or _UNCONFIGURED}")
        lines.append(f"   Best for: {_format_best_for_sentence(librarian.best_for)}")
        lines.append(f"   Evidence: {_format_match_evidence(match)}")

    lines.append(_RECOMMENDATION_FOOTER)
    return "\n".join(lines)
