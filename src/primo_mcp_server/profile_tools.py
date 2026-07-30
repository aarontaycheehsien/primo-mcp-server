"""Curator CLI for librarian profile directories.

Two subcommands keep the profile data pipeline reproducible:

- ``convert``: build the JSON directory the server consumes from a CSV
  source, so enrichment lives in a regenerable file instead of hand-edited
  JSON that silently diverges from its source.
- ``lint``: report profile problems that weaken matching -- filler-only
  terms, terms so ubiquitous that IDF neutralises them, profiles the
  semantic path can never rank, missing contact details, and deny-list
  terms broad enough to suppress a librarian on almost any query.

Usage:
    python -m primo_mcp_server.profile_tools convert profiles.csv profiles.json
    python -m primo_mcp_server.profile_tools lint [profiles.json]

``lint`` without a path checks the configured PRIMO_LIBRARIANS_FILE. Exit
codes: 0 clean, 1 findings, 2 unusable input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarians import (
    LibrarianDirectory,
    LibrarianProfile,
    _duplicate_ids,
    _FILLER_TERMS,
    _is_filler_term,
    _librarian_terms,
    _normalise_text,
    _STOPWORDS,
    load_librarian_directory,
)

_SCALAR_FIELDS = {"id", "name", "url", "title", "email", "notes"}

# Canonical list-field name by normalised CSV header. Real exports use
# singular headers ("subject") and spaces ("resource types"); accept both.
_LIST_FIELD_HEADERS = {
    "subject": "subjects",
    "subjects": "subjects",
    "keyword": "keywords",
    "keywords": "keywords",
    "alias": "aliases",
    "aliases": "aliases",
    "best_for": "best_for",
    "bestfor": "best_for",
    "school": "schools",
    "schools": "schools",
    "resource_type": "resource_types",
    "resource_types": "resource_types",
    "exclude": "excludes",
    "excludes": "excludes",
}

# Directory-wide ubiquity report: a term listed by this share of profiles
# carries almost no routing signal (its IDF multiplier is ~1.0), so listing
# it everywhere is wasted curation. Only meaningful once the directory is
# big enough for "most profiles" to mean something.
_UBIQUITY_SHARE = 0.8
_UBIQUITY_MIN_PROFILES = 4


def _canonical_field(header: str) -> str | None:
    key = header.strip().casefold().replace(" ", "_").replace("-", "_")
    if key in _SCALAR_FIELDS:
        return key
    return _LIST_FIELD_HEADERS.get(key)


def _split_cell(value: str) -> list[str]:
    """Split a multi-value CSV cell on ";" when present, else on ",".

    Semicolons take precedence so terms containing commas ("Head, Data
    Services" style values) survive when the source uses semicolons.
    """
    cleaned = value.strip()
    if not cleaned:
        return []
    separator = ";" if ";" in cleaned else ","
    return [part.strip() for part in cleaned.split(separator) if part.strip()]


def convert(csv_path: str, json_path: str) -> int:
    """Build a JSON librarian directory from a CSV source. Returns exit code."""
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    except OSError as e:
        print(f"Cannot read {csv_path}: {e}", file=sys.stderr)
        return 2

    unknown = [
        column
        for column in header
        if column and _canonical_field(column) is None
    ]
    if unknown:
        print(
            "Ignoring unrecognised column(s): " + ", ".join(unknown),
            file=sys.stderr,
        )

    librarians: list[dict] = []
    for row in rows:
        data: dict = {}
        for raw_key, raw_value in row.items():
            if raw_key is None:
                continue
            field = _canonical_field(raw_key)
            if field is None:
                continue
            value = (raw_value or "").strip()
            if field in _SCALAR_FIELDS:
                data[field] = value
            else:
                data[field] = _split_cell(value)
        if not data.get("id") and not data.get("name"):
            continue  # blank or separator row
        librarians.append(data)

    try:
        directory = LibrarianDirectory.model_validate({"librarians": librarians})
    except ValidationError as e:
        first = e.errors()[0]
        print(
            f"Profile validation failed: {first['msg']} at {first['loc']}.",
            file=sys.stderr,
        )
        return 2

    duplicates = _duplicate_ids(directory)
    if duplicates:
        print(
            f"Duplicate librarian id(s): {', '.join(duplicates)}. "
            "Each id must be unique.",
            file=sys.stderr,
        )
        return 2

    Path(json_path).write_text(
        json.dumps(directory.model_dump(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(directory.librarians)} profile(s) to {json_path}")
    return 0


def _is_low_signal_exclude(term: str) -> bool:
    """True when an exclude would fire on almost any query.

    ``excludes`` are matched against the raw user query, so a filler or
    stopword-only deny term ("research", "help with") suppresses the
    librarian on nearly every search instead of patching one misrouting.
    """
    tokens = _normalise_text(term).split()
    return bool(tokens) and all(
        token in _FILLER_TERMS or token in _STOPWORDS for token in tokens
    )


def _profile_findings(librarian: LibrarianProfile) -> list[str]:
    findings: list[str] = []
    terms = list(_librarian_terms(librarian))

    fillers = sorted({term for term in terms if _is_filler_term(term)})
    if fillers:
        findings.append(
            f"{librarian.id}: filler-only term(s) that never match: "
            + ", ".join(fillers)
        )

    seen: dict[str, str] = {}
    duplicated: set[str] = set()
    for term in terms:
        norm = _normalise_text(term)
        if not norm:
            continue
        if norm in seen and seen[norm] != term:
            duplicated.add(f'"{seen[norm]}" / "{term}"')
        else:
            seen.setdefault(norm, term)
    if duplicated:
        findings.append(
            f"{librarian.id}: term variants that normalise identically "
            "(only one can ever score): " + "; ".join(sorted(duplicated))
        )

    if not any(term.strip() for term in terms) and not librarian.notes.strip():
        findings.append(
            f"{librarian.id}: no terms or notes; the profile can never be "
            "matched or embedded"
        )

    if not librarian.email.strip() and not librarian.url.strip():
        findings.append(
            f"{librarian.id}: no email or url; recommendations cannot link "
            "to a contact"
        )

    low_signal = sorted(
        {term for term in librarian.excludes if _is_low_signal_exclude(term)}
    )
    if low_signal:
        findings.append(
            f"{librarian.id}: exclude term(s) broad enough to suppress this "
            "librarian on almost any query: " + ", ".join(low_signal)
        )

    return findings


def _directory_findings(directory: LibrarianDirectory) -> list[str]:
    findings: list[str] = []
    n = len(directory.librarians)
    if n >= _UBIQUITY_MIN_PROFILES:
        doc_freq: dict[str, tuple[int, str]] = {}
        for librarian in directory.librarians:
            seen: set[str] = set()
            for term in _librarian_terms(librarian):
                norm = _normalise_text(term)
                if norm and norm not in seen:
                    seen.add(norm)
                    count, example = doc_freq.get(norm, (0, term))
                    doc_freq[norm] = (count + 1, example)
        ubiquitous = sorted(
            f'"{example}" ({count}/{n} profiles)'
            for count, example in doc_freq.values()
            if count / n >= _UBIQUITY_SHARE
        )
        if ubiquitous:
            findings.append(
                "Term(s) listed by most profiles carry almost no routing "
                "signal (IDF neutralises them): " + ", ".join(ubiquitous)
            )
    return findings


def lint(json_path: str | None) -> int:
    """Report profile problems that weaken matching. Returns exit code."""
    path = json_path or PrimoConfig().librarians_file
    if not path:
        print(
            "No directory given and PRIMO_LIBRARIANS_FILE is not set.",
            file=sys.stderr,
        )
        return 2

    directory, message = load_librarian_directory(path)
    if message or directory is None:
        print(message, file=sys.stderr)
        return 2

    findings: list[str] = []
    for librarian in directory.librarians:
        findings.extend(_profile_findings(librarian))
    findings.extend(_directory_findings(directory))

    print(f"Checked {len(directory.librarians)} profile(s) in {path}")
    if not findings:
        print("No problems found.")
        return 0
    for finding in findings:
        print(f"- {finding}")
    print(f"{len(findings)} finding(s).")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="primo-profiles",
        description="Convert and lint librarian profile directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert", help="Build the JSON directory from a CSV source."
    )
    convert_parser.add_argument("csv_path", help="Source CSV file.")
    convert_parser.add_argument("json_path", help="JSON file to write.")

    lint_parser = subparsers.add_parser(
        "lint", help="Report profile problems that weaken matching."
    )
    lint_parser.add_argument(
        "json_path",
        nargs="?",
        default=None,
        help="JSON directory to check (default: PRIMO_LIBRARIANS_FILE).",
    )

    args = parser.parse_args()
    if args.command == "convert":
        sys.exit(convert(args.csv_path, args.json_path))
    sys.exit(lint(args.json_path))


if __name__ == "__main__":
    main()
