"""Tests for librarian recommendation loading and matching."""

from __future__ import annotations

import json

from primo_mcp_server.librarians import (
    LibrarianDirectory,
    format_librarian_recommendations,
    load_librarian_directory,
    recommend_librarians,
)
from primo_mcp_server.models import PrimoRecord


def _write_directory(tmp_path, data: dict) -> str:
    path = tmp_path / "librarians.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _directory() -> LibrarianDirectory:
    return LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "title": "Business Research Librarian",
                    "email": "accounting@example.edu",
                    "url": "https://library.example.edu/accounting",
                    "subjects": ["accounting", "audit fees"],
                    "keywords": ["corporate governance"],
                    "aliases": ["financial reporting"],
                    "best_for": ["accounting datasets", "audit research"],
                    "schools": ["School of Accountancy"],
                    "resource_types": ["databases"],
                    "notes": "Consult for accounting and audit research.",
                },
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["law"],
                    "aliases": ["legal research"],
                },
            ]
        }
    )


def test_load_librarian_directory_from_json(tmp_path):
    path = _write_directory(
        tmp_path,
        {
            "librarians": [
                {
                    "id": "biz",
                    "name": "Business Librarian",
                    "subjects": ["business"],
                }
            ]
        },
    )

    directory, message = load_librarian_directory(path)

    assert message is None
    assert directory is not None
    assert directory.librarians[0].name == "Business Librarian"


def test_missing_librarian_file_returns_guidance(tmp_path):
    directory, message = load_librarian_directory(tmp_path / "missing.json")

    assert directory is None
    assert message is not None
    assert "does not exist" in message
    assert "PRIMO_LIBRARIANS_FILE" in message


def test_invalid_json_returns_guidance(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    directory, message = load_librarian_directory(path)

    assert directory is None
    assert message is not None
    assert "Invalid JSON" in message


def test_invalid_profile_returns_validation_guidance(tmp_path):
    path = _write_directory(tmp_path, {"librarians": [{"id": "missing-name"}]})

    directory, message = load_librarian_directory(path)

    assert directory is None
    assert message is not None
    assert "Profile validation failed" in message


def test_subject_match_from_query_and_record_metadata_scores_highest():
    record = PrimoRecord(
        title="Audit fees and corporate governance in Singapore",
        resource_type="article",
        subjects=["Audit fees", "Corporate governance"],
    )

    matches = recommend_librarians(
        _directory(),
        "audit fees in Singapore",
        [record],
        limit=2,
    )

    assert matches[0].librarian.id == "accounting"
    assert "audit fees" in matches[0].matched_terms
    assert "query" in matches[0].evidence_fields
    assert "subjects" in matches[0].evidence_fields


def test_alias_match_from_query_is_recommended():
    matches = recommend_librarians(
        _directory(),
        "financial reporting standards",
        [],
    )

    assert len(matches) == 1
    assert matches[0].librarian.id == "accounting"
    assert matches[0].matched_terms == ["financial reporting"]


def test_best_for_match_from_query_is_recommended():
    matches = recommend_librarians(
        _directory(),
        "accounting datasets",
        [],
    )

    assert len(matches) == 1
    assert matches[0].librarian.id == "accounting"
    assert "accounting datasets" in matches[0].matched_terms


def test_record_subject_metadata_can_drive_recommendation():
    record = PrimoRecord(
        title="Annual reports",
        subjects=["Accounting"],
    )

    matches = recommend_librarians(_directory(), "annual reports", [record])

    assert len(matches) == 1
    assert matches[0].librarian.id == "accounting"
    assert matches[0].evidence_fields == ["subjects"]


def test_default_returns_top_two_and_orders_by_score():
    record = PrimoRecord(
        title="Law and accounting",
        subjects=["Law", "Accounting", "Audit fees"],
    )

    matches = recommend_librarians(_directory(), "law accounting", [record])

    assert len(matches) == 2
    assert matches[0].librarian.id == "accounting"
    assert matches[1].librarian.id == "law"


def test_low_confidence_match_returns_no_recommendation():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "databases",
                    "name": "Database Librarian",
                    "resource_types": ["databases"],
                }
            ]
        }
    )
    record = PrimoRecord(resource_type="database")

    assert recommend_librarians(directory, "general research", [record]) == []


def test_format_recommendations_includes_validation_instruction():
    matches = recommend_librarians(
        _directory(),
        "financial reporting",
        [],
    )

    output = format_librarian_recommendations(matches, "financial reporting")

    assert output.startswith("## Recommended librarian help:")
    assert "Status: matched" in output
    assert (
        "1. Name: [Accounting Librarian](https://library.example.edu/accounting)"
        in output
    )
    assert "Title: Business Research Librarian" in output
    assert "Contact: accounting@example.edu" in output
    assert "Contact: accounting@example.edu | https://library.example.edu/accounting" not in output
    assert "Best for: Consult for accounting datasets and audit research." in output
    assert "Evidence: matched terms: financial reporting; evidence fields: query" in output
    assert "Why:" not in output
    assert "Match score:" not in output
    assert "Notes:" not in output
    assert "do not invent or substitute names" in output


def test_format_recommendations_links_name_with_email_fallback():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "data",
                    "name": "Data Librarian",
                    "email": "data@example.edu",
                    "subjects": ["data"],
                }
            ]
        }
    )
    matches = recommend_librarians(directory, "data", [])

    output = format_librarian_recommendations(matches, "data")

    assert "1. Name: [Data Librarian](mailto:data@example.edu)" in output
    assert "Title: Not configured" in output
    assert "Best for: No configured areas of support." in output
    assert "Notes:" not in output


def test_format_recommendations_no_match_uses_heading():
    output = format_librarian_recommendations([], "general research")

    assert output.startswith("## Recommended librarian help:")
    assert "Status: no_match" in output
    assert "Query: general research" in output
    assert (
        'No librarian recommendation met the confidence threshold for "general research".'
        in output
    )
