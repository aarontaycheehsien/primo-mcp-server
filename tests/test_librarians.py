"""Tests for librarian recommendation loading and matching."""

from __future__ import annotations

import json

from primo_mcp_server.librarians import (
    LibrarianDirectory,
    LibrarianMatch,
    _stem,
    _term_specificity,
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
    records = [
        PrimoRecord(
            title="Annual reports",
            subjects=["Accounting", "Audit fees"],
        ),
        PrimoRecord(
            title="Audit fee disclosures",
            subjects=["Audit fees"],
        ),
    ]

    matches = recommend_librarians(_directory(), "annual reports", records)

    assert len(matches) == 1
    assert matches[0].librarian.id == "accounting"
    assert "subjects" in matches[0].evidence_fields


def test_single_record_metadata_term_does_not_drive_recommendation():
    record = PrimoRecord(
        title="Annual reports",
        subjects=["Accounting", "Audit fees"],
    )

    assert recommend_librarians(_directory(), "annual reports", [record]) == []


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


def test_generic_source_and_description_terms_do_not_drive_recommendation():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "research",
                    "name": "Research Librarian",
                    "subjects": ["research"],
                },
                {
                    "id": "social",
                    "name": "Social Science Librarian",
                    "subjects": ["Social Science"],
                },
                {
                    "id": "policy",
                    "name": "Policy Librarian",
                    "subjects": ["policy"],
                },
            ]
        }
    )
    records = [
        PrimoRecord(
            title="Medicine",
            description="A record from a social science collection.",
            source_label="ProQuest research library",
        )
    ]

    assert recommend_librarians(directory, "medicine", records) == []


def test_generic_terms_still_match_when_user_query_is_direct():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "research",
                    "name": "Research Librarian",
                    "subjects": ["research"],
                }
            ]
        }
    )

    matches = recommend_librarians(directory, "research support", [])

    assert len(matches) == 1
    assert matches[0].librarian.id == "research"
    assert matches[0].evidence_fields == ["query"]


def test_description_and_source_only_terms_do_not_drive_recommendation():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "altmetrics",
                    "name": "Altmetrics Librarian",
                    "subjects": ["altmetrics"],
                }
            ]
        }
    )
    records = [
        PrimoRecord(
            title="Research impact",
            description="This study analyses altmetrics for policy engagement.",
            source_label="Altmetrics research library",
        )
    ]

    matches = recommend_librarians(directory, "impact", records)

    assert matches == []


def test_high_signal_metadata_terms_can_drive_recommendation():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "altmetrics",
                    "name": "Altmetrics Librarian",
                    "subjects": ["altmetrics", "policy engagement"],
                }
            ]
        }
    )
    records = [
        PrimoRecord(
            title="Research impact",
            subjects=["Altmetrics"],
            keywords=["Policy engagement"],
        ),
        PrimoRecord(
            title="Publication metrics",
            subjects=["Altmetrics"],
        )
    ]

    matches = recommend_librarians(directory, "impact", records)

    assert len(matches) == 1
    assert matches[0].librarian.id == "altmetrics"
    assert matches[0].evidence_fields == ["subjects", "keywords"]


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
    assert "Best for:" not in output
    assert "Notes:" not in output


def test_format_semantic_recommendations_uses_profile_topics_not_best_for():
    librarian = _directory().librarians[0]
    match = LibrarianMatch(
        librarian=librarian,
        score=0.6623,
        evidence_fields=["semantic"],
    )

    output = format_librarian_recommendations(
        [match],
        "transparent evidence map workflow",
        semantic=True,
    )

    assert "Status: matched (semantic fallback)" in output
    assert "Best for:" not in output
    assert (
        "Similar profile topics: accounting datasets, audit research, "
        "accounting, audit fees, financial reporting"
    ) in output
    assert (
        "Evidence: Matched by semantic similarity. "
        "No exact keyword match was found"
    ) in output
    assert "0.66" not in output
    assert "semantic similarity 0.66" not in output


def test_format_semantic_recommendations_uses_not_configured_without_topics():
    match = LibrarianMatch(
        librarian=LibrarianDirectory.model_validate(
            {"librarians": [{"id": "general", "name": "General Librarian"}]}
        ).librarians[0],
        score=0.71,
        evidence_fields=["semantic"],
    )

    output = format_librarian_recommendations(
        [match],
        "transparent evidence map workflow",
        semantic=True,
    )

    assert "Best for:" not in output
    assert "Similar profile topics: Not configured" in output
    assert (
        "Evidence: Matched by semantic similarity. "
        "No exact keyword match was found"
    ) in output


def test_query_subphrase_matches_longer_profile_term():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "ai",
                    "name": "AI Librarian",
                    "subjects": ["AI deep research", "Deep research tools"],
                }
            ]
        }
    )

    matches = recommend_librarians(directory, "deep research", [])

    assert len(matches) == 1
    assert matches[0].librarian.id == "ai"
    assert "query" in matches[0].evidence_fields


def test_single_generic_word_does_not_match_longer_profile_term():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "ai",
                    "name": "AI Librarian",
                    "subjects": ["AI deep research"],
                }
            ]
        }
    )

    assert recommend_librarians(directory, "deep", []) == []
    assert recommend_librarians(directory, "research", []) == []


def test_record_metadata_subphrase_does_not_match_longer_profile_term():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "ai",
                    "name": "AI Librarian",
                    "subjects": ["AI deep research"],
                }
            ]
        }
    )
    records = [
        PrimoRecord(title="A study", subjects=["Deep research"]),
        PrimoRecord(title="Another study", subjects=["Deep research"]),
    ]

    assert recommend_librarians(directory, "irrelevant query", records) == []


def test_stemmed_query_subphrase_still_matches_longer_profile_term():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "ai",
                    "name": "AI Librarian",
                    "subjects": ["AI deep research"],
                }
            ]
        }
    )

    matches = recommend_librarians(directory, "deep researches", [])

    assert len(matches) == 1
    assert matches[0].librarian.id == "ai"


def test_stem_collapses_regular_inflections():
    assert _stem("reviews") == _stem("review")
    assert _stem("bibliometrics") == "bibliometric"
    assert _stem("datasets") == "dataset"
    assert _stem("studies") == "study"
    # Short tokens / acronyms are left intact.
    assert _stem("esg") == "esg"
    assert _stem("ink") == "ink"


def test_plural_query_matches_singular_subject_via_stemming():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "synthesis",
                    "name": "Synthesis Librarian",
                    "subjects": ["Systematic review"],
                }
            ]
        }
    )

    plural = recommend_librarians(directory, "systematic reviews", [])
    singular = recommend_librarians(directory, "systematic review", [])

    assert len(plural) == 1
    assert plural[0].librarian.id == "synthesis"
    # Plural and singular phrasing now score identically.
    assert plural[0].score == singular[0].score


def test_specificity_amplifies_rare_terms_over_shared_ones():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": "a", "name": "A", "subjects": ["research", "altmetrics"]},
                {"id": "b", "name": "B", "subjects": ["research"]},
                {"id": "c", "name": "C", "subjects": ["research"]},
            ]
        }
    )

    specificity = _term_specificity(directory)

    # "research" is shared by every librarian -> near-neutral weight; the
    # term unique to one librarian is amplified above it.
    assert specificity["altmetric"] > specificity["research"]


def test_distinctive_sparse_profile_outranks_generic_padded_profile():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "preservation",
                    "name": "Preservation Librarian",
                    "subjects": ["digital preservation"],
                },
                {
                    "id": "generalist",
                    "name": "Generalist Librarian",
                    "subjects": [
                        "data",
                        "data management",
                        "data services",
                        "research data",
                    ],
                },
                {"id": "filler", "name": "Filler", "subjects": ["data"]},
            ]
        }
    )

    matches = recommend_librarians(directory, "digital preservation data", [])

    assert matches[0].librarian.id == "preservation"


def test_format_recommendations_no_match_uses_heading():
    output = format_librarian_recommendations([], "general research")

    assert output.startswith("## Recommended librarian help:")
    assert "Status: no_match" in output
    assert "Query: general research" in output
    assert (
        'No librarian recommendation met the confidence threshold for "general research".'
        in output
    )
