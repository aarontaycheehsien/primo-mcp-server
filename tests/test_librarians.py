"""Tests for librarian recommendation loading and matching."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from primo_mcp_server.librarians import (
    LibrarianDirectory,
    LibrarianMatch,
    _stem,
    _term_specificity,
    format_librarian_recommendations,
    load_librarian_directory,
    load_librarian_directory_cached,
    looks_like_identifier,
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


def test_load_librarian_directory_rejects_duplicate_ids(tmp_path):
    # Ids are keyed into the embedding cache and the keyword/semantic dedup
    # set downstream, so a duplicate would silently collide there rather
    # than surfacing as a clear configuration error.
    path = _write_directory(
        tmp_path,
        {
            "librarians": [
                {"id": "biz", "name": "Business Librarian A", "subjects": ["business"]},
                {"id": "BIZ", "name": "Business Librarian B", "subjects": ["finance"]},
            ]
        },
    )

    directory, message = load_librarian_directory(path)

    assert directory is None
    assert message is not None
    assert "Duplicate librarian id" in message
    assert "biz" in message.casefold()


def test_load_librarian_directory_cached_reuses_parsed_directory(tmp_path):
    path = _write_directory(
        tmp_path,
        {
            "librarians": [
                {"id": "biz", "name": "Business Librarian", "subjects": ["business"]}
            ]
        },
    )

    directory1, message1, specificity1 = load_librarian_directory_cached(path)
    directory2, message2, specificity2 = load_librarian_directory_cached(path)

    assert message1 is None and message2 is None
    # Unchanged mtime -> the cached parse and IDF map are reused, not rebuilt.
    assert directory1 is directory2
    assert specificity1 is specificity2


def test_load_librarian_directory_cached_reloads_after_file_change(tmp_path):
    path = _write_directory(
        tmp_path,
        {
            "librarians": [
                {"id": "biz", "name": "Business Librarian", "subjects": ["business"]}
            ]
        },
    )
    directory1, _, _ = load_librarian_directory_cached(path)

    Path(path).write_text(
        json.dumps(
            {
                "librarians": [
                    {"id": "biz", "name": "Renamed Librarian", "subjects": ["business"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    # Force the mtime forward so the cache reliably detects the change
    # regardless of filesystem timestamp resolution.
    future = time.time() + 5
    os.utime(path, (future, future))

    directory2, message2, _ = load_librarian_directory_cached(path)

    assert message2 is None
    assert directory1.librarians[0].name == "Business Librarian"
    assert directory2.librarians[0].name == "Renamed Librarian"


def test_load_librarian_directory_cached_missing_file_returns_guidance(tmp_path):
    directory, message, specificity = load_librarian_directory_cached(
        tmp_path / "missing.json"
    )

    assert directory is None
    assert specificity == {}
    assert message is not None
    assert "does not exist" in message


def test_load_librarian_directory_cached_does_not_cache_failures(tmp_path):
    path = tmp_path / "librarians.json"
    path.write_text("{", encoding="utf-8")

    directory1, message1, _ = load_librarian_directory_cached(path)
    assert directory1 is None
    assert "Invalid JSON" in message1

    # Fixing the file should take effect immediately, not be masked by a
    # cached failure.
    path.write_text(
        json.dumps(
            {"librarians": [{"id": "biz", "name": "Business Librarian"}]}
        ),
        encoding="utf-8",
    )
    directory2, message2, specificity2 = load_librarian_directory_cached(path)
    assert message2 is None
    assert directory2 is not None
    assert directory2.librarians[0].name == "Business Librarian"


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


def test_precomputed_specificity_matches_default_computation():
    directory = _directory()
    record = PrimoRecord(
        title="Law and accounting",
        subjects=["Law", "Accounting", "Audit fees"],
    )

    default = recommend_librarians(directory, "law accounting", [record])
    with_precomputed = recommend_librarians(
        directory,
        "law accounting",
        [record],
        specificity=_term_specificity(directory),
    )

    assert [m.librarian.id for m in with_precomputed] == [
        m.librarian.id for m in default
    ]
    assert [m.score for m in with_precomputed] == [m.score for m in default]


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


def test_filler_terms_never_match_even_from_direct_query():
    # "I need research support" says nothing about which librarian to
    # consult, yet real profiles list "research" and "support" as standalone
    # terms. Filler terms earn no score from any field, query included.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "research",
                    "name": "Research Librarian",
                    "subjects": ["research", "support", "consultation"],
                }
            ]
        }
    )

    assert recommend_librarians(directory, "I need research support", []) == []


def test_filler_only_phrases_never_match():
    # Real profiles list phrases like "Research support" and "Research
    # consultation" -- every token is filler, so the phrase carries no
    # routing signal either.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "team",
                    "name": "Research Support Team",
                    "aliases": ["Research support", "Research consultation"],
                }
            ]
        }
    )

    assert recommend_librarians(directory, "I need research support", []) == []
    assert recommend_librarians(directory, "research consultation", []) == []


def test_multi_word_terms_containing_filler_words_still_match():
    # The filler tier suppresses whole terms only; "legal research" is a
    # specific topic even though "research" alone is filler.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["legal research"],
                }
            ]
        }
    )

    matches = recommend_librarians(directory, "legal research methods", [])

    assert len(matches) == 1
    assert matches[0].librarian.id == "law"


def test_non_filler_generic_terms_still_match_when_query_is_direct():
    # Terms like "policy" are too weak to trust from noisy record metadata
    # but remain meaningful when the user types them directly.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "policy",
                    "name": "Policy Librarian",
                    "subjects": ["policy"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    matches = recommend_librarians(directory, "policy analysis singapore", [])

    assert len(matches) == 1
    assert matches[0].librarian.id == "policy"
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
    )

    assert "Status: matched (semantic fallback)" in output
    assert "Best for:" not in output
    assert (
        "Similar profile topics: accounting datasets, audit research, "
        "accounting, audit fees, financial reporting"
    ) in output
    # The cosine is surfaced so the calling model can reason about confidence.
    assert (
        "Evidence: Matched by semantic similarity (cosine 0.66). "
        "No exact keyword match was found"
    ) in output


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
    )

    assert "Best for:" not in output
    assert "Similar profile topics: Not configured" in output
    assert (
        "Evidence: Matched by semantic similarity (cosine 0.71). "
        "No exact keyword match was found"
    ) in output


def test_format_mixed_keyword_and_semantic_matches():
    keyword_match = LibrarianMatch(
        librarian=_directory().librarians[0],
        score=15.0,
        matched_terms=["accounting"],
        evidence_fields=["query", "subjects"],
    )
    semantic_match = LibrarianMatch(
        librarian=LibrarianDirectory.model_validate(
            {"librarians": [{"id": "general", "name": "General Librarian"}]}
        ).librarians[0],
        score=0.78,
        evidence_fields=["semantic"],
    )

    output = format_librarian_recommendations(
        [keyword_match, semantic_match],
        "accounting standards",
    )

    # Mixed results keep the plain "matched" status; each entry carries its
    # own evidence style.
    assert "Status: matched\n" in output
    assert "Status: matched (semantic fallback)" not in output
    assert "matched terms: accounting" in output
    assert "Matched by semantic similarity (cosine 0.78)" in output


def test_format_recommendations_surfaces_semantic_error_on_no_match():
    output = format_librarian_recommendations(
        [],
        "obscure topic",
        semantic_error="HTTPStatusError",
    )

    assert "Status: no_match" in output
    assert "semantic fallback errored" in output
    assert "HTTPStatusError" in output


def test_format_recommendations_skip_reason():
    output = format_librarian_recommendations(
        [],
        "10.1145/1571941.1572114",
        skip_reason="The query looks like a record identifier.",
    )

    assert "Status: skipped" in output
    assert "record identifier" in output
    assert "no_match" not in output


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
    # Only equality classes matter: both sides of a match reduce alike.
    assert _stem("reviews") == _stem("review")
    assert _stem("bibliometrics") == _stem("bibliometric")
    assert _stem("datasets") == _stem("dataset")
    assert _stem("studies") == _stem("study")
    # Short tokens / acronyms are left intact.
    assert _stem("esg") == "esg"
    assert _stem("ink") == "ink"


def test_stem_aligns_derivational_families():
    """Snowball collapses whole derivational families, in both en-AU and
    en-US spellings -- the class of miss behind real zero-match queries
    ("anonymising" never matched a profile listing "anonymisation")."""
    families = [
        ("anonymising", "anonymisation", "anonymized"),
        ("preserving", "preservation", "preserved"),
        ("digitising", "digitisation", "digitization"),
        ("visualising", "visualisation", "visualization"),
        ("organising", "organisation", "organization"),
        ("cataloguing", "catalogs", "catalogued"),
        ("behavioural", "behavioral", "behaviours"),
    ]
    for family in families:
        stems = {_stem(word) for word in family}
        assert len(stems) == 1, f"{family} -> {stems}"
    # The -our fold must not corrupt words that are not en-GB variants.
    assert _stem("detour") != _stem("detor")
    assert _stem("contour").startswith("contour")


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
    # term unique to one librarian is amplified above it. Keys are the
    # normalised (stemmed) term forms.
    from primo_mcp_server.librarians import _normalise_text

    assert (
        specificity[_normalise_text("altmetrics")]
        > specificity[_normalise_text("research")]
    )


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


def test_looks_like_identifier_detects_identifiers():
    identifiers = [
        "10.1145/1571941.1572114",
        "https://doi.org/10.18653/v1/D19-1006",
        "doi: 10.1038/nature12373",
        "ISBN 978-0-13-468599-1",
        "isbn:9780134685991",
        "9780134685991",
        "0-13-468599-7",
        "ISSN 2049-3630",
        "1476-4687",
        "alma991234567890123456",
        "cdi_proquest_journals_2461976001",
    ]
    for query in identifiers:
        assert looks_like_identifier(query), query


def test_looks_like_identifier_leaves_topics_alone():
    topics = [
        "digital preservation",
        "executive compensation data",
        "history of singapore 1965",
        "top 10 accounting journals",
        "covid-19 policy response",
        "",
    ]
    for query in topics:
        assert not looks_like_identifier(query), query


def test_duplicate_terms_within_a_profile_score_once():
    # Real profiles repeat terms ("TDM" three times in one keyword list);
    # each concept may earn score only once.
    other = {"id": "history", "name": "History Librarian", "subjects": ["history"]}
    duplicated = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "tdm",
                    "name": "TDM Librarian",
                    "subjects": ["TDM", "TDM", "TDM"],
                },
                other,
            ]
        }
    )
    single = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "tdm",
                    "name": "TDM Librarian",
                    "subjects": ["TDM"],
                },
                other,
            ]
        }
    )

    dup_matches = recommend_librarians(duplicated, "TDM licence for corpus work", [])
    single_matches = recommend_librarians(single, "TDM licence for corpus work", [])

    assert len(dup_matches) == len(single_matches) == 1
    assert dup_matches[0].score == single_matches[0].score


def test_case_and_plural_variants_across_groups_score_once():
    # "Altmetric" (subject) and "altmetrics" (keyword) normalise to the same
    # stem; without dedup the one concept would earn credit under both
    # groups' weights.
    other = {"id": "history", "name": "History Librarian", "subjects": ["history"]}
    variants = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "metrics",
                    "name": "Metrics Librarian",
                    "subjects": ["Altmetric"],
                    "keywords": ["altmetrics"],
                },
                other,
            ]
        }
    )
    subject_only = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "metrics",
                    "name": "Metrics Librarian",
                    "subjects": ["Altmetric"],
                },
                other,
            ]
        }
    )

    variant_matches = recommend_librarians(variants, "altmetric impact", [])
    subject_matches = recommend_librarians(subject_only, "altmetric impact", [])

    assert len(variant_matches) == len(subject_matches) == 1
    # The first group in the fixed order (subjects) wins for the repeat.
    assert variant_matches[0].score == subject_matches[0].score


def test_term_specificity_is_capped():
    # Directory-local uniqueness is not real-world specificity: without a
    # cap, one idiosyncratic term on one profile in a 30-profile directory
    # would be amplified ~4.4x.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {"id": f"lib{i}", "name": f"Librarian {i}", "subjects": [f"topic{i}"]}
                for i in range(30)
            ]
        }
    )

    specificity = _term_specificity(directory)

    assert all(value <= 3.0 for value in specificity.values())
    assert specificity["topic0"] == 3.0


def test_excluded_term_in_query_suppresses_librarian():
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "accounting",
                    "name": "Accounting Librarian",
                    "subjects": ["accounting"],
                    "excludes": ["tax"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    assert recommend_librarians(directory, "tax accounting treatment", []) == []

    matches = recommend_librarians(directory, "accounting standards", [])
    assert len(matches) == 1
    assert matches[0].librarian.id == "accounting"


def test_load_warns_about_filler_terms_in_profiles(tmp_path, caplog):
    path = _write_directory(
        tmp_path,
        {
            "librarians": [
                {
                    "id": "biz",
                    "name": "Business Librarian",
                    "subjects": ["business", "research", "support"],
                }
            ]
        },
    )

    directory, message = load_librarian_directory(path)

    assert message is None
    assert directory is not None
    assert "filler term" in caplog.text
    assert "research" in caplog.text
    assert "support" in caplog.text


def test_incidental_word_in_long_query_does_not_drive_recommendation():
    # One generic-ish word buried in a long natural-language question is
    # weak evidence; previously it earned the same query weight as an exact
    # one-word query.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "policy",
                    "name": "Policy Librarian",
                    "subjects": ["policy"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    long_query = (
        "impact of monetary policy on regional banking employment trends"
    )
    assert recommend_librarians(directory, long_query, []) == []

    # The same term as (most of) the query remains a confident match.
    matches = recommend_librarians(directory, "policy", [])
    assert [m.librarian.id for m in matches] == ["policy"]


def test_multi_word_terms_keep_strength_in_medium_queries():
    # sqrt dampening: a specific two-word alias in a four-content-word query
    # keeps most of its weight and still routes.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "sysrev",
                    "name": "Systematic Review Librarian",
                    "aliases": ["systematic review"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    matches = recommend_librarians(
        directory, "systematic review screening tools medicine", []
    )

    assert [m.librarian.id for m in matches] == ["sysrev"]


def test_filler_only_subphrase_does_not_claim_longer_profile_term():
    # "information services" inside "legal information services" strips the
    # qualifier that made the term specific; a sub-phrase made entirely of
    # filler/function words never matches by reverse containment.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "law",
                    "name": "Law Librarian",
                    "subjects": ["legal information services"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    assert recommend_librarians(directory, "information services", []) == []


def test_short_subphrase_cannot_claim_long_specialised_term():
    # A two-word query sub-phrase covering under half of a five-word
    # specialised term is qualifier-stripping, not a match.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "qual",
                    "name": "Qualitative Methods Librarian",
                    "subjects": ["qualitative data analysis software workshops"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    assert recommend_librarians(directory, "data analysis", []) == []


def test_subphrase_covering_half_of_term_still_matches():
    # The existing "deep research" -> "AI deep research" behaviour is
    # preserved: the sub-phrase has a content word and covers 2 of 3 tokens.
    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "ai",
                    "name": "AI Librarian",
                    "subjects": ["AI deep research"],
                },
                {
                    "id": "history",
                    "name": "History Librarian",
                    "subjects": ["history"],
                },
            ]
        }
    )

    matches = recommend_librarians(directory, "deep research", [])

    assert [m.librarian.id for m in matches] == ["ai"]


def test_format_recommendations_surfaces_semantic_skip_on_no_match():
    output = format_librarian_recommendations(
        [],
        "preservation",
        semantic_skipped=(
            "the query has too few topical words for reliable semantic "
            "matching (needs at least 2)"
        ),
    )

    assert "Status: no_match" in output
    assert "semantic fallback was skipped" in output
    assert "too few topical words" in output


def test_format_librarian_directory_lists_every_profile():
    from primo_mcp_server.librarians import format_librarian_directory

    output = format_librarian_directory(_directory())

    assert "## Configured librarians:" in output
    assert "2 librarian profile(s) are configured." in output
    assert "[Accounting Librarian](https://library.example.edu/accounting)" in output
    assert "Best for: Consult for accounting datasets and audit research." in output
    assert "Schools: School of Accountancy" in output
    # A profile without url or email still renders as a link (to "#").
    assert "[Law Librarian](#)" in output
    assert "do not invent or substitute names" in output
    # Matching-only fields stay out of the listing.
    assert "financial reporting" not in output
    assert "legal research" not in output


def test_format_librarian_directory_caps_listed_subjects():
    from primo_mcp_server.librarians import format_librarian_directory

    directory = LibrarianDirectory.model_validate(
        {
            "librarians": [
                {
                    "id": "broad",
                    "name": "Broad Librarian",
                    "subjects": [f"subject {i}" for i in range(15)],
                }
            ]
        }
    )

    output = format_librarian_directory(directory)

    assert "subject 11" in output
    assert "subject 12" not in output
    assert "(+3 more)" in output


class TestNearMisses:
    def test_rank_librarians_keeps_below_threshold_candidates(self):
        from primo_mcp_server.librarians import rank_librarians

        directory = _directory()
        candidates = rank_librarians(directory, "audit fees dataset")

        assert candidates, "expected at least one scored candidate"
        assert candidates[0].librarian.id == "accounting"
        assert candidates[0].matched_terms
        # The same query filtered at an unreachable threshold returns
        # nothing -- the candidates themselves survive in rank_librarians.
        assert (
            recommend_librarians(directory, "audit fees dataset", min_score=10_000.0)
            == []
        )

    def test_no_match_output_shows_near_misses_with_evidence(self):
        directory = _directory()
        from primo_mcp_server.librarians import rank_librarians

        near_misses = rank_librarians(directory, "audit fees dataset")[:2]
        output = format_librarian_recommendations(
            [],
            "audit fees dataset",
            near_misses=near_misses,
        )

        assert "Status: no_match" in output
        assert "Closest configured profiles" in output
        assert "NOT validated recommendations" in output
        assert "Evidence: matched terms:" in output
        assert "(below the confidence threshold)" in output
        assert "always include the evidence shown above" in output
        assert "[Accounting Librarian](https://library.example.edu/accounting)" in output

    def test_no_match_output_without_near_misses_points_to_directory(self):
        output = format_librarian_recommendations([], "xyzzy")

        assert "Status: no_match" in output
        assert "Closest configured profiles" not in output
        assert "primo_list_librarians" in output
        assert "never present a librarian as recommended without showing evidence" in output
