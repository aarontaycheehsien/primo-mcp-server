"""Tests for the curator profile tools CLI."""

from __future__ import annotations

import json

from primo_mcp_server.profile_tools import _split_cell, convert, lint


def _write_csv(tmp_path, text: str) -> str:
    path = tmp_path / "profiles.csv"
    # utf-8-sig mirrors real spreadsheet exports, which lead with a BOM.
    path.write_text(text, encoding="utf-8-sig")
    return str(path)


def test_split_cell_prefers_semicolons_over_commas():
    assert _split_cell("law; legal research, cases") == [
        "law",
        "legal research, cases",
    ]
    assert _split_cell("law, legal research") == ["law", "legal research"]
    assert _split_cell("  ") == []


def test_convert_builds_json_directory(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "id,name,url,title,email,subject,keywords,aliases,schools,"
        "resource_types,notes\n"
        'acc,Accounting Librarian,https://example.edu/acc,Business Librarian,'
        'acc@example.edu,"accounting; audit fees",corporate governance,'
        "financial reporting,School of Accountancy,databases,Notes here\n"
        "law,Law Librarian,,,,law,,,,,\n",
    )
    json_path = tmp_path / "profiles.json"

    assert convert(csv_path, str(json_path)) == 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    librarians = {entry["id"]: entry for entry in data["librarians"]}
    assert set(librarians) == {"acc", "law"}
    assert librarians["acc"]["subjects"] == ["accounting", "audit fees"]
    assert librarians["acc"]["keywords"] == ["corporate governance"]
    assert librarians["acc"]["notes"] == "Notes here"
    assert librarians["law"]["subjects"] == ["law"]
    assert librarians["law"]["email"] == ""


def test_convert_accepts_plural_and_spaced_headers(tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "id,name,subjects,best for,resource types\n"
        "law,Law Librarian,law,case law lookup,databases\n",
    )
    json_path = tmp_path / "profiles.json"

    assert convert(csv_path, str(json_path)) == 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["librarians"][0]["best_for"] == ["case law lookup"]
    assert data["librarians"][0]["resource_types"] == ["databases"]


def test_convert_skips_blank_rows_and_warns_on_unknown_columns(
    tmp_path, capsys
):
    csv_path = _write_csv(
        tmp_path,
        "id,name,subject,favourite_colour\n"
        "law,Law Librarian,law,blue\n"
        ",,,\n",
    )
    json_path = tmp_path / "profiles.json"

    assert convert(csv_path, str(json_path)) == 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["librarians"]) == 1
    assert "favourite_colour" in capsys.readouterr().err


def test_convert_rejects_duplicate_ids(tmp_path, capsys):
    csv_path = _write_csv(
        tmp_path,
        "id,name,subject\nlaw,Law Librarian,law\nLAW,Other Librarian,history\n",
    )
    json_path = tmp_path / "profiles.json"

    assert convert(csv_path, str(json_path)) == 2
    assert not json_path.exists()
    assert "Duplicate librarian id(s)" in capsys.readouterr().err


def test_convert_missing_file_returns_error(tmp_path, capsys):
    assert convert(str(tmp_path / "missing.csv"), str(tmp_path / "out.json")) == 2
    assert "Cannot read" in capsys.readouterr().err


def _write_directory(tmp_path, librarians: list[dict]) -> str:
    path = tmp_path / "librarians.json"
    path.write_text(json.dumps({"librarians": librarians}), encoding="utf-8")
    return str(path)


def test_lint_clean_directory_passes(tmp_path, capsys):
    path = _write_directory(
        tmp_path,
        [
            {
                "id": "law",
                "name": "Law Librarian",
                "email": "law@example.edu",
                "subjects": ["law", "legal research"],
            }
        ],
    )

    assert lint(path) == 0
    assert "No problems found." in capsys.readouterr().out


def test_lint_reports_profile_problems(tmp_path, capsys):
    path = _write_directory(
        tmp_path,
        [
            {
                # Filler-only term, normalising variants, no contact, and a
                # deny term broad enough to always fire.
                "id": "messy",
                "name": "Messy Librarian",
                "subjects": ["research support", "Altmetric", "law"],
                "keywords": ["altmetrics"],
                "excludes": ["research"],
            },
            {
                # No terms and no notes: unmatchable and unembeddable.
                "id": "empty",
                "name": "Empty Librarian",
                "email": "empty@example.edu",
            },
        ],
    )

    assert lint(path) == 1

    output = capsys.readouterr().out
    assert "messy: filler-only term(s)" in output
    assert "research support" in output
    assert '"Altmetric" / "altmetrics"' in output
    assert "messy: no email or url" in output
    assert "messy: exclude term(s) broad enough" in output
    assert "empty: no terms or notes" in output


def test_lint_reports_ubiquitous_terms(tmp_path, capsys):
    librarians = [
        {
            "id": f"lib{i}",
            "name": f"Librarian {i}",
            "email": f"lib{i}@example.edu",
            "subjects": ["data management", f"specialty {i}"],
        }
        for i in range(4)
    ]
    path = _write_directory(tmp_path, librarians)

    assert lint(path) == 1

    output = capsys.readouterr().out
    assert "listed by most profiles" in output
    assert '"data management" (4/4 profiles)' in output


def test_lint_unreadable_directory_returns_error(tmp_path, capsys):
    assert lint(str(tmp_path / "missing.json")) == 2
    assert "does not exist" in capsys.readouterr().err
