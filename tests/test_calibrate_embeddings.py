"""Tests for the semantic-fallback calibration CLI."""

from __future__ import annotations

import json

from primo_mcp_server import calibrate_embeddings as calibrate
from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_embeddings import ProfileSimilarity


def _setup(monkeypatch, tmp_path, **overrides) -> None:
    path = tmp_path / "librarians.json"
    path.write_text(
        json.dumps(
            {
                "librarians": [
                    {"id": "law", "name": "Law Librarian", "subjects": ["law"],
                     "excludes": ["tax"]},
                    {"id": "data", "name": "Data Librarian", "subjects": ["data"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    values = {
        "_env_file": None,
        "librarians_file": str(path),
        "embedding_provider": "local",
        "embedding_local_model": "embeddinggemma",
        "embedding_api_key": None,
        "librarian_semantic_min_similarity": 0.1,
    }
    values.update(overrides)
    config = PrimoConfig(**values)
    monkeypatch.setattr(calibrate, "PrimoConfig", lambda: config)

    async def fake_score_profiles(directory, query, config):
        law, data = directory.librarians
        return [
            ProfileSimilarity(0.9, law, "law"),
            ProfileSimilarity(0.2, data, "data"),
        ]

    monkeypatch.setattr(calibrate, "score_profiles", fake_score_profiles)


def test_local_provider_runs_without_a_gemini_key(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path)

    assert calibrate.main(["tax law"]) == 0

    out = capsys.readouterr().out
    assert "Model: embeddinggemma (local," in out


def test_gemini_provider_still_requires_a_key(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path, embedding_provider="gemini")

    assert calibrate.main(["tax law"]) == 1
    assert "PRIMO_EMBEDDING_API_KEY" in capsys.readouterr().err


def test_excluded_profiles_are_marked_not_accepted(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path)

    calibrate.main(["tax law"])

    law_line = next(
        line for line in capsys.readouterr().out.splitlines() if "(law)" in line
    )
    assert "EXCL" in law_line
    assert "ACCEPT" not in law_line
