"""Smoke tests for MCP tool entrypoints."""

from types import SimpleNamespace
import json

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.models import PrimoRecord, SearchResponse
from primo_mcp_server.server import (
    primo_cite,
    primo_export,
    primo_get_record,
    primo_recommend_librarians,
    primo_search,
)


class _FakeClient:
    def __init__(self, records: list[PrimoRecord] | None = None):
        self.records = records or [
            PrimoRecord(
                record_id="alma123",
                title="Executive Compensation Data",
                resource_type="database",
                subjects=["Accounting", "Executive compensation"],
                keywords=["Corporate governance"],
            )
        ]

    async def search(self, **kwargs) -> SearchResponse:
        return SearchResponse.model_validate(
            {
                "info": {"total": len(self.records)},
                "records": self.records,
            }
        )

    async def get_record(self, record_id: str) -> PrimoRecord:
        return PrimoRecord(
            record_id=record_id,
            title="Executive Compensation Data",
            resource_type="database",
        )

    async def get_records(self, record_ids: list[str]) -> list[PrimoRecord]:
        return [
            PrimoRecord(
                record_id=record_id,
                title="Executive Compensation Data",
                resource_type="book",
                creators=["Tan, Mei"],
                creation_date="2024",
                subjects=["Accounting"],
            )
            for record_id in record_ids
        ]


def _write_librarians_file(tmp_path) -> str:
    path = tmp_path / "librarians.json"
    path.write_text(
        json.dumps(
            {
                "librarians": [
                    {
                        "id": "accounting",
                        "name": "Accounting Librarian",
                        "title": "Business Research Librarian",
                        "email": "accounting@example.edu",
                        "url": "https://library.example.edu/accounting",
                        "subjects": ["accounting", "executive compensation"],
                        "keywords": ["corporate governance"],
                        "best_for": ["accounting datasets", "audit research"],
                    },
                    {
                        "id": "data",
                        "name": "Data Librarian",
                        "title": "Data Services Librarian",
                        "email": "data@example.edu",
                        "url": "https://library.example.edu/data",
                        "subjects": ["executive compensation"],
                        "best_for": ["dataset access", "database selection"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _fake_context(
    *,
    client: _FakeClient | None = None,
    config_overrides: dict | None = None,
) -> SimpleNamespace:
    config_values = {
        "base_url": "https://example.test/primaws/rest/pub",
    }
    if config_overrides:
        config_values.update(config_overrides)
    lifespan_context = {
        "client": client or _FakeClient(),
        "config": PrimoConfig(**config_values, _env_file=None),
    }
    return SimpleNamespace(
        request_context=SimpleNamespace(lifespan_context=lifespan_context)
    )


async def test_primo_search_smoke_does_not_return_unexpected_error():
    output = await primo_search(
        _fake_context(),
        "ceo compensation",
        scope="catalogue",
        include_unavailable=True,
    )

    assert "Unexpected error" not in output
    assert "Queries run:" in output
    assert "- Results found: [any,contains,ceo compensation](" in output
    assert "pcAvailability=true" in output
    assert "Executive Compensation Data" in output


async def test_primo_search_appends_inline_librarian_recommendation(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
    )

    assert "## Recommended librarian help:" in output
    assert "[Accounting Librarian](https://library.example.edu/accounting)" in output
    assert "[Data Librarian](https://library.example.edu/data)" in output
    assert "Best for: Consult for accounting datasets and audit research." in output
    assert "matched terms:" in output
    assert "evidence fields:" in output
    assert "Why:" not in output
    assert "Match score:" not in output
    assert "Notes:" not in output


async def test_primo_search_can_disable_inline_librarian_recommendation(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
        recommend_librarians=False,
    )

    assert "Executive Compensation Data" in output
    assert "## Recommended librarian help:" not in output


async def test_primo_search_respects_inline_recommendation_config(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "inline_librarian_recommendations": False,
            }
        ),
        "executive compensation",
        scope="catalogue",
    )

    assert "Executive Compensation Data" in output
    assert "## Recommended librarian help:" not in output


async def test_primo_get_record_smoke_does_not_return_unexpected_error():
    output = await primo_get_record(_fake_context(), "alma123")

    assert "Unexpected error" not in output
    assert "Executive Compensation Data" in output


async def test_primo_cite_accepts_case_insensitive_style():
    output = await primo_cite(_fake_context(), ["alma123"], style="APA7")

    assert "Unexpected error" not in output
    assert "Executive Compensation Data" in output


async def test_primo_recommend_librarians_uses_search_metadata(tmp_path):
    output = await primo_recommend_librarians(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
    )

    assert "Unexpected error" not in output
    assert "## Recommended librarian help:" in output
    assert "[Accounting Librarian](https://library.example.edu/accounting)" in output
    assert "[Data Librarian](https://library.example.edu/data)" in output
    assert "Best for:" in output
    assert "matched terms:" in output
    assert "evidence fields:" in output
    assert "Why:" not in output
    assert "Match score:" not in output
    assert "Notes:" not in output
    assert "do not invent or substitute names" in output


async def test_primo_recommend_librarians_uses_record_ids(tmp_path):
    output = await primo_recommend_librarians(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "accounting",
        record_ids=["alma123"],
    )

    assert "Accounting Librarian" in output


async def test_primo_recommend_librarians_without_config_returns_guidance():
    output = await primo_recommend_librarians(_fake_context(), "accounting")

    assert output.startswith("## Recommended librarian help:")
    assert "Librarian recommendations unavailable" in output
    assert "PRIMO_LIBRARIANS_FILE" in output


async def test_primo_export_accepts_case_insensitive_format():
    output = await primo_export(_fake_context(), ["alma123"], format="BibTeX")

    assert "Unexpected error" not in output
    assert "@book{" in output
