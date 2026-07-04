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
    def __init__(
        self,
        records: list[PrimoRecord] | None = None,
        records_by_query: dict[str, list[PrimoRecord]] | None = None,
    ):
        self.records = (
            records
            if records is not None
            else [
                PrimoRecord(
                    record_id="alma123",
                    title="Executive Compensation Data",
                    resource_type="database",
                    subjects=["Accounting", "Executive compensation"],
                    keywords=["Corporate governance"],
                )
            ]
        )
        self.records_by_query = records_by_query or {}
        self.search_calls: list[dict] = []

    async def search(self, **kwargs) -> SearchResponse:
        self.search_calls.append(kwargs)
        query = kwargs.get("query", "")
        records = self.records_by_query.get(query, self.records)
        return SearchResponse.model_validate(
            {
                "info": {"total": len(records)},
                "records": records,
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


async def test_primo_search_zero_results_guides_llm_iteration():
    client = _FakeClient(records=[])

    output = await primo_search(
        _fake_context(client=client),
        "autism",
        resource_type="databases",
        recommend_librarians=False,
    )

    assert [call["query"] for call in client.search_calls] == ["autism"]
    assert 'No results found for "autism".' in output
    assert "Iterative search guidance:" in output
    assert "Try up to five total attempts" in output
    assert "start retries with catalogue databases" in output
    assert 'resource_type="databases"' in output
    assert "direct searches for likely database names" in output
    assert "OR queries for close alternatives" in output
    assert "combine all relevant results found across attempts" in output


def test_primo_search_description_documents_dataset_database_first_policy():
    from primo_mcp_server.policy import PRIMO_SEARCH_DESCRIPTION

    assert "For dataset or data-source requests" in PRIMO_SEARCH_DESCRIPTION
    assert 'scope="catalogue"' in PRIMO_SEARCH_DESCRIPTION
    assert 'resource_type="databases"' in PRIMO_SEARCH_DESCRIPTION
    assert "to articles or books" in PRIMO_SEARCH_DESCRIPTION


async def test_primo_search_tool_serves_the_policy_description():
    from primo_mcp_server.policy import PRIMO_SEARCH_DESCRIPTION, SERVER_INSTRUCTIONS
    from primo_mcp_server.server import mcp

    tools = await mcp.list_tools()
    search_tool = next(t for t in tools if t.name == "primo_search")

    assert search_tool.description == PRIMO_SEARCH_DESCRIPTION
    # The server instructions carry the same single-source policy prose.
    assert "Scope selection policy for callers:" in SERVER_INSTRUCTIONS
    assert "Zero-result policy for callers:" in SERVER_INSTRUCTIONS


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


def _write_metrics_librarians_file(tmp_path) -> str:
    """Directory where a one-term keyword match scores below the
    second-guess threshold (4.0 query weight x idf ~1.69 = ~6.8 < 12)."""
    path = tmp_path / "metrics-librarians.json"
    path.write_text(
        json.dumps(
            {
                "librarians": [
                    {
                        "id": "metrics",
                        "name": "Metrics Librarian",
                        "keywords": ["bibliometrics"],
                    },
                    {
                        "id": "gis",
                        "name": "GIS Librarian",
                        "email": "gis@example.edu",
                        "subjects": ["geospatial analysis"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _fake_semantic(librarian_id: str, calls: list):
    from primo_mcp_server.librarian_embeddings import SemanticFallbackResult
    from primo_mcp_server.librarians import LibrarianMatch

    async def fake(directory, query, records, config, *, limit=2, timeout=None, **kwargs):
        calls.append({"query": query, "timeout": timeout})
        librarian = next(
            lib for lib in directory.librarians if lib.id == librarian_id
        )
        return SemanticFallbackResult(
            [
                LibrarianMatch(
                    librarian=librarian,
                    score=0.82,
                    evidence_fields=["semantic"],
                )
            ]
        )

    return fake


async def test_primo_search_skips_recommendations_for_identifier_query(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "10.1145/1571941.1572114",
        scope="everything",
    )

    assert "Unexpected error" not in output
    assert "## Recommended librarian help:" not in output


async def test_primo_recommend_librarians_skips_identifier_query(tmp_path):
    output = await primo_recommend_librarians(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "ISBN 978-0-13-468599-1",
    )

    assert "Status: skipped" in output
    assert "record identifier" in output


async def test_weak_keyword_match_is_second_guessed_semantically(
    tmp_path, monkeypatch
):
    calls: list = []
    monkeypatch.setattr(
        "primo_mcp_server.recommendation.semantic_fallback",
        _fake_semantic("gis", calls),
    )

    output = await primo_recommend_librarians(
        _fake_context(
            config_overrides={
                "librarians_file": _write_metrics_librarians_file(tmp_path),
                "librarian_semantic_fallback": True,
            }
        ),
        "bibliometrics",
        record_ids=["alma123"],
    )

    # The weak keyword win stays primary; the semantic candidate is appended.
    assert len(calls) == 1
    assert "Status: matched\n" in output
    assert output.index("Metrics Librarian") < output.index("GIS Librarian")
    assert "matched terms: bibliometrics" in output
    assert "Matched by semantic similarity (cosine 0.82)" in output
    # Explicit tool keeps the full embedding timeout budget.
    assert calls[0]["timeout"] is None


async def test_strong_keyword_match_skips_semantic_second_guess(
    tmp_path, monkeypatch
):
    calls: list = []
    monkeypatch.setattr(
        "primo_mcp_server.recommendation.semantic_fallback",
        _fake_semantic("data", calls),
    )

    output = await primo_recommend_librarians(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "librarian_semantic_fallback": True,
            }
        ),
        "executive compensation",
    )

    assert "Accounting Librarian" in output
    assert calls == []  # no embedding cost when keywords are confident


async def test_inline_search_uses_tighter_embedding_timeout(
    tmp_path, monkeypatch
):
    calls: list = []
    monkeypatch.setattr(
        "primo_mcp_server.recommendation.semantic_fallback",
        _fake_semantic("gis", calls),
    )

    output = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_metrics_librarians_file(tmp_path),
                "librarian_semantic_fallback": True,
            }
        ),
        "bibliometrics",
        scope="everything",
    )

    assert "Unexpected error" not in output
    assert len(calls) == 1
    assert calls[0]["timeout"] == 2.5


async def test_primo_list_librarians_lists_configured_profiles(tmp_path):
    from primo_mcp_server.server import primo_list_librarians

    output = await primo_list_librarians(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        )
    )

    assert "## Configured librarians:" in output
    assert "Accounting Librarian" in output
    assert "Data Librarian" in output
    assert "do not invent or substitute names" in output


async def test_primo_list_librarians_without_config_returns_guidance():
    from primo_mcp_server.server import primo_list_librarians

    output = await primo_list_librarians(_fake_context())

    assert output.startswith("Librarian directory unavailable:")
    assert "PRIMO_LIBRARIANS_FILE" in output


async def test_primo_search_forwards_compound_clauses_to_client():
    from primo_mcp_server.query import QueryClause

    client = _FakeClient()
    clauses = [
        QueryClause(field="title", value="capital", connector="AND"),
        QueryClause(field="creator", value="piketty"),
    ]

    output = await primo_search(
        _fake_context(client=client),
        "piketty capital",
        clauses=clauses,
        recommend_librarians=False,
    )

    assert client.search_calls[0]["clauses"] == clauses
    assert "Unexpected error" not in output


async def test_primo_search_no_match_shows_closest_profiles_with_evidence(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                # Unreachable threshold forces no_match while keeping the
                # scored candidates as evidence-bearing near-misses.
                "librarian_min_score": 10_000.0,
            }
        ),
        "executive compensation",
        scope="catalogue",
    )

    assert "Status: no_match" in output
    assert "Closest configured profiles" in output
    assert "Evidence: matched terms:" in output
    assert "(below the confidence threshold)" in output
    assert "closest configured contact" in output
