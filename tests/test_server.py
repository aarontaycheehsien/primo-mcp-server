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
        # These tests exercise a configured institution. Inline
        # recommendations ship OFF (no librarian data is bundled), so the
        # harness opts in; tests of the off path override it explicitly.
        "inline_librarian_recommendations": True,
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


def _search_text(output) -> str:
    """Extract the backwards-compatible readable text from an MCP result."""
    assert len(output.content) == 1
    return output.content[0].text


async def test_primo_search_smoke_does_not_return_unexpected_error():
    output = await primo_search(
        _fake_context(),
        "ceo compensation",
        scope="catalogue",
        include_unavailable=True,
    )
    output = _search_text(output)

    assert "Unexpected error" not in output
    assert "Queries attempted:" in output
    assert "- Results found: [any,contains,ceo compensation](" in output
    assert "pcAvailability=true" in output
    assert "Executive Compensation Data" in output


async def test_primo_search_appends_inline_librarian_recommendation(tmp_path):
    result = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
    )
    output = _search_text(result)

    assert output.startswith("## Required librarian referral")
    assert (
        "Caller action: You MUST include every recommended librarian's name, "
        "title, contact, and evidence" in output
    )
    assert output.index("## Recommended librarian help:") < output.index(
        "## Primo search results"
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
    assert result.structuredContent["librarian_status"] == "matched"
    assert result.structuredContent["caller_action"] == (
        "include_in_user_response_with_evidence"
    )
    recommended = result.structuredContent["librarian_recommendations"]
    assert recommended[0]["name"] == "Accounting Librarian"
    assert "executive compensation" in recommended[0]["evidence"]["matched_terms"]
    assert "subjects" in recommended[0]["evidence"]["evidence_fields"]


async def test_primo_search_can_disable_inline_librarian_recommendation(tmp_path):
    output = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
        recommend_librarians=False,
    )
    output = _search_text(output)

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
    output = _search_text(output)

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
    output = _search_text(output)

    assert [call["query"] for call in client.search_calls] == ["autism"]
    assert 'No results found for "autism".' in output
    assert "Iterative search guidance:" in output
    assert "Try up to five total attempts" in output
    assert "start retries with catalogue databases" in output
    assert 'resource_type="databases"' in output
    assert "direct searches for likely database names" in output
    assert "OR queries for close alternatives" in output
    assert "combine all relevant results found across attempts" in output


async def test_primo_search_always_demands_queries_attempted_and_counts():
    """The transparency banner and count block ride on every hit search."""
    result = await primo_search(
        _fake_context(),
        "ceo compensation",
        scope="catalogue",
        recommend_librarians=False,
    )
    output = _search_text(result)

    assert output.startswith("## Required search transparency")
    assert '"Queries attempted:" list naming every query run' in output
    assert "number of results each returned" in output
    assert "click it to re-run the search" in output
    assert "Queries attempted:" in output
    assert "- Results found: [any,contains,ceo compensation](" in output
    assert "-- 1 result" in output


async def test_zero_result_search_still_reports_the_attempt_and_count():
    """A miss is the attempt most worth reporting, so it carries the block too."""
    result = await primo_search(
        _fake_context(client=_FakeClient(records=[])),
        "autism",
        resource_type="databases",
        recommend_librarians=False,
    )
    output = _search_text(result)

    assert output.startswith("## Required search transparency")
    assert "Queries attempted:" in output
    assert "- No results: [any,contains,autism](" in output
    assert "-- 0 results" in output


async def test_search_transparency_is_exposed_as_structured_metadata():
    """A caller reading metadata rather than prose sees the same obligation."""
    result = await primo_search(
        _fake_context(),
        "ceo compensation",
        scope="catalogue",
        recommend_librarians=False,
    )

    transparency = result.structuredContent["search_transparency"]
    assert transparency["caller_action"] == (
        "report_queries_attempted_with_result_counts"
    )
    assert transparency["total_results"] == 1
    attempted = transparency["queries_attempted"]
    assert len(attempted) == 1
    assert attempted[0]["query"] == "any,contains,ceo compensation"
    assert attempted[0]["results"] == 1
    assert "any%2Ccontains%2Cceo+compensation" in attempted[0]["url"]


async def test_transparency_survives_the_librarian_referral_prepend(tmp_path):
    """The librarian banner may lead, but the counts must not be dropped."""
    result = await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
    )
    output = _search_text(result)

    assert output.startswith("## Required librarian referral")
    assert "## Required search transparency" in output
    assert "Queries attempted:" in output
    assert "-- 1 result" in output
    assert result.structuredContent["search_transparency"]["total_results"] == 1


async def test_routing_request_leads_the_search_result(tmp_path):
    """Owed work must lead, not trail several screens of hits.

    Appended last, the routing task was the easiest thing in the response
    for a caller to never reach -- and a caller that never reaches it
    silently produces no recommendation at all.
    """
    result = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "librarian_llm_fallback": True,
                "librarian_llm_inline": True,
                "llm_provider": "caller",
                # Force a keyword miss so the routing tier engages.
                "librarian_min_score": 10_000.0,
            }
        ),
        "deep sea fishing quotas",
        scope="catalogue",
    )
    output = _search_text(result)

    assert output.startswith("## Required librarian routing decision")
    assert "primo_submit_librarian_choice" in output
    # Deciding that nobody fits has to read as a real option, or the only
    # way to satisfy the banner looks like naming someone.
    assert "no configured librarian" in output
    assert "## Primo search results" in output
    assert "## Required search transparency" in output

    structured = result.structuredContent
    # Still no_match: a request for a decision is never a recommendation.
    assert structured["librarian_status"] == "no_match"
    assert structured["caller_action"] == "submit_librarian_choice_or_state_no_match"
    assert structured["librarian_recommendations"] == []


async def test_pending_routing_withholds_names_from_the_metadata_too(tmp_path):
    """Withholding a name in the text is worth nothing if metadata carries it.

    The text asks the caller to route on expertise and to obtain a name
    only from primo_submit_librarian_choice. A caller reads
    structuredContent just as easily, so a near-miss payload with name and
    email in it hands back precisely what the text declined to give.
    """
    result = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "librarian_llm_fallback": True,
                "librarian_llm_inline": True,
                "llm_provider": "caller",
                "librarian_min_score": 10_000.0,
            }
        ),
        "executive compensation",
        scope="catalogue",
    )

    contacts = result.structuredContent["closest_configured_contacts"]
    assert contacts, "near-miss evidence is the point of this payload"
    for contact in contacts:
        assert contact["id"]
        # Evidence survives anonymisation: the caller still needs to know
        # why this profile came close in order to decide.
        assert contact["evidence"]["matched_terms"] is not None
        assert "name" not in contact
        assert "email" not in contact
        assert "url" not in contact

    blob = json.dumps(result.structuredContent)
    assert "Accounting Librarian" not in blob
    assert "accounting@example.edu" not in blob


async def test_plain_no_match_still_names_the_closest_contacts(tmp_path):
    """Anonymising is scoped to a pending decision, not to every no-match.

    With no routing request outstanding there is no tool standing between
    the caller and a name, so stripping the identity fields would only
    remove the one contact a caller could offer, with its evidence.
    """
    result = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "librarian_min_score": 10_000.0,
            }
        ),
        "executive compensation",
        scope="catalogue",
    )

    contacts = result.structuredContent["closest_configured_contacts"]
    assert contacts
    assert all(contact["name"] for contact in contacts)


async def test_no_routing_request_means_no_caller_action(tmp_path):
    """A plain no-match owes the caller nothing, and must not claim to."""
    result = await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "librarian_min_score": 10_000.0,
            }
        ),
        "deep sea fishing quotas",
        scope="catalogue",
    )

    assert not _search_text(result).startswith("## Required librarian routing")
    assert result.structuredContent["librarian_status"] == "no_match"
    assert result.structuredContent["caller_action"] is None


async def test_submit_librarian_choice_publishes_its_contract_in_the_schema():
    """The id/confidence/reason contract must survive in the JSON schema.

    A client that reads the schema rather than the prose description has
    nothing else to go on, and a guessed key name fails silently -- the
    choice is dropped in validation and reads as "no librarian fits".
    """
    from primo_mcp_server.server import mcp

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    schema = tools["primo_submit_librarian_choice"].inputSchema

    choice = schema["$defs"]["LibrarianChoice"]
    assert set(choice["required"]) == {"id", "confidence", "reason"}
    assert choice["properties"]["confidence"]["type"] == "number"
    # Each field carries its own description, so the contract does not
    # depend on the caller having read the tool description.
    for field in ("id", "confidence", "reason"):
        assert choice["properties"][field]["description"].strip()


def test_policy_text_states_the_transparency_obligation():
    from primo_mcp_server.policy import (
        PRIMO_SEARCH_DESCRIPTION,
        SERVER_INSTRUCTIONS,
    )

    for text in (PRIMO_SEARCH_DESCRIPTION, SERVER_INSTRUCTIONS):
        assert "Search transparency policy for callers:" in text
        assert '"Queries attempted:" list' in text
        assert "number of results it returned" in text
        assert "including attempts that returned zero results" in text


def test_primo_search_description_documents_dataset_database_first_policy():
    from primo_mcp_server.policy import PRIMO_SEARCH_DESCRIPTION

    assert "For dataset or data-source requests" in PRIMO_SEARCH_DESCRIPTION
    assert 'scope="catalogue"' in PRIMO_SEARCH_DESCRIPTION
    assert 'resource_type="databases"' in PRIMO_SEARCH_DESCRIPTION
    assert "to articles or books" in PRIMO_SEARCH_DESCRIPTION
    assert "callers MUST include every recommended librarian's name" in (
        PRIMO_SEARCH_DESCRIPTION
    )


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
    output = _search_text(output)

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
    output = _search_text(output)

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
    output = _search_text(output)

    assert client.search_calls[0]["clauses"] == clauses
    assert "Unexpected error" not in output


async def test_primo_search_no_match_shows_closest_profiles_with_evidence(tmp_path):
    result = await primo_search(
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
    output = _search_text(result)

    assert "Status: no_match" in output
    assert "Closest configured profiles" in output
    assert "Evidence: matched terms:" in output
    assert "(below the confidence threshold)" in output
    assert "closest configured contact" in output
    assert result.structuredContent["librarian_status"] == "no_match"
    assert result.structuredContent["caller_action"] is None
    assert result.structuredContent["librarian_recommendations"] == []
    near_miss = result.structuredContent["closest_configured_contacts"][0]
    assert near_miss["name"] == "Accounting Librarian"
    assert near_miss["evidence"]["matched_terms"]


async def test_recommendation_outcomes_are_logged_when_opted_in(tmp_path):
    log_path = tmp_path / "recommend.jsonl"

    # A matched outcome and a no_match outcome (unreachable threshold)
    # both append one line.
    await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "recommend_log_file": str(log_path),
            }
        ),
        "executive compensation",
        scope="catalogue",
    )
    await primo_search(
        _fake_context(
            config_overrides={
                "librarians_file": _write_librarians_file(tmp_path),
                "recommend_log_file": str(log_path),
                "librarian_min_score": 10_000.0,
            }
        ),
        "executive compensation",
        scope="catalogue",
    )

    lines = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    matched, missed = lines
    assert matched["status"] == "matched"
    assert matched["query"] == "executive compensation"
    assert matched["matches"][0]["id"] == "accounting"
    assert matched["matches"][0]["terms"]
    assert missed["status"] == "no_match"
    assert missed["matches"] == []
    assert missed["near_misses"][0]["id"] == "accounting"
    assert "time" in matched


def test_logged_records_carry_only_matcher_fields_and_replay_as_eval_case(tmp_path):
    from primo_mcp_server.config import PrimoConfig
    from primo_mcp_server.evaluate_recommendations import EvalCase
    from primo_mcp_server.models import PrimoRecord
    from primo_mcp_server.recommendation import RecommendationOutcome
    from primo_mcp_server.librarians import EVIDENCE_TEXT_CAP as _LOGGED_TEXT_CAP
    from primo_mcp_server.server import _log_recommendation_outcome

    log_path = tmp_path / "recommend.jsonl"
    record = PrimoRecord(
        record_id="alma123",
        doi="10.1/x",
        title="Hospital outcomes",
        subjects=["Medicine", "Social science"],
        description="x" * (_LOGGED_TEXT_CAP + 500),
    )
    _log_recommendation_outcome(
        PrimoConfig(_env_file=None, recommend_log_file=str(log_path)),
        "medicine",
        RecommendationOutcome([]),
        [record],
    )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    logged = entry["records"][0]
    assert set(logged) == {"title", "subjects", "description"}
    assert len(logged["description"]) == _LOGGED_TEXT_CAP
    case = EvalCase.model_validate(
        {"query": entry["query"], "records": entry["records"]}
    )
    assert case.records[0].subjects == ["Medicine", "Social science"]


async def test_no_log_file_is_written_without_opt_in(tmp_path):
    await primo_search(
        _fake_context(
            config_overrides={"librarians_file": _write_librarians_file(tmp_path)}
        ),
        "executive compensation",
        scope="catalogue",
    )

    assert not list(tmp_path.glob("*.jsonl"))


async def test_lifespan_fires_local_embedding_warmup(monkeypatch):
    import asyncio

    from primo_mcp_server.server import app_lifespan, mcp

    calls: list = []

    async def fake_warmup(config):
        calls.append(config)

    monkeypatch.setattr(
        "primo_mcp_server.server.warm_up_local_embedder", fake_warmup
    )

    async with app_lifespan(mcp) as context:
        # Yield control so the background task runs.
        await asyncio.sleep(0)
        assert "client" in context and "config" in context

    assert len(calls) == 1


async def test_primo_search_forwards_facet_filters_to_client():
    client = _FakeClient()

    result = await primo_search(
        _fake_context(client=client),
        "economics",
        facet_filters={"topic": "Economics"},
        facet_exclusions={"rtype": "reviews"},
        recommend_librarians=False,
    )
    output = _search_text(result)

    assert client.search_calls[0]["facet_filters"] == {"topic": "Economics"}
    assert client.search_calls[0]["facet_exclusions"] == {"rtype": "reviews"}
    assert "Unexpected error" not in output


async def test_unexpected_tool_error_is_logged_with_traceback(caplog):
    import logging

    class _ExplodingClient(_FakeClient):
        async def search(self, **kwargs):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="primo_mcp_server.server"):
        result = await primo_search(
            _fake_context(client=_ExplodingClient()),
            "economics",
            recommend_librarians=False,
        )

    assert _search_text(result) == "Unexpected error: boom"
    record = next(
        r for r in caplog.records if "Unexpected error in primo_search" in r.message
    )
    assert record.exc_info is not None


# ---------------------------------------------------------------------------
# Caller-reasoned librarian routing (tier 3, "caller" backend).
# ---------------------------------------------------------------------------


def _routing_context(tmp_path):
    return _fake_context(
        config_overrides={
            "librarians_file": _write_librarians_file(tmp_path),
            "librarian_llm_fallback": True,
            "llm_provider": "caller",
            # Force a keyword miss so the routing tier engages.
            "librarian_min_score": 10_000.0,
        }
    )


async def test_keyword_miss_asks_the_caller_to_route(tmp_path):
    from primo_mcp_server.server import primo_recommend_librarians

    output = await primo_recommend_librarians(
        _routing_context(tmp_path), "deep sea fishing quotas"
    )

    assert "Status: no_match" in output
    assert "librarian routing needed" in output
    assert "Profile id: accounting" in output and "Profile id: data" in output
    assert "primo_submit_librarian_choice" in output
    # Names are withheld until a choice is validated, so the only way to
    # name a librarian is to go through the tool that re-checks the id.
    assert "Accounting Librarian" not in output


async def test_submitted_choice_is_validated_and_formatted(tmp_path):
    from primo_mcp_server.server import primo_submit_librarian_choice

    output = await primo_submit_librarian_choice(
        _routing_context(tmp_path),
        "audit datasets",
        [{"id": "accounting", "confidence": 0.88, "reason": "covers audit data"}],
    )

    assert "Status: matched (LLM reasoning)" in output
    assert "Accounting Librarian" in output
    assert "covers audit data" in output
    assert "self-reported confidence 0.88" in output


async def test_submitting_an_invented_id_names_no_librarian(tmp_path):
    """The gate: a fabricated id must never come back as a recommendation."""
    from primo_mcp_server.server import primo_submit_librarian_choice

    output = await primo_submit_librarian_choice(
        _routing_context(tmp_path),
        "deep sea fishing quotas",
        [{"id": "marine-biology", "confidence": 0.95, "reason": "invented"}],
    )

    assert "No submitted choice passed validation" in output
    assert "marine-biology" not in output
    assert "Status: matched" not in output
    # The caller is told what it may do instead of naming someone.
    assert "primo_list_librarians" in output


async def test_submission_without_a_directory_cannot_invent_one():
    from primo_mcp_server.server import primo_submit_librarian_choice

    output = await primo_submit_librarian_choice(
        _fake_context(),
        "anything",
        [{"id": "accounting", "confidence": 0.9, "reason": "x"}],
    )

    assert "Librarian directory unavailable" in output
