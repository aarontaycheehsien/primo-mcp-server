"""Live checks against the real SMU Primo API.

Every other test mocks Primo, so these are the only ones that can catch a
wrong assumption about the service itself, or a change on the Primo side.
Each test checks one behaviour the code relies on. They are deselected by
default (pyproject addopts) and never run in CI:

    uv run pytest -m live -v

They use the SMU defaults regardless of the local .env (PRIMO_* environment
variables still override), run sequentially, send about 30 requests, and
assert properties rather than exact counts, since holdings change daily.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from primo_mcp_server.citations import CITATION_STYLES, format_citation
from primo_mcp_server.client import PrimoClient
from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.exporters import export_bibtex, export_csv, export_ris
from primo_mcp_server.formatter import format_record_detail, format_search_results

pytestmark = pytest.mark.live


@pytest.fixture
def config() -> PrimoConfig:
    return PrimoConfig(_env_file=None)


@pytest.fixture
async def client(config: PrimoConfig):
    async with httpx.AsyncClient(
        base_url=config.base_url,
        timeout=config.request_timeout,
        headers={"User-Agent": config.user_agent},
    ) as http_client:
        yield PrimoClient(http_client, config)


def _rtype_counts(response) -> dict[str, int]:
    for facet in response.facets:
        if facet.name == "rtype":
            return {v.value: v.count for v in facet.values}
    return {}


# 1
async def test_search_returns_parseable_records(client, config):
    response = await client.search("Singapore", scope="everything", limit=5)

    assert response.info.total > 0
    assert response.records
    assert all(r.record_id and r.title for r in response.records)
    text = format_search_results(response, "Singapore", config=config)
    assert "Queries attempted:" in text
    assert "Unexpected error" not in text


# 2
@pytest.mark.parametrize("scope", ["catalogue", "everything", "books_videos"])
async def test_every_configured_scope_returns_results(client, scope):
    response = await client.search("Singapore", scope=scope, limit=3)

    assert response.info.total > 0, f"scope {scope} returned nothing"


# 3
async def test_query_with_semicolon_is_accepted(client):
    response = await client.search("inflation; monetary policy", limit=3)

    assert response.info.total > 0


# 4
async def test_creator_search_with_comma_still_matches(client):
    response = await client.search(
        "Lee, Kuan Yew", field="creator", scope="catalogue", limit=3
    )

    assert response.info.total > 0


# 5
async def test_date_range_filter_bounds_record_years(client):
    ranged = await client.search(
        "economics", date_from="2020", date_to="2021", limit=10
    )
    single = await client.search("economics", date_from="2020", limit=10)

    # Primo's date filter is loose: a CDI ebook whose every date field says
    # 2017 was returned for [2020 TO 2020]. The request is right, so require
    # most records in range, not all (the tool description warns callers).
    for response, allowed in ((ranged, {"2020", "2021"}), (single, {"2020"})):
        years = [r.year for r in response.records if r.year]
        assert years
        in_range = sum(year in allowed for year in years)
        assert in_range / len(years) >= 0.8, f"{allowed}: {years}"


# 6
async def test_books_filter_returns_books(client):
    response = await client.search(
        "Singapore", scope="catalogue", resource_type="books", limit=10
    )

    assert response.records
    types = {r.resource_type for r in response.records}
    assert all(t.lower().startswith("book") for t in types), types


# 7
async def test_peer_reviewed_filter_records_parse_as_peer_reviewed(client):
    # Primo's peer-reviewed filter also returns local journal titles that
    # carry no peer-review marker anywhere in their data, so those cannot
    # be labelled. Every other record must parse the marker.
    response = await client.search("economics", peer_reviewed=True, limit=10)

    assert any(r.peer_reviewed for r in response.records)
    unexplained = [
        (r.record_id, r.resource_type)
        for r in response.records
        if not r.peer_reviewed and r.resource_type != "journal"
    ]
    assert not unexplained, f"unflagged non-journal records: {unexplained}"


# 8
async def test_everything_search_serves_consistent_facets(client):
    response = await client.search(
        "Singapore", scope="everything", limit=3, include_facets=True
    )

    counts = _rtype_counts(response)
    assert counts, "no rtype facet returned"
    assert max(counts.values()) <= response.info.total


# 9
async def test_concurrent_searches_each_get_their_own_facets(client):
    alone_a = await client.search("coral reef ecology", limit=3, include_facets=True)
    alone_b = await client.search("corporate governance", limit=3, include_facets=True)

    together_a, together_b = await asyncio.gather(
        client.search("coral reef ecology", limit=3, include_facets=True),
        client.search("corporate governance", limit=3, include_facets=True),
    )

    assert _rtype_counts(alone_a) != _rtype_counts(alone_b), "queries too similar to tell apart"
    assert _rtype_counts(together_a) == _rtype_counts(alone_a)
    assert _rtype_counts(together_b) == _rtype_counts(alone_b)


# 10
async def test_guest_token_is_issued_with_a_readable_expiry(client):
    token = await client._guest_jwt()

    assert token
    assert PrimoClient._jwt_expiry_epoch(token) is not None


# 11
async def test_alma_record_resolves_through_the_direct_endpoint(client, config):
    found = await client.search("Singapore", scope="catalogue", limit=5)
    alma = next(r for r in found.records if r.record_id.startswith("alma"))

    search_paths: list[str] = []
    original_get = client._get

    async def spy(path, params):
        search_paths.append(path)
        return await original_get(path, params)

    client._get = spy
    record = await client.get_record(alma.record_id)

    assert record is not None and record.record_id == alma.record_id
    assert search_paths == [], "direct lookup failed; fell back to search"
    assert "Record ID:" in format_record_detail(record, config=config)


# 12
async def test_cdi_record_resolves_by_id(client):
    # Local records outrank CDI for many topics (none in the top 50 for
    # "corporate governance"), so use a topic SMU holds little on.
    found = await client.search("coral reef ecology", scope="everything", limit=20)
    cdi = next((r for r in found.records if r.record_id.startswith("cdi_")), None)
    assert cdi is not None, "no CDI record in the top 20; pick another query"

    record = await client.get_record(cdi.record_id)

    assert record is not None and record.record_id == cdi.record_id


# 13
async def test_live_records_cite_and_export_in_every_format(client):
    response = await client.search("corporate governance", limit=5)
    records = response.records
    assert records

    for style in CITATION_STYLES:
        for record in records:
            assert format_citation(record, style).strip()
    assert export_bibtex(records).count("@") == len(records)
    assert export_ris(records).count("ER  - ") == len(records)
    assert export_csv(records).count("\n") >= len(records) + 1


# 14
async def test_suggest_returns_completions(client):
    suggestions = await client.suggest("entrepre")

    assert suggestions
