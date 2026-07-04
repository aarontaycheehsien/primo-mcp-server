"""Tests for Primo API client request construction."""

from __future__ import annotations

import httpx
import pytest

from primo_mcp_server.client import PrimoAPIError, PrimoClient, _normalise_scope
from primo_mcp_server.config import PrimoConfig


def _config() -> PrimoConfig:
    return PrimoConfig(
        base_url="https://example.test/primaws/rest/pub",
        vid="TEST:VID",
        tab_catalogue="Catalogue",
        tab_everything="Everything",
        tab_books_videos="booksandvideos",
        scope_local="MyInstitution",
        scope_combined="MyInst_and_CI",
        scope_books_videos="BooksVideos",
    )


def _empty_response() -> httpx.Response:
    return httpx.Response(200, json={"info": {"total": 0}, "docs": []})


class TestScopeResolution:
    def test_scope_aliases(self):
        assert _normalise_scope("catalog") == "catalogue"
        assert _normalise_scope("my_institution") == "catalogue"
        assert _normalise_scope("all") == "everything"
        assert _normalise_scope("myinst_and_ci") == "everything"
        assert _normalise_scope("books/videos") == "books_videos"
        assert _normalise_scope("books & videos") == "books_videos"
        assert _normalise_scope("booksandvideos") == "books_videos"

    def test_invalid_scope_raises_clear_error(self):
        with pytest.raises(PrimoAPIError, match='Invalid scope "catalogg"'):
            _normalise_scope("catalogg")


class TestSearchRequestScopes:
    @pytest.mark.parametrize(
        ("scope", "expected_tab", "expected_scope"),
        [
            ("catalogue", "Catalogue", "MyInstitution"),
            ("everything", "Everything", "MyInst_and_CI"),
            ("books_videos", "booksandvideos", "BooksVideos"),
            ("BooksVideos", "booksandvideos", "BooksVideos"),
        ],
    )
    async def test_search_maps_scope_to_primo_params(
        self,
        scope: str,
        expected_tab: str,
        expected_scope: str,
    ):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            await client.search("Singapore", scope=scope)

        params = requests[0].url.params
        assert params["tab"] == expected_tab
        assert params["scope"] == expected_scope

    async def test_invalid_scope_does_not_make_request(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            with pytest.raises(PrimoAPIError):
                await client.search("Singapore", scope="catalogg")

        assert requests == []


class TestSearchRequestFilters:
    async def test_search_normalises_field_sort_and_resource_type_aliases(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            await client.search(
                "Singapore",
                field="subject",
                sort_by="newest",
                resource_type="book",
            )

        params = requests[0].url.params
        assert params["q"] == "sub,contains,Singapore"
        assert params["sortby"] == "date"
        assert params["qInclude"] == "facet_rtype,exact,books"

    async def test_search_uses_documented_date_range_facet(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            await client.search("Singapore", date_from="2020", date_to="2022")

        params = requests[0].url.params
        assert (
            params["qInclude"]
            == "facet_searchcreationdate,exact,[2020 TO 2022]"
        )

    async def test_search_uses_exact_year_range_for_date_from_only(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            await client.search("Singapore", date_from="2020")

        params = requests[0].url.params
        assert (
            params["qInclude"]
            == "facet_searchcreationdate,exact,[2020 TO 2020]"
        )

    async def test_search_combines_date_range_with_other_filters(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            await client.search(
                "Singapore",
                resource_type="articles",
                date_from="2020",
                date_to="2022",
                peer_reviewed=True,
            )

        params = requests[0].url.params
        assert params["qInclude"] == (
            "facet_rtype,exact,articles|,|"
            "facet_searchcreationdate,exact,[2020 TO 2022]|,|"
            "facet_tlevel,exact,peer_reviewed"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"field": "badfield"},
            {"sort_by": "recently"},
            {"resource_type": "podcast"},
            {"date_from": "20XX"},
            {"date_to": "2022"},
            {"date_from": "2024", "date_to": "2020"},
        ],
    )
    async def test_invalid_search_parameters_do_not_make_request(self, kwargs):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            with pytest.raises(PrimoAPIError):
                await client.search("Singapore", **kwargs)

        assert requests == []


class TestGetRecord:
    def _alma_doc(self) -> dict:
        return {
            "pnx": {
                "control": {"recordid": ["alma99317560802601"]},
                "display": {"title": ["Anyuan"], "type": ["book"]},
            }
        }

    async def test_get_record_resolves_prefixed_alma_id_with_numeric_lookup(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path or "/pnxs/" in request.url.path:
                return httpx.Response(404)
            requests.append(request)
            q = request.url.params["q"]
            if q == "any,contains,99317560802601":
                return httpx.Response(
                    200,
                    json={"info": {"total": 1}, "docs": [self._alma_doc()]},
                )
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is not None
        assert record.record_id == "alma99317560802601"
        assert requests[0].url.params["scope"] == "MyInstitution"
        assert requests[1].url.params["q"] == "any,contains,99317560802601"

    async def test_get_record_resolves_numeric_alma_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path or "/pnxs/" in request.url.path:
                return httpx.Response(404)
            q = request.url.params["q"]
            if q == "any,contains,99317560802601":
                return httpx.Response(
                    200,
                    json={"info": {"total": 1}, "docs": [self._alma_doc()]},
                )
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("99317560802601")

        assert record is not None
        assert record.record_id == "alma99317560802601"


def _fake_jwt(exp_offset: int = 3600) -> str:
    import base64 as _b64
    import json as _json
    import time as _time

    payload = _json.dumps({"exp": int(_time.time()) + exp_offset}).encode()
    seg = _b64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"header.{seg}.sig"


class TestGetRecordDirect:
    def _direct_doc(self) -> dict:
        return {
            "context": "L",
            "pnx": {
                "control": {"recordid": ["alma99317560802601"]},
                "display": {"title": ["Anyuan"], "type": ["book"]},
            },
            "delivery": {
                "deliveryCategory": ["Alma-P"],
                "availability": ["available_in_library"],
            },
        }

    async def test_direct_lookup_uses_guest_jwt_and_skips_search(self):
        requests: list[httpx.Request] = []
        token = _fake_jwt()

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if "guestJwt" in request.url.path:
                return httpx.Response(200, json=token)
            if request.url.path.endswith("/pnxs/L/alma99317560802601"):
                assert request.headers["Authorization"] == f"Bearer {token}"
                return httpx.Response(200, json=self._direct_doc())
            raise AssertionError(f"unexpected request: {request.url}")

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is not None
        assert record.record_id == "alma99317560802601"
        assert record.delivery_category == "Alma-P"
        assert len(requests) == 2

    async def test_direct_lookup_url_encodes_record_id_path_segment(self):
        token = _fake_jwt()
        record_id = "cdi/example id:10.123/a b"

        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path:
                return httpx.Response(200, json=token)
            if "/pnxs/" in request.url.path:
                assert (
                    b"/pnxs/PC/cdi%2Fexample%20id%3A10.123%2Fa%20b"
                    in request.url.raw_path
                )
                return httpx.Response(
                    200,
                    json={
                        "context": "PC",
                        "pnx": {
                            "control": {"recordid": [record_id]},
                            "display": {"title": ["Remote"], "type": ["article"]},
                        },
                    },
                )
            raise AssertionError(f"unexpected request: {request.url}")

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record(record_id)

        assert record is not None
        assert record.record_id == record_id

    async def test_direct_lookup_refreshes_jwt_on_403(self):
        stale, fresh = _fake_jwt(3600), _fake_jwt(7200)
        issued: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path:
                token = stale if not issued else fresh
                issued.append(token)
                return httpx.Response(200, json=token)
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {stale}":
                return httpx.Response(403)
            return httpx.Response(200, json=self._direct_doc())

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is not None
        assert issued == [stale, fresh]

    async def test_falls_back_to_search_when_jwt_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path:
                return httpx.Response(500)
            if request.url.path.endswith("/pnxs"):
                q = request.url.params["q"]
                if "99317560802601" in q:
                    return httpx.Response(
                        200,
                        json={
                            "info": {"total": 1},
                            "docs": [
                                {
                                    "pnx": {
                                        "control": {
                                            "recordid": ["alma99317560802601"]
                                        },
                                        "display": {
                                            "title": ["Anyuan"],
                                            "type": ["book"],
                                        },
                                    }
                                }
                            ],
                        },
                    )
                return _empty_response()
            raise AssertionError(f"unexpected request: {request.url}")

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is not None
        assert record.record_id == "alma99317560802601"

    @pytest.mark.parametrize(
        "direct_response",
        [
            httpx.Response(200, text="not-json"),
            httpx.Response(200, json={"pnx": "malformed"}),
        ],
    )
    async def test_falls_back_to_search_when_direct_response_is_malformed(
        self, direct_response: httpx.Response
    ):
        token = _fake_jwt()

        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path:
                return httpx.Response(200, json=token)
            if "/pnxs/" in request.url.path:
                return direct_response
            if request.url.path.endswith("/pnxs"):
                q = request.url.params["q"]
                if "99317560802601" in q:
                    return httpx.Response(
                        200,
                        json={
                            "info": {"total": 1},
                            "docs": [self._direct_doc()],
                        },
                    )
                return _empty_response()
            raise AssertionError(f"unexpected request: {request.url}")

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is not None
        assert record.record_id == "alma99317560802601"

    async def test_direct_mismatch_returns_none_not_wrong_record(self):
        token = _fake_jwt()

        def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path:
                return httpx.Response(200, json=token)
            if "/pnxs/" in request.url.path:
                wrong = self._direct_doc()
                wrong["pnx"]["control"]["recordid"] = ["alma990000000000001"]
                return httpx.Response(200, json=wrong)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            record = await client.get_record("alma99317560802601")

        assert record is None


class TestPcAvailability:
    """pcAvailability must be controllable, not hardcoded to true."""

    @staticmethod
    async def _search_params(config: PrimoConfig, **kwargs) -> httpx.QueryParams:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, config)
            await client.search("Singapore", **kwargs)
        return requests[0].url.params

    async def test_default_excludes_unavailable(self):
        params = await self._search_params(_config())
        assert params["pcAvailability"] == "false"

    async def test_explicit_true_includes_unavailable(self):
        params = await self._search_params(_config(), include_unavailable=True)
        assert params["pcAvailability"] == "true"

    async def test_explicit_false_overrides_config_default(self):
        config = _config()
        config.include_unavailable = True
        params = await self._search_params(config, include_unavailable=False)
        assert params["pcAvailability"] == "false"

    async def test_none_falls_back_to_config_default(self):
        config = _config()
        config.include_unavailable = True
        params = await self._search_params(config, include_unavailable=None)
        assert params["pcAvailability"] == "true"


def _search_payload() -> dict:
    return {
        "info": {"total": 1},
        "docs": [
            {
                "pnx": {
                    "control": {"recordid": ["alma1"]},
                    "display": {"title": ["A title"]},
                }
            }
        ],
    }


_FACETS_PAYLOAD = {
    "facets": [
        {"name": "rtype", "values": [{"value": "articles", "count": "42"}]}
    ]
}


class TestSearchFacets:
    async def _run_search(self, handler, **kwargs):
        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            return await client.search("Singapore", **kwargs)

    async def test_search_fetches_facets_with_same_query(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/facets"):
                return httpx.Response(200, json=_FACETS_PAYLOAD)
            return httpx.Response(200, json=_search_payload())

        response = await self._run_search(handler)

        assert [r.url.path for r in requests] == [
            "/primaws/rest/pub/pnxs",
            "/primaws/rest/pub/facets",
        ]
        # The /facets endpoint returns facets for the query the session just
        # ran, so the parameters must match the /pnxs request.
        assert requests[1].url.params["q"] == requests[0].url.params["q"]
        assert requests[1].url.params["tab"] == requests[0].url.params["tab"]
        assert [(f.name, f.values[0].count) for f in response.facets] == [
            ("rtype", 42)
        ]

    async def test_facets_failure_degrades_to_no_facets(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/facets"):
                return httpx.Response(500)
            return httpx.Response(200, json=_search_payload())

        response = await self._run_search(handler)
        assert response.records
        assert response.facets == []

    async def test_no_facets_request_for_empty_results(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        response = await self._run_search(handler)
        assert [r.url.path for r in requests] == ["/primaws/rest/pub/pnxs"]
        assert response.facets == []

    async def test_include_facets_false_skips_facets_request(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_search_payload())

        response = await self._run_search(handler, include_facets=False)
        assert [r.url.path for r in requests] == ["/primaws/rest/pub/pnxs"]
        assert response.facets == []

    async def test_config_default_can_disable_facets(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_search_payload())

        config = _config()
        config.search_facets = False
        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, config)
            response = await client.search("Singapore")

        assert [r.url.path for r in requests] == ["/primaws/rest/pub/pnxs"]
        assert response.facets == []


def _search_doc(record_id: str) -> dict:
    return {
        "pnx": {
            "control": {"recordid": [record_id]},
            "display": {"title": [f"Title {record_id}"]},
        }
    }


class TestGetRecords:
    def _http_client(self, handler) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        )

    async def test_preserves_input_order_and_drops_missing_ids(self):
        import asyncio

        async def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path or "/pnxs/" in request.url.path:
                return httpx.Response(404)
            rid = request.url.params["q"].rsplit(",", 1)[-1]
            if rid == "cdi_slow":
                # The slowest lookup is the FIRST id; order must still hold.
                await asyncio.sleep(0.05)
                return httpx.Response(
                    200, json={"info": {"total": 1}, "docs": [_search_doc(rid)]}
                )
            if rid == "cdi_fast":
                return httpx.Response(
                    200, json={"info": {"total": 1}, "docs": [_search_doc(rid)]}
                )
            return _empty_response()

        async with self._http_client(handler) as http_client:
            client = PrimoClient(http_client, _config())
            records = await client.get_records(
                ["cdi_slow", "cdi_missing", "cdi_fast"]
            )

        assert [r.record_id for r in records] == ["cdi_slow", "cdi_fast"]

    async def test_lookups_overlap_instead_of_running_sequentially(self):
        import asyncio

        inflight = 0
        max_inflight = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal inflight, max_inflight
            if "guestJwt" in request.url.path or "/pnxs/" in request.url.path:
                return httpx.Response(404)
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            await asyncio.sleep(0.02)
            inflight -= 1
            rid = request.url.params["q"].rsplit(",", 1)[-1]
            return httpx.Response(
                200, json={"info": {"total": 1}, "docs": [_search_doc(rid)]}
            )

        async with self._http_client(handler) as http_client:
            client = PrimoClient(http_client, _config())
            records = await client.get_records(["cdi_a", "cdi_b", "cdi_c"])

        assert len(records) == 3
        assert max_inflight > 1

    async def test_propagates_lookup_errors_after_all_settle(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if "guestJwt" in request.url.path or "/pnxs/" in request.url.path:
                return httpx.Response(404)
            rid = request.url.params["q"].rsplit(",", 1)[-1]
            if rid == "cdi_bad":
                return httpx.Response(500)
            return httpx.Response(
                200, json={"info": {"total": 1}, "docs": [_search_doc(rid)]}
            )

        async with self._http_client(handler) as http_client:
            client = PrimoClient(http_client, _config())
            with pytest.raises(PrimoAPIError):
                await client.get_records(["cdi_good", "cdi_bad"])

    async def test_concurrent_lookups_share_one_guest_jwt_fetch(self):
        token = _fake_jwt()
        jwt_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal jwt_calls
            if "guestJwt" in request.url.path:
                jwt_calls += 1
                return httpx.Response(200, json=token)
            if "/pnxs/" in request.url.path:
                rid = request.url.path.rsplit("/", 1)[-1]
                return httpx.Response(
                    200,
                    json={
                        "context": "PC",
                        "pnx": {
                            "control": {"recordid": [rid]},
                            "display": {"title": [f"Title {rid}"]},
                        },
                    },
                )
            raise AssertionError(f"unexpected request: {request.url}")

        async with self._http_client(handler) as http_client:
            client = PrimoClient(http_client, _config())
            records = await client.get_records(["cdi_a", "cdi_b", "cdi_c"])

        assert [r.record_id for r in records] == ["cdi_a", "cdi_b", "cdi_c"]
        assert jwt_calls == 1


class TestSearchClauses:
    async def _search(self, handler, **kwargs):
        async with httpx.AsyncClient(
            base_url="https://example.test/primaws/rest/pub",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrimoClient(http_client, _config())
            return await client.search("summary text", **kwargs)

    async def test_clauses_compile_to_multi_clause_q(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        await self._search(
            handler,
            clauses=[
                {"field": "title", "value": "capital", "connector": "AND"},
                {"field": "creator", "value": "piketty"},
            ],
        )

        assert requests[0].url.params["q"] == (
            "title,contains,capital,AND;creator,contains,piketty"
        )

    async def test_clause_aliases_and_value_sanitisation(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        await self._search(
            handler,
            clauses=[
                {
                    "field": "author",
                    "operator": "equals",
                    "value": "Piketty, Thomas; economist",
                    "connector": "or",
                },
                # Trailing connector on the last clause is ignored.
                {"value": "wealth", "connector": "NOT"},
            ],
        )

        assert requests[0].url.params["q"] == (
            "creator,exact,Piketty Thomas economist,OR;any,contains,wealth"
        )

    async def test_invalid_clause_operator_makes_no_request(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        with pytest.raises(PrimoAPIError, match="Invalid operator"):
            await self._search(
                handler,
                clauses=[{"value": "capital", "operator": "fuzzy"}],
            )
        assert requests == []

    async def test_empty_clause_value_makes_no_request(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _empty_response()

        with pytest.raises(PrimoAPIError, match="empty value"):
            await self._search(handler, clauses=[{"value": " ,; "}])
        assert requests == []
