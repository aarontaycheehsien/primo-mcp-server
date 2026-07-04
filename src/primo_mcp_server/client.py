"""Async HTTP client for the Primo REST API."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import quote

import httpx

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.models import Facet, PrimoRecord, SearchResponse
from primo_mcp_server.query import (
    date_range_facet_value,
    normalise_resource_type,
    normalise_scope,
    normalise_search_field,
    normalise_sort_by,
)


def _normalise_alma_id(record_id: str) -> str:
    """Strip the Alma prefix for MMS-ID catalogue lookups and matching."""
    rid = record_id.strip()
    return rid[4:] if rid.lower().startswith("alma") else rid


class PrimoAPIError(Exception):
    """Raised when the Primo API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


def _normalise_parameter(func, *args):
    """Convert query validation errors into MCP-facing Primo errors."""
    try:
        return func(*args)
    except ValueError as e:
        raise PrimoAPIError(str(e), status_code=400) from e


def _normalise_scope(scope: str) -> str:
    """Resolve caller-friendly scope aliases to canonical scope names."""
    return _normalise_parameter(normalise_scope, scope)


def _normalise_search_field(field: str) -> str:
    """Resolve caller-friendly field aliases to Primo field names."""
    return _normalise_parameter(normalise_search_field, field)


def _normalise_sort_by(sort_by: str) -> str:
    """Resolve caller-friendly sort aliases to Primo sort names."""
    return _normalise_parameter(normalise_sort_by, sort_by)


def _normalise_resource_type(resource_type: str | None) -> str | None:
    """Resolve caller-friendly resource type aliases to Primo facet values."""
    return _normalise_parameter(normalise_resource_type, resource_type)


def _date_range_facet_value(date_from: str | None, date_to: str | None) -> str | None:
    """Return Primo's documented creation-date range facet value."""
    return _normalise_parameter(date_range_facet_value, date_from, date_to)


class PrimoClient:
    """Async client for the Ex Libris Primo public API."""

    def __init__(self, http_client: httpx.AsyncClient, config: PrimoConfig):
        self._http = http_client
        self._config = config
        self._guest_jwt_token: str | None = None
        self._guest_jwt_expiry: float = 0.0

    # -- Guest JWT handling -------------------------------------------------
    #
    # The primaws full-display endpoint (/pnxs/{context}/{docid}) requires an
    # Authorization header, unlike the search endpoint. Primo VE issues an
    # anonymous guest JWT to any caller -- no API key or credentials -- from
    # /institution/{code}/guestJwt. Tokens last about 24 hours; cache one and
    # refresh on expiry or on a 401/403.

    _JWT_SAFETY_MARGIN_SECONDS = 300.0
    _JWT_FALLBACK_LIFETIME_SECONDS = 1800.0

    def _institution_code(self) -> str:
        cfg = self._config
        if cfg.institution_code:
            return cfg.institution_code
        return cfg.vid.split(":", 1)[0]

    def _view_id(self) -> str:
        vid = self._config.vid
        return vid.split(":", 1)[1] if ":" in vid else vid

    @staticmethod
    def _jwt_expiry_epoch(token: str) -> float | None:
        """Read the exp claim from a JWT payload without verifying it."""
        try:
            payload_segment = token.split(".")[1]
            padded = payload_segment + "=" * (-len(payload_segment) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            exp = payload.get("exp")
            return float(exp) if exp is not None else None
        except Exception:
            return None

    async def _guest_jwt(self, force_refresh: bool = False) -> str:
        """Return a cached anonymous guest JWT, fetching one when needed."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._guest_jwt_token
            and now < self._guest_jwt_expiry
        ):
            return self._guest_jwt_token

        cfg = self._config
        path = f"/institution/{self._institution_code()}/guestJwt"
        params = {
            "vid": cfg.vid,
            "lang": cfg.language,
            "isGuest": "true",
            "viewId": self._view_id(),
        }
        try:
            response = await self._http.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise PrimoAPIError(
                f"Could not obtain a Primo guest token from {path}. "
                "Direct record lookup is unavailable; falling back to "
                f"search-based lookup. ({e})",
            ) from e

        token = response.text.strip().strip('"')
        if not token:
            raise PrimoAPIError("Primo guest token endpoint returned an empty token.")

        exp = self._jwt_expiry_epoch(token)
        if exp is not None:
            lifetime = max(exp - time.time() - self._JWT_SAFETY_MARGIN_SECONDS, 60.0)
        else:
            lifetime = self._JWT_FALLBACK_LIFETIME_SECONDS
        self._guest_jwt_token = token
        self._guest_jwt_expiry = now + lifetime
        return token

    async def search(
        self,
        query: str,
        field: str = "any",
        scope: str = "everything",
        sort_by: str = "rank",
        limit: int = 10,
        offset: int = 0,
        resource_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        peer_reviewed: bool | None = None,
        include_unavailable: bool | None = None,
        include_facets: bool | None = None,
    ) -> SearchResponse:
        """Search the Primo catalogue.

        Args:
            query: Search terms.
            field: Search field (any, title, creator, sub, isbn, oclcnum).
            scope: "everything" for local + PCI, "catalogue" for local only,
                "books_videos" for the institution's books/videos scope.
            sort_by: rank, date, or title.
            limit: Number of results (capped at max_results_per_request).
            offset: Pagination offset.
            resource_type: Filter by type (books, articles, journals, etc.).
            date_from: Start year (YYYY).
            date_to: End year (YYYY).
            peer_reviewed: Filter to peer-reviewed items only.
            include_unavailable: Include CDI records the institution has no
                full text access to (Primo's pcAvailability "expanded"
                search). None uses the configured default.
            include_facets: Also fetch the facet summary for the search.
                None uses the configured default; facet failures degrade
                to a response without facets.

        Returns:
            SearchResponse with parsed records, pagination info, and any
            available facets.
        """
        cfg = self._config
        limit = min(max(1, limit), cfg.max_results_per_request)
        offset = max(0, offset)

        if include_unavailable is None:
            include_unavailable = cfg.include_unavailable

        canonical_scope = _normalise_scope(scope)
        field = _normalise_search_field(field)
        sort_by = _normalise_sort_by(sort_by)
        resource_type = _normalise_resource_type(resource_type)
        date_range = _date_range_facet_value(date_from, date_to)

        # Resolve scope to tab + scope params
        if canonical_scope == "catalogue":
            tab = cfg.tab_catalogue
            scope_param = cfg.scope_local
        elif canonical_scope == "everything":
            tab = cfg.tab_everything
            scope_param = cfg.scope_combined
        else:
            tab = cfg.tab_books_videos
            scope_param = cfg.scope_books_videos

        params: dict[str, Any] = {
            "vid": cfg.vid,
            "tab": tab,
            "scope": scope_param,
            "q": f"{field},contains,{query}",
            "offset": str(offset),
            "limit": str(limit),
            "lang": cfg.language,
            "sortby": sort_by,
            "pcAvailability": "true" if include_unavailable else "false",
        }

        # Facet filters
        q_include: list[str] = []
        if resource_type:
            q_include.append(f"facet_rtype,exact,{resource_type}")
        if date_range:
            q_include.append(f"facet_searchcreationdate,exact,{date_range}")
        if peer_reviewed:
            q_include.append("facet_tlevel,exact,peer_reviewed")

        # Add all qInclude params
        if q_include:
            params["qInclude"] = "|,|".join(q_include)

        data = await self._get("/pnxs", params=params)
        response = SearchResponse.from_api_response(data)

        if include_facets is None:
            include_facets = cfg.search_facets
        if include_facets and response.records:
            response.facets = await self._fetch_facets(params)
        return response

    async def _fetch_facets(self, params: dict[str, Any]) -> list[Facet]:
        """Fetch the facet summary for the search just executed.

        Primo's /facets endpoint returns facets for the search the current
        cookie session already ran, so it must be called with the same
        parameters straight after the /pnxs request (the shared httpx client
        carries the session cookie). Facets are auxiliary context: any
        failure degrades to no facets rather than failing the search, and
        scopes Primo serves no facets for (e.g. the local catalogue scope)
        simply return an empty list.
        """
        try:
            data = await self._get("/facets", params=params)
        except PrimoAPIError:
            return []
        return Facet.list_from_api_response(data)

    async def get_record(self, record_id: str) -> PrimoRecord | None:
        """Fetch a single record by its Primo record ID.

        Tries the direct full-display endpoint first (authenticated with an
        anonymous guest JWT), then falls back to search-based lookup. In both
        paths a result is returned only when its record ID verifiably matches
        the requested ID, so callers never receive a different record than
        the one requested.
        """
        rid = record_id.strip()
        if not rid:
            return None

        record = await self._get_record_direct(rid)
        if record is not None:
            return record

        search_plan = self._record_search_plan(rid)

        for tab, scope_param, query in search_plan:
            params: dict[str, Any] = {
                "vid": self._config.vid,
                "tab": tab,
                "scope": scope_param,
                "q": f"any,contains,{query}",
                "offset": "0",
                "limit": "5",
                "lang": self._config.language,
            }
            data = await self._get("/pnxs", params=params)
            response = SearchResponse.from_api_response(data)

            for record in response.records:
                if self._record_ids_match(record.record_id, rid):
                    return record

        return None

    async def _get_record_direct(self, record_id: str) -> PrimoRecord | None:
        """Fetch a record from the /pnxs/{context}/{docid} endpoint.

        Returns None on any failure so get_record can fall back to
        search-based lookup. The endpoint requires a guest JWT; the response
        is a single PNX document with delivery data at the top level rather
        than inside the pnx block, so availability fields are mapped across
        before parsing.
        """
        contexts = (
            ("L", "PC")
            if record_id.lower().startswith("alma") or record_id.isdigit()
            else ("PC", "L")
        )

        try:
            token = await self._guest_jwt()
        except PrimoAPIError:
            return None

        for context in contexts:
            data = await self._fetch_direct(context, record_id, token)
            if data == "auth":
                try:
                    token = await self._guest_jwt(force_refresh=True)
                except PrimoAPIError:
                    return None
                data = await self._fetch_direct(context, record_id, token)
            if not isinstance(data, dict):
                continue

            doc = self._merge_direct_delivery(data)
            record = PrimoRecord.from_api_doc(doc)
            if self._record_ids_match(record.record_id, record_id):
                return record

        return None

    async def _fetch_direct(
        self, context: str, record_id: str, token: str
    ) -> dict[str, Any] | str | None:
        """Return the direct doc, "auth" for 401/403, or None on failure."""
        cfg = self._config
        encoded_record_id = quote(record_id, safe="")
        try:
            response = await self._http.get(
                f"/pnxs/{context}/{encoded_record_id}",
                params={"vid": cfg.vid, "lang": cfg.language},
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code in (401, 403):
                return "auth"
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("pnx"), dict):
                return None
            return data
        except (httpx.HTTPError, ValueError):
            return None

    @staticmethod
    def _merge_direct_delivery(data: dict[str, Any]) -> dict[str, Any]:
        """Map the direct endpoint's top-level delivery into pnx shape."""
        pnx = data.get("pnx", {})
        if not isinstance(pnx, dict):
            return data
        if pnx.get("delivery"):
            return data
        top_delivery = data.get("delivery", {})
        if not isinstance(top_delivery, dict):
            return data
        mapped = {
            "delcategory": top_delivery.get("deliveryCategory", []),
            "fulltext": top_delivery.get("availability", []),
        }
        merged = dict(data)
        merged["pnx"] = {**pnx, "delivery": mapped}
        return merged

    def _record_search_plan(self, record_id: str) -> list[tuple[str, str, str]]:
        """Return the search attempts used to resolve a Primo record ID."""
        rid = record_id.strip()
        normalised = _normalise_alma_id(rid)

        if rid.lower().startswith("alma") or rid.isdigit():
            queries = [rid]
            if normalised != rid:
                queries.append(normalised)
            alma_prefixed = f"alma{normalised}" if normalised.isdigit() else normalised
            if alma_prefixed not in queries:
                queries.append(alma_prefixed)
            return [
                (self._config.tab_catalogue, self._config.scope_local, query)
                for query in queries
            ] + [
                (self._config.tab_everything, self._config.scope_combined, query)
                for query in queries
            ]

        return [(self._config.tab_everything, self._config.scope_combined, rid)]

    @staticmethod
    def _record_ids_match(found_id: str, requested_id: str) -> bool:
        """Match exact IDs or equivalent Alma IDs with/without prefix."""
        found = found_id.strip()
        requested = requested_id.strip()
        return (
            found == requested
            or _normalise_alma_id(found) == _normalise_alma_id(requested)
        )

    async def suggest(self, query: str) -> list[str]:
        """Get autocomplete suggestions for a search term."""
        cfg = self._config
        params = {
            "vid": cfg.vid,
            "q": query,
            "lang": cfg.language,
        }
        data = await self._get("/suggest", params=params)

        # Extract suggestion texts
        response = data.get("response", {})
        docs = response.get("docs", [])
        return [doc.get("text", "") for doc in docs if doc.get("text")]

    async def get_records(self, record_ids: list[str]) -> list[PrimoRecord]:
        """Fetch multiple records by their IDs."""
        records = []
        for rid in record_ids:
            record = await self.get_record(rid)
            if record:
                records.append(record)
        return records

    async def _get(self, path: str, params: dict[str, Any]) -> dict:
        """Make a GET request to the Primo API."""
        try:
            response = await self._http.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise PrimoAPIError(
                f"Request timed out after {self._config.request_timeout}s. "
                "The Primo API may be slow or unavailable. Try again shortly.",
            ) from e
        except httpx.ConnectError as e:
            raise PrimoAPIError(
                f"Could not connect to {self._config.base_url}. "
                "Check your network connection and that the Primo API is available.",
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 400:
                raise PrimoAPIError(
                    f"Bad request (HTTP 400). Check your search query and parameters.",
                    status_code=400,
                ) from e
            elif status >= 500:
                raise PrimoAPIError(
                    f"Primo API server error (HTTP {status}). "
                    "The service may be experiencing issues. Try again later.",
                    status_code=status,
                ) from e
            else:
                raise PrimoAPIError(
                    f"Primo API returned HTTP {status}.",
                    status_code=status,
                ) from e
        except Exception as e:
            raise PrimoAPIError(
                f"Unexpected error querying Primo: {e}",
            ) from e
