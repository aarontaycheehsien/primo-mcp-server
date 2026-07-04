"""Pydantic models for Primo PNX response data.

Primo's API returns inconsistent field shapes -- the same field may be
a string, a list of strings, or missing entirely. These models normalise
everything into predictable types.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


def _to_list(v: Any) -> list[str]:
    """Normalise a Primo field into a list of strings."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return [str(item) for item in v if item is not None]
    return [str(v)]


def _first_or_empty(v: str | list[str] | None) -> str:
    """Extract the first element, or return empty string."""
    items = _to_list(v)
    return items[0] if items else ""


# Tokens in delivery/fulltext that explicitly signal NO full text. Observed
# live values are "fulltext", "fulltext_linktorsrc", "fulltext_multiple"
# (available) and "no_fulltext" (not available); "fulltext_not_available"
# is included defensively.
_FULLTEXT_NEGATIVE = {"no_fulltext", "fulltext_not_available"}


def _fulltext_available(value: str | list[str] | None) -> bool:
    """Return True when any delivery/fulltext token signals available full text.

    Tokens are matched exactly against known negatives and by prefix for
    positives. The previous substring test ("fulltext" in str(value)) also
    matched "no_fulltext", so records with no full text were reported as
    available.
    """
    for token in _to_list(value):
        t = str(token).strip().lower()
        if t in _FULLTEXT_NEGATIVE:
            continue
        if t.startswith("fulltext"):
            return True
    return False


# MARC relator terms that Primo appends to display names, e.g.
# "Mueller, John, 1958- author." -- stripped for clean author display.
_RELATORS = (
    "joint author", "issuing body", "edited by", "author", "authors",
    "editor", "editors", "narrator", "translator", "translators",
    "illustrator", "compiler", "contributor", "writer", "interviewer",
    "interviewee", "performer", "director", "producer", "composer",
    "photographer",
)
# Require the trailing period that Primo always appends ("author.", "editor."),
# so a genuine name segment that happens to be a relator word is left intact.
_RELATOR_RE = re.compile(
    r"[,\s]+(?:" + "|".join(re.escape(r) for r in _RELATORS) + r")\.\s*$",
    re.IGNORECASE,
)
_SUBFIELD_RE = re.compile(r"\$\$([A-Za-z])([^$]*)")
_VALUE_SEPARATORS_RE = re.compile(r"[;\uff1b]")
# Prefixes sometimes carried by DOI values: "doi:10.x", "DOI: 10.x",
# "https://doi.org/10.x", "http://dx.doi.org/10.x".
_DOI_PREFIX_RE = re.compile(
    r"^(?:doi\s*:?\s*|https?://(?:dx\.)?doi\.org/)+", re.IGNORECASE
)


def _clean_doi(value: str) -> str:
    """Return a bare DOI, stripping doi: labels and resolver URL prefixes."""
    return _DOI_PREFIX_RE.sub("", value.strip()).strip()


def _parse_identifiers(raw_values: list[str]) -> tuple[list[str], str]:
    """Parse display/identifier into clean strings and extract the first DOI.

    Primo encodes identifiers in two shapes, often semicolon-joined within
    a single string: PNX subfields ("$$CISBN$$V981-15-1967-6", where C is
    the type code and V the value) and plain labels ("DOI: 10.1234/x").
    The previous extraction detected "DOI:" case-insensitively but split on
    it case-sensitively, so lowercase "doi:" values kept their prefix and
    produced broken https://doi.org/doi:10.x links, and subfield-encoded
    DOIs were never detected at all. Identifiers are returned in readable
    "CODE: value" form rather than raw subfield strings.
    """
    cleaned: list[str] = []
    doi = ""
    for raw in raw_values:
        for part in _VALUE_SEPARATORS_RE.split(raw):
            part = part.strip()
            if not part:
                continue
            code = ""
            value = part
            if "$$" in part:
                pairs = _SUBFIELD_RE.findall(part)
                code = next((t.strip() for c, t in pairs if c == "C"), "")
                value = next(
                    (t.strip() for c, t in pairs if c == "V" and t.strip()), ""
                ) or _strip_subfields(part)
            elif ":" in part and not part.lower().startswith(
                ("http://", "https://")
            ):
                head, tail = part.split(":", 1)
                if tail.strip():
                    code, value = head.strip(), tail.strip()
            if not value:
                continue
            cleaned.append(f"{code}: {value}" if code else value)
            if not doi and (code.upper() == "DOI" or _DOI_PREFIX_RE.match(value)):
                doi = _clean_doi(value)
    return cleaned, doi


def _strip_subfields(value: str) -> str:
    """Drop Primo PNX '$$X' subfield codes, returning the human display text.

    Primo encodes subfields as '<display>$$Q<normalised>$$...'. The display
    text precedes the first '$$'; purely coded values fall back to the
    $$Q/$$V subfield content.
    """
    if not value or "$$" not in value:
        return value.strip() if value else ""
    head = value.split("$$", 1)[0].strip()
    if head:
        return head
    matches = _SUBFIELD_RE.findall(value)
    for preferred in ("N", "Q", "V", "a", "T", "L", "F"):
        for code, text in matches:
            if code == preferred and text.strip():
                return text.strip()
    for _code, text in matches:
        if text.strip():
            return text.strip()
    return value.strip()


def _clean_names(
    raw_values: list[str],
    *,
    split: bool = True,
    strip_relators: bool = True,
) -> list[str]:
    """Normalise name fields.

    For display.creator/contributor (split=True, strip_relators=True) this
    splits semicolon-joined entries and removes PNX subfield codes ($$Q...)
    and trailing MARC relator terms. For the already-clean structured
    addata.au/addau lists, pass split=False, strip_relators=False so only
    defensive subfield stripping is applied.
    """
    names: list[str] = []
    for raw in raw_values:
        base = _strip_subfields(raw)
        for part in (_VALUE_SEPARATORS_RE.split(base) if split else [base]):
            name = part.strip()
            if strip_relators:
                name = _RELATOR_RE.sub("", name)
            name = name.strip().rstrip(",").strip()
            if name:
                names.append(name)
    return names


class HoldingLocation(BaseModel):
    """A physical holding location for a record."""

    library: str = ""
    location: str = ""
    call_number: str = ""
    status: str = ""


class AccessLink(BaseModel):
    """A direct electronic access link for a record."""

    label: str = ""
    url: str = ""


def _parse_locations(delivery: dict) -> list[HoldingLocation]:
    """Parse bestlocation/holding entries from a doc-level delivery block."""
    raw_entries: list[Any] = []
    best = delivery.get("bestlocation")
    if isinstance(best, dict):
        raw_entries.append(best)
    holding = delivery.get("holding")
    if isinstance(holding, list):
        raw_entries.extend(holding)

    locations: list[HoldingLocation] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        location = HoldingLocation(
            library=str(entry.get("mainLocation") or "").strip(),
            location=str(entry.get("subLocation") or "").strip(),
            call_number=str(entry.get("callNumber") or "").strip(),
            status=str(entry.get("availabilityStatus") or "").strip(),
        )
        # bestlocation usually repeats the first holding; dedupe on shelf.
        key = (location.library, location.location, location.call_number)
        if any(key) and key not in seen:
            seen.add(key)
            locations.append(location)
    return locations


def _parse_access_links(delivery: dict) -> list[AccessLink]:
    """Parse direct full-text links (linktorsrc) from a delivery block."""
    links: list[AccessLink] = []
    seen: set[str] = set()
    for raw in delivery.get("link") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("linkType") or "").strip().lower() != "linktorsrc":
            continue
        url = str(raw.get("linkURL") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        label = str(raw.get("displayLabel") or "").strip()
        # Unresolved labels come through as PNX subfield codes ("$$Elinktorsrc").
        if not label or label.startswith("$$"):
            label = "Full text"
        links.append(AccessLink(label=label, url=url))
    return links


def _parse_openurl(delivery: dict) -> str:
    """Return the Alma link-resolver openurl, made safe for Markdown links."""
    openurl = str(delivery.get("almaOpenurl") or "").strip()
    # Live openurls carry literal spaces (e.g. in ctx_tim timestamps).
    return openurl.replace(" ", "%20")


class PrimoRecord(BaseModel):
    """A normalised Primo catalogue record."""

    # Identity
    record_id: str = ""
    source_id: str = ""
    source_system: str = ""

    # Display
    title: str = ""
    resource_type: str = ""
    language: str = ""
    creators: list[str] = Field(default_factory=list)
    contributors: list[str] = Field(default_factory=list)
    publisher: str = ""
    creation_date: str = ""
    source_label: str = ""
    description: str = ""
    snippet: str = ""
    subjects: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    is_part_of: str = ""

    # Identifiers
    identifiers: list[str] = Field(default_factory=list)
    doi: str = ""
    isbn: list[str] = Field(default_factory=list)
    issn: list[str] = Field(default_factory=list)

    # Academic data
    journal_title: str = ""
    volume: str = ""
    issue: str = ""
    start_page: str = ""
    end_page: str = ""
    peer_reviewed: bool = False
    ris_type: str = ""
    authors_structured: list[str] = Field(default_factory=list)
    additional_authors: list[str] = Field(default_factory=list)

    # Availability
    fulltext_available: bool = False
    delivery_category: str = ""
    # From the doc-level delivery block: where physical copies sit, direct
    # electronic access links, and the Alma link-resolver openurl.
    locations: list[HoldingLocation] = Field(default_factory=list)
    access_links: list[AccessLink] = Field(default_factory=list)
    openurl: str = ""

    # Relevance
    score: float = 0.0
    context: str = ""

    @classmethod
    def from_api_doc(cls, doc: dict) -> PrimoRecord:
        """Parse a single document from the Primo /pnxs response."""
        pnx = doc.get("pnx", {})
        display = pnx.get("display", {})
        control = pnx.get("control", {})
        addata = pnx.get("addata", {})
        search = pnx.get("search", {})
        delivery = pnx.get("delivery", {})
        # Search docs and direct full-display responses both carry a rich
        # doc-level delivery block (holdings, links, openurl) alongside the
        # classic pnx delivery tokens.
        doc_delivery = doc.get("delivery")
        if not isinstance(doc_delivery, dict):
            doc_delivery = {}

        # Identifiers and DOI. Prefer the structured addata/doi field (clean
        # bare DOIs in CDI records), falling back to DOIs found in
        # display/identifier, which handles both subfield-encoded ($$CDOI)
        # and labelled ("DOI: 10.x") shapes case-insensitively.
        identifiers, ident_doi = _parse_identifiers(_to_list(display.get("identifier")))
        addata_doi = _first_or_empty(addata.get("doi")).strip()
        doi = _clean_doi(addata_doi) if addata_doi else ident_doi

        # Parse creators -- display.creator is often a single semicolon-separated
        # string and carries $$ subfield codes plus trailing relator terms.
        creators = _clean_names(_to_list(display.get("creator")))

        # Subjects -- may be semicolon-separated
        raw_subjects = _to_list(display.get("subject"))
        subjects = []
        for s in raw_subjects:
            for part in _VALUE_SEPARATORS_RE.split(s):
                p = _strip_subfields(part)
                if p:
                    subjects.append(p)

        # Keywords
        raw_keywords = _to_list(display.get("keyword"))
        keywords = []
        for k in raw_keywords:
            for part in _VALUE_SEPARATORS_RE.split(k):
                p = _strip_subfields(part)
                if p:
                    keywords.append(p)

        # Peer review
        lds50 = _to_list(display.get("lds50"))
        peer_reviewed = any("peer_review" in x.lower() for x in lds50)

        # Score
        score_raw = _to_list(control.get("score"))
        try:
            score = float(score_raw[0]) if score_raw else 0.0
        except (ValueError, IndexError):
            score = 0.0

        return cls(
            record_id=_first_or_empty(control.get("recordid")),
            source_id=_first_or_empty(control.get("sourceid")),
            source_system=_first_or_empty(control.get("sourcesystem")),
            title=_strip_subfields(_first_or_empty(display.get("title"))),
            resource_type=_first_or_empty(display.get("type")),
            language=_first_or_empty(display.get("language")),
            creators=creators,
            contributors=_clean_names(_to_list(display.get("contributor"))),
            publisher=_strip_subfields(_first_or_empty(display.get("publisher"))),
            creation_date=_first_or_empty(display.get("creationdate"))
                or _first_or_empty(addata.get("date")),
            source_label=_first_or_empty(display.get("source")),
            description=_first_or_empty(display.get("description"))
                or _first_or_empty(addata.get("abstract")),
            snippet=_first_or_empty(display.get("snippet")),
            subjects=subjects,
            keywords=keywords,
            is_part_of=_strip_subfields(_first_or_empty(display.get("ispartof"))),
            identifiers=identifiers,
            doi=doi,
            isbn=_to_list(addata.get("isbn")),
            issn=_to_list(addata.get("issn")),
            journal_title=_strip_subfields(_first_or_empty(addata.get("jtitle"))),
            volume=_first_or_empty(addata.get("volume")),
            issue=_first_or_empty(addata.get("issue")),
            start_page=_first_or_empty(addata.get("spage")),
            end_page=_first_or_empty(addata.get("epage")),
            peer_reviewed=peer_reviewed,
            ris_type=_first_or_empty(addata.get("ristype")),
            authors_structured=_clean_names(
                _to_list(addata.get("au")), split=False, strip_relators=False
            ),
            additional_authors=_clean_names(
                _to_list(addata.get("addau")), split=False, strip_relators=False
            ),
            fulltext_available=_fulltext_available(delivery.get("fulltext")),
            delivery_category=_first_or_empty(delivery.get("delcategory"))
                or _first_or_empty(doc_delivery.get("deliveryCategory")),
            locations=_parse_locations(doc_delivery),
            access_links=_parse_access_links(doc_delivery),
            openurl=_parse_openurl(doc_delivery),
            score=score,
            context=doc.get("context", ""),
        )

    @property
    def display_authors(self) -> list[str]:
        """Best available author list, in order of preference.

        Primo splits names across several fields: structured authors
        (addata.au) and additional authors/editors (addata.addau) are clean,
        while display.creator/contributor carry subfield noise. Falling back
        through them avoids "Unknown author" when only an edited-book editor
        list (addau) or a contributor is present.
        """
        return (
            self.authors_structured
            or self.additional_authors
            or self.creators
            or self.contributors
        )

    @property
    def year(self) -> str:
        """Four-digit year extracted from creation_date.

        Primo dates come in many shapes -- "2021", "2021-03", "c1996",
        "[2019]" -- so the first run of four digits is taken rather than a
        naive slice (which turned "c1996" into "c199").
        """
        m = re.search(r"\d{4}", self.creation_date)
        return m.group(0) if m else ""


class SearchInfo(BaseModel):
    """Pagination and total count info from a search response."""

    total: int = 0
    total_local: int = 0
    total_pc: int = 0
    first: int = 0
    last: int = 0


class FacetValue(BaseModel):
    """A single facet value with its result count."""

    value: str = ""
    count: int = 0


class Facet(BaseModel):
    """One search facet (e.g. rtype, topic) summarising all matching results."""

    name: str = ""
    values: list[FacetValue] = Field(default_factory=list)

    @classmethod
    def list_from_api_response(cls, data: dict) -> list[Facet]:
        """Parse a Primo /facets response into Facet models.

        Counts arrive as strings ("412"); unparseable counts become 0.
        Facets without a name or without any named values are dropped.
        """
        facets: list[Facet] = []
        for raw in data.get("facets") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            values: list[FacetValue] = []
            for raw_value in raw.get("values") or []:
                if not isinstance(raw_value, dict):
                    continue
                value = str(raw_value.get("value", "")).strip()
                if not value:
                    continue
                try:
                    count = int(str(raw_value.get("count", 0)))
                except ValueError:
                    count = 0
                values.append(FacetValue(value=value, count=count))
            if name and values:
                facets.append(cls(name=name, values=values))
        return facets


class SearchResponse(BaseModel):
    """Parsed Primo search response."""

    info: SearchInfo = Field(default_factory=SearchInfo)
    records: list[PrimoRecord] = Field(default_factory=list)
    # Facets summarising ALL matching results (not just the returned page).
    # Populated from Primo's separate /facets endpoint after the search;
    # empty when facets are unavailable (e.g. local catalogue scope).
    facets: list[Facet] = Field(default_factory=list)

    @property
    def total_results(self) -> int:
        """Compatibility alias for older callers."""
        return self.info.total

    @property
    def total_local_results(self) -> int:
        """Compatibility alias for older callers."""
        return self.info.total_local

    @property
    def total_pc_results(self) -> int:
        """Compatibility alias for older callers."""
        return self.info.total_pc

    @classmethod
    def from_api_response(cls, data: dict) -> SearchResponse:
        """Parse the full /pnxs API response."""
        info_raw = data.get("info", {})
        info = SearchInfo(
            total=info_raw.get("total", 0),
            total_local=info_raw.get("totalResultsLocal", 0),
            total_pc=info_raw.get("totalResultsPC", 0),
            first=info_raw.get("first", 0),
            last=info_raw.get("last", 0),
        )
        records = [
            PrimoRecord.from_api_doc(doc)
            for doc in data.get("docs", [])
        ]
        return cls(info=info, records=records)
