"""Citation formatting for Primo records.

Supports APA 7th, Harvard, Chicago, IEEE, and Vancouver styles.
"""

from __future__ import annotations

import re

from primo_mcp_server.models import PrimoRecord


_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _has_cjk(value: str) -> bool:
    """Return true when a name contains CJK characters."""
    return bool(_CJK_RE.search(value))


def _authors_last_initials(creators: list[str]) -> list[str]:
    """Convert 'Last, First' to 'Last, F.' format."""
    result = []
    for c in creators:
        if _has_cjk(c):
            result.append(c.strip())
            continue
        parts = c.split(",", 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            initials = ". ".join(w[0].upper() for w in first.split() if w) + "."
            result.append(f"{last}, {initials}")
        else:
            result.append(c.strip())
    return result


def _authors_apa(creators: list[str]) -> str:
    """Format authors in APA 7th style."""
    formatted = _authors_last_initials(creators)
    if not formatted:
        return "Unknown author"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    # 20+ authors: first 19, ..., last
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def _authors_harvard(creators: list[str]) -> str:
    """Format authors in Harvard style (same as APA for most cases)."""
    return _authors_apa(creators)


def _authors_chicago(creators: list[str]) -> str:
    """Format authors in Chicago style."""
    formatted = _authors_last_initials(creators)
    if not formatted:
        return "Unknown author"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    if len(formatted) == 3:
        return ", ".join(formatted[:-1]) + ", and " + formatted[-1]
    return formatted[0] + " et al."


def _authors_ieee(creators: list[str]) -> list[str]:
    """Format authors in IEEE style (F. Last)."""
    result = []
    for c in creators:
        if _has_cjk(c):
            result.append(c.strip())
            continue
        parts = c.split(",", 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            initials = ". ".join(w[0].upper() for w in first.split() if w) + "."
            result.append(f"{initials} {last}")
        else:
            result.append(c.strip())
    return result


def _authors_vancouver(creators: list[str]) -> str:
    """Format authors in Vancouver style."""
    formatted = []
    for c in creators:
        if _has_cjk(c):
            formatted.append(c.strip())
            continue
        parts = c.split(",", 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            initials = "".join(w[0].upper() for w in first.split() if w)
            formatted.append(f"{last} {initials}")
        else:
            formatted.append(c.strip())
    if not formatted:
        return "Unknown author"
    if len(formatted) <= 6:
        return ", ".join(formatted)
    return ", ".join(formatted[:6]) + ", et al"


def _year(record: PrimoRecord) -> str:
    """Extract 4-digit year from creation date."""
    return record.year or "n.d."


def _cite_article_apa(r: PrimoRecord) -> str:
    """APA 7 article citation."""
    authors = _authors_apa(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    journal = r.journal_title

    parts = [f"{authors} ({year}). {title}."]
    if journal:
        vol_info = f"*{journal}*"
        if r.volume:
            vol_info += f", *{r.volume}*"
        if r.issue:
            vol_info += f"({r.issue})"
        if r.start_page:
            vol_info += f", {r.start_page}"
            if r.end_page:
                vol_info += f"-{r.end_page}"
        parts.append(f"{vol_info}.")
    if r.doi:
        parts.append(f"https://doi.org/{r.doi}")
    return " ".join(parts)


def _cite_book_apa(r: PrimoRecord) -> str:
    """APA 7 book citation."""
    authors = _authors_apa(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    publisher = r.publisher or ""

    parts = [f"{authors} ({year}). *{title}*."]
    if publisher:
        parts.append(f"{publisher}.")
    if r.doi:
        parts.append(f"https://doi.org/{r.doi}")
    return " ".join(parts)


def _cite_article_harvard(r: PrimoRecord) -> str:
    """Harvard article citation."""
    authors = _authors_harvard(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    journal = r.journal_title

    parts = [f"{authors} ({year}) '{title}',"]
    if journal:
        vol_info = f"*{journal}*"
        if r.volume:
            vol_info += f", vol. {r.volume}"
        if r.issue:
            vol_info += f", no. {r.issue}"
        if r.start_page:
            vol_info += f", pp. {r.start_page}"
            if r.end_page:
                vol_info += f"-{r.end_page}"
        parts.append(f"{vol_info}.")
    if r.doi:
        parts.append(f"https://doi.org/{r.doi}")
    return " ".join(parts)


def _cite_book_harvard(r: PrimoRecord) -> str:
    """Harvard book citation."""
    authors = _authors_harvard(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    publisher = r.publisher or ""

    parts = [f"{authors} ({year}) *{title}*,"]
    if publisher:
        parts.append(f"{publisher}.")
    return " ".join(parts)


def _cite_article_chicago(r: PrimoRecord) -> str:
    """Chicago article citation."""
    authors = _authors_chicago(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    journal = r.journal_title

    parts = [f'{authors}. "{title}."']
    if journal:
        vol_info = f"*{journal}*"
        if r.volume:
            vol_info += f" {r.volume}"
        if r.issue:
            vol_info += f", no. {r.issue}"
        parts.append(f"{vol_info} ({year})")
        if r.start_page:
            start = r.start_page
            if r.end_page:
                start += f"-{r.end_page}"
            parts[-1] += f": {start}"
        parts[-1] += "."
    else:
        # Without a journal title the year was previously dropped entirely,
        # since it was only emitted inside the journal block.
        parts.append(f"{year}.")
    if r.doi:
        parts.append(f"https://doi.org/{r.doi}")
    return " ".join(parts)


def _cite_book_chicago(r: PrimoRecord) -> str:
    """Chicago book citation."""
    authors = _authors_chicago(r.display_authors)
    year = _year(r)
    title = r.title.rstrip(".")
    publisher = r.publisher or ""

    parts = [f"{authors}. *{title}*."]
    if publisher:
        parts.append(f"{publisher}, {year}.")
    else:
        parts.append(f"{year}.")
    return " ".join(parts)


def _cite_article_ieee(r: PrimoRecord) -> str:
    """IEEE article citation."""
    authors_list = _authors_ieee(r.display_authors)
    if not authors_list:
        authors_str = "Unknown author"
    elif len(authors_list) == 1:
        authors_str = authors_list[0]
    elif len(authors_list) == 2:
        authors_str = f"{authors_list[0]} and {authors_list[1]}"
    elif len(authors_list) == 3:
        authors_str = ", ".join(authors_list[:-1]) + ", and " + authors_list[-1]
    else:
        authors_str = authors_list[0] + " et al."

    title = r.title.rstrip(".")
    journal = r.journal_title or ""
    year = _year(r)

    parts = [f'{authors_str}, "{title},"']
    if journal:
        vol_info = f"*{journal}*"
        if r.volume:
            vol_info += f", vol. {r.volume}"
        if r.issue:
            vol_info += f", no. {r.issue}"
        if r.start_page:
            vol_info += f", pp. {r.start_page}"
            if r.end_page:
                vol_info += f"-{r.end_page}"
        parts.append(f"{vol_info}, {year}.")
    else:
        # Without a journal title the year was previously dropped entirely,
        # since it was only emitted inside the journal block.
        parts.append(f"{year}.")
    if r.doi:
        parts.append(f"doi: {r.doi}.")
    return " ".join(parts)


def _cite_book_ieee(r: PrimoRecord) -> str:
    """IEEE book citation."""
    authors_list = _authors_ieee(r.display_authors)
    if not authors_list:
        authors_str = "Unknown author"
    else:
        authors_str = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors_str += " et al."

    title = r.title.rstrip(".")
    publisher = r.publisher or ""
    year = _year(r)

    parts = [f"{authors_str}, *{title}*."]
    if publisher:
        parts.append(f"{publisher}, {year}.")
    else:
        parts.append(f"{year}.")
    return " ".join(parts)


def _cite_article_vancouver(r: PrimoRecord) -> str:
    """Vancouver article citation."""
    authors = _authors_vancouver(r.display_authors)
    title = r.title.rstrip(".")
    journal = r.journal_title or ""
    year = _year(r)

    parts = [f"{authors}. {title}."]
    if journal:
        vol_info = f"{journal}. {year}"
        if r.volume:
            vol_info += f";{r.volume}"
        if r.issue:
            vol_info += f"({r.issue})"
        if r.start_page:
            vol_info += f":{r.start_page}"
            if r.end_page:
                vol_info += f"-{r.end_page}"
        parts.append(f"{vol_info}.")
    else:
        # Without a journal title the year was previously dropped entirely,
        # since it was only emitted inside the journal block.
        parts.append(f"{year}.")
    if r.doi:
        parts.append(f"doi:{r.doi}")
    return " ".join(parts)


def _cite_book_vancouver(r: PrimoRecord) -> str:
    """Vancouver book citation."""
    authors = _authors_vancouver(r.display_authors)
    title = r.title.rstrip(".")
    publisher = r.publisher or ""
    year = _year(r)

    parts = [f"{authors}. {title}."]
    if publisher:
        parts.append(f"{publisher}; {year}.")
    else:
        parts.append(f"{year}.")
    return " ".join(parts)


_STYLE_MAP = {
    "apa7": {"article": _cite_article_apa, "book": _cite_book_apa},
    "harvard": {"article": _cite_article_harvard, "book": _cite_book_harvard},
    "chicago": {"article": _cite_article_chicago, "book": _cite_book_chicago},
    "ieee": {"article": _cite_article_ieee, "book": _cite_book_ieee},
    "vancouver": {"article": _cite_article_vancouver, "book": _cite_book_vancouver},
}

CITATION_STYLES = tuple(_STYLE_MAP)


def invalid_style_message(style: str) -> str | None:
    """Caller-facing error for an unsupported style, or None when valid."""
    if style.strip().lower() in _STYLE_MAP:
        return None
    return (
        f'Invalid citation style "{style}". '
        f"Use one of: {', '.join(sorted(CITATION_STYLES))}"
    )


def _record_type_key(resource_type: str) -> str:
    """Normalise Primo record type labels for citation routing."""
    return resource_type.strip().lower().replace("-", "_")


def format_citation(record: PrimoRecord, style: str = "apa7") -> str:
    """Format a citation for a record in the specified style."""
    style = style.strip().lower()
    style_funcs = _STYLE_MAP.get(style, _STYLE_MAP["apa7"])

    # Determine if article or book
    rtype = _record_type_key(record.resource_type)
    if rtype in (
        "article",
        "articles",
        "review",
        "reviews",
        "newspaper_article",
        "newspaper_articles",
    ):
        func = style_funcs["article"]
    else:
        func = style_funcs["book"]

    return func(record)
