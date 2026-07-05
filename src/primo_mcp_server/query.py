"""Validation and normalisation for Primo query parameters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


_SCOPE_ALIASES = {
    "catalogue": "catalogue",
    "catalog": "catalogue",
    "local": "catalogue",
    "myinstitution": "catalogue",
    "my_institution": "catalogue",
    "everything": "everything",
    "all": "everything",
    "combined": "everything",
    "myinst_and_ci": "everything",
    "pci": "everything",
    "books_videos": "books_videos",
    "booksvideos": "books_videos",
    "booksandvideos": "books_videos",
    "books/videos": "books_videos",
    "books_&_videos": "books_videos",
    "books_and_videos": "books_videos",
}

_FIELD_ALIASES = {
    "any": "any",
    "title": "title",
    "creator": "creator",
    "author": "creator",
    "authors": "creator",
    "subject": "sub",
    "subjects": "sub",
    "sub": "sub",
    "isbn": "isbn",
    "issn": "issn",
    "oclc": "oclcnum",
    "oclcnum": "oclcnum",
}

_SORT_ALIASES = {
    "rank": "rank",
    "relevance": "rank",
    "date": "date",
    "newest": "date",
    "title": "title",
}

_RESOURCE_TYPE_ALIASES = {
    "article": "articles",
    "articles": "articles",
    "book": "books",
    "books": "books",
    "journal": "journals",
    "journals": "journals",
    "database": "databases",
    "databases": "databases",
    "video": "videos",
    "videos": "videos",
    "dissertation": "dissertations",
    "dissertations": "dissertations",
    "thesis": "dissertations",
    "conference_proceeding": "conference_proceedings",
    "conference_proceedings": "conference_proceedings",
    "conference proceeding": "conference_proceedings",
    "conference proceedings": "conference_proceedings",
}

# Caller-friendly aliases for Primo facet field names, used by the generic
# facet_filters/facet_exclusions parameters. Keys not listed here pass
# through unchanged (after "facet_" prefix stripping), so institution-local
# facets (local1..local50, lds*) remain reachable.
_FACET_NAME_ALIASES = {
    "resource_type": "rtype",
    "type": "rtype",
    "subject": "topic",
    "subjects": "topic",
    "author": "creator",
    "journal": "jtitle",
    "journal_title": "jtitle",
    "language": "lang",
    "availability": "tlevel",
    "creation_date": "searchcreationdate",
    "creationdate": "searchcreationdate",
    "collection": "domain",
}

_FACET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_OPERATOR_ALIASES = {
    "contains": "contains",
    "exact": "exact",
    "equals": "exact",
    "is": "exact",
    "begins_with": "begins_with",
    "beginswith": "begins_with",
    "starts_with": "begins_with",
    "startswith": "begins_with",
}

_CONNECTOR_ALIASES = {
    "and": "AND",
    "or": "OR",
    "not": "NOT",
}

_YEAR_RE = re.compile(r"^\d{4}$")
# Commas and semicolons are Primo's clause syntax separators; they cannot
# appear inside a clause value.
_CLAUSE_SEPARATOR_CHARS_RE = re.compile(r"[,;]+")


def _key(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _valid_options(aliases: dict[str, str]) -> str:
    return ", ".join(sorted(set(aliases.values())))


def _normalise_alias(value: str, aliases: dict[str, str], label: str) -> str:
    key = _key(value)
    try:
        return aliases[key]
    except KeyError as e:
        raise ValueError(
            f'Invalid {label} "{value}". Use one of: {_valid_options(aliases)}.'
        ) from e


def normalise_scope(scope: str) -> str:
    """Resolve caller-friendly scope aliases to canonical scope names."""
    return _normalise_alias(scope, _SCOPE_ALIASES, "scope")


def normalise_search_field(field: str) -> str:
    """Resolve caller-friendly search field aliases to Primo field names."""
    return _normalise_alias(field, _FIELD_ALIASES, "search field")


def normalise_sort_by(sort_by: str) -> str:
    """Resolve caller-friendly sort aliases to Primo sort names."""
    return _normalise_alias(sort_by, _SORT_ALIASES, "sort_by")


def normalise_resource_type(resource_type: str | None) -> str | None:
    """Resolve caller-friendly resource type aliases to Primo facet values."""
    if resource_type is None:
        return None
    return _normalise_alias(resource_type, _RESOURCE_TYPE_ALIASES, "resource_type")


def normalise_facet_name(name: str) -> str:
    """Resolve a caller-friendly facet name to Primo's facet field name.

    Accepts bare names ("topic"), aliased names ("subject"), and fully
    prefixed names ("facet_topic"); returns the bare Primo name without the
    "facet_" prefix.
    """
    key = _key(name)
    if key.startswith("facet_"):
        key = key[len("facet_"):]
    key = _FACET_NAME_ALIASES.get(key, key)
    if not _FACET_NAME_RE.match(key):
        raise ValueError(
            f'Invalid facet name "{name}". Use a Primo facet name such as: '
            "rtype, topic, creator, jtitle, lang, tlevel, library, domain."
        )
    return key


def compile_facet_filters(
    filters: Mapping[str, str] | None, label: str = "facet_filters"
) -> list[str]:
    """Compile a {facet: value} mapping into Primo qInclude/qExclude parts.

    Each part is ``facet_<name>,exact,<value>``, the shape Primo's qInclude
    and qExclude parameters share (parts joined with ``|,|``). Values are
    matched exactly against facet values as Primo reports them, e.g. in the
    facet summary of a previous search.
    """
    if not filters:
        return []
    parts: list[str] = []
    for raw_name, raw_value in filters.items():
        name = normalise_facet_name(str(raw_name))
        value = str(raw_value).strip()
        if not value:
            raise ValueError(f'{label} value for "{raw_name}" is empty.')
        parts.append(f"facet_{name},exact,{value}")
    return parts


def normalise_operator(operator: str) -> str:
    """Resolve caller-friendly operator aliases to Primo query operators."""
    return _normalise_alias(operator, _OPERATOR_ALIASES, "operator")


def normalise_connector(connector: str) -> str:
    """Resolve boolean connector aliases to Primo's AND/OR/NOT."""
    return _normalise_alias(connector, _CONNECTOR_ALIASES, "connector")


class QueryClause(BaseModel):
    """One clause of a compound Primo query.

    ``connector`` joins this clause to the NEXT one and is ignored on the
    final clause.
    """

    value: str
    field: str = "any"
    operator: str = "contains"
    connector: str = "AND"


def _clause_fields(clause: Any, position: int) -> dict:
    if isinstance(clause, QueryClause):
        return clause.model_dump()
    if isinstance(clause, Mapping):
        return dict(clause)
    raise ValueError(
        f"Clause {position} must be an object with a value and optional "
        "field, operator, and connector."
    )


def query_clause_parts(clauses: Sequence[Any]) -> list[str]:
    """Compile structured clauses into Primo clause strings.

    Each part is ``field,operator,value`` with the boolean connector to the
    next clause appended (``field,operator,value,AND``) on every clause but
    the last -- the shape shared by the Primo API ``q`` parameter (parts
    joined with ``;``) and the Primo UI advanced-search ``query`` parameters
    (one part each).

    Raises ValueError for empty clause lists, empty values, or unknown
    field/operator/connector names.
    """
    if not clauses:
        raise ValueError("clauses must contain at least one clause.")

    parts: list[str] = []
    last = len(clauses) - 1
    for i, raw in enumerate(clauses):
        clause = _clause_fields(raw, i + 1)
        # Separator characters would splice into Primo's clause syntax.
        value = _CLAUSE_SEPARATOR_CHARS_RE.sub(" ", str(clause.get("value") or ""))
        value = " ".join(value.split())
        if not value:
            raise ValueError(f"Clause {i + 1} has an empty value.")
        field = normalise_search_field(str(clause.get("field") or "any"))
        operator = normalise_operator(str(clause.get("operator") or "contains"))
        part = f"{field},{operator},{value}"
        if i != last:
            part += f",{normalise_connector(str(clause.get('connector') or 'AND'))}"
        parts.append(part)
    return parts


def compile_query_clauses(clauses: Sequence[Any]) -> str:
    """Compile structured clauses into the Primo API multi-clause ``q``."""
    return ";".join(query_clause_parts(clauses))


def normalise_year(value: str | None, label: str) -> str | None:
    """Validate and strip a four-digit year parameter."""
    if value is None:
        return None
    year = value.strip()
    if not _YEAR_RE.match(year):
        raise ValueError(f'Invalid {label} "{value}". Use YYYY format, e.g. "2020".')
    return year


def normalise_date_range(
    date_from: str | None,
    date_to: str | None,
) -> tuple[str | None, str | None]:
    """Validate year filters and return stripped values."""
    start = normalise_year(date_from, "date_from")
    end = normalise_year(date_to, "date_to")
    if end and not start:
        raise ValueError("date_to requires date_from.")
    if start and end and int(end) < int(start):
        raise ValueError("date_to must be greater than or equal to date_from.")
    return start, end


def date_range_facet_value(date_from: str | None, date_to: str | None) -> str | None:
    """Return Primo's documented creation-date range facet value."""
    start, end = normalise_date_range(date_from, date_to)
    if not start:
        return None
    return f"[{start} TO {end or start}]"
