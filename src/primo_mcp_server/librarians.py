"""Librarian recommendation models and deterministic matching."""

from __future__ import annotations

import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import snowballstemmer
from pydantic import BaseModel, Field, ValidationError

from primo_mcp_server.models import PrimoRecord

logger = logging.getLogger(__name__)


_MAX_RECOMMENDATIONS = 3
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SECTION_HEADING = "## Recommended librarian help:"
_RECOMMENDATION_FOOTER = (
    "Recommendations are limited to configured librarian profiles; "
    "do not invent or substitute names."
)
_UNCONFIGURED = "Not configured"
_NOISY_METADATA_FIELDS = {"description", "source"}
_HIGH_SIGNAL_METADATA_FIELDS = {"subjects", "keywords"}
_GENERIC_METADATA_TERMS = {
    "analysis",
    "data",
    "policy",
    "research",
    "social science",
    "support",
}


class LibrarianProfile(BaseModel):
    """A configured librarian or librarian team profile."""

    id: str
    name: str
    title: str = ""
    email: str = ""
    url: str = ""
    subjects: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    best_for: list[str] = Field(default_factory=list)
    schools: list[str] = Field(default_factory=list)
    resource_types: list[str] = Field(default_factory=list)
    # Curator deny-list: if any of these terms appears in the user's query,
    # this librarian is never recommended (keyword or semantic path). Lets an
    # institution patch an observed misrouting without retuning weights.
    excludes: list[str] = Field(default_factory=list)
    notes: str = ""


class LibrarianDirectory(BaseModel):
    """External librarian directory loaded from JSON."""

    librarians: list[LibrarianProfile] = Field(default_factory=list)


class LibrarianMatch(BaseModel):
    """A validated librarian recommendation."""

    librarian: LibrarianProfile
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    evidence_fields: list[str] = Field(default_factory=list)


def _configuration_message(path: str | None = None, detail: str | None = None) -> str:
    location = (
        f'PRIMO_LIBRARIANS_FILE is set to "{path}". '
        if path
        else "Set PRIMO_LIBRARIANS_FILE to the path of a JSON file. "
    )
    suffix = f" {detail}" if detail else ""
    return (
        "Librarian recommendations are not configured. "
        + location
        + 'Expected shape: {"librarians": [{"id": "...", "name": "..."}]}.'
        + suffix
    )


def load_librarian_directory(
    path: str | Path | None,
) -> tuple[LibrarianDirectory | None, str | None]:
    """Load an external JSON librarian directory.

    Returns (directory, message). The message is populated when the directory
    cannot be loaded and is intended for MCP-facing guidance.
    """
    if path is None or str(path).strip() == "":
        return None, _configuration_message()

    resolved = Path(path).expanduser()
    try:
        with resolved.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, _configuration_message(
            str(resolved), "The file does not exist."
        )
    except PermissionError:
        return None, _configuration_message(
            str(resolved), "The file is not readable."
        )
    except json.JSONDecodeError as e:
        return None, _configuration_message(
            str(resolved), f"Invalid JSON at line {e.lineno}, column {e.colno}."
        )
    except OSError as e:
        return None, _configuration_message(str(resolved), str(e))

    try:
        directory = LibrarianDirectory.model_validate(data)
    except ValidationError as e:
        return None, _configuration_message(
            str(resolved), f"Profile validation failed: {e.errors()[0]['msg']}."
        )

    if not directory.librarians:
        return None, _configuration_message(
            str(resolved), "The directory contains no librarians."
        )

    duplicates = _duplicate_ids(directory)
    if duplicates:
        return None, _configuration_message(
            str(resolved),
            f"Duplicate librarian id(s): {', '.join(duplicates)}. "
            "Each id must be unique.",
        )

    _warn_about_filler_terms(directory)
    return directory, None


def _warn_about_filler_terms(directory: LibrarianDirectory) -> None:
    """Log profiles listing filler terms so curators can clean the source.

    Filler terms are silently ignored during scoring (see _FILLER_TERMS), so
    listing them is harmless but pointless; the warning surfaces the wasted
    entries once per directory (re)load without failing the load.
    """
    for librarian in directory.librarians:
        fillers = _unique(
            term
            for term in _librarian_terms(librarian)
            if _is_filler_term(term)
        )
        if fillers:
            logger.warning(
                "Librarian profile %r lists filler term(s) that never "
                "match: %s",
                librarian.id,
                ", ".join(fillers),
            )


def _duplicate_ids(directory: LibrarianDirectory) -> list[str]:
    """Ids that appear more than once (case-insensitively), in file order.

    Unenforced duplicates would silently collide downstream: the embedding
    cache and the keyword/semantic de-dup logic both key results by id, so a
    repeated id makes one profile's data overwrite or hide another's with no
    error anywhere.
    """
    seen: dict[str, int] = {}
    for librarian in directory.librarians:
        key = librarian.id.casefold()
        seen[key] = seen.get(key, 0) + 1
    return _unique(
        librarian.id for librarian in directory.librarians if seen[librarian.id.casefold()] > 1
    )


class _DirectoryCacheEntry(NamedTuple):
    mtime_ns: int
    directory: LibrarianDirectory
    specificity: dict[str, float]


# Keyed by resolved absolute path. Every primo_search / primo_recommend_librarians
# call needs the directory and its IDF specificity weights; re-reading and
# re-parsing the JSON file and recomputing IDF on every request is wasted
# work since the directory changes rarely. A file's mtime is cheap to check
# (a single stat syscall) and lets repeat calls skip the reparse entirely.
_directory_cache: dict[str, _DirectoryCacheEntry] = {}


def load_librarian_directory_cached(
    path: str | Path | None,
) -> tuple[LibrarianDirectory | None, str | None, dict[str, float]]:
    """Load a directory, reusing a cached parse when the file is unchanged.

    Returns (directory, message, specificity). ``specificity`` is the IDF
    weight map from ``_term_specificity``, computed once per cache refresh
    rather than once per call. Failure results (missing/invalid file) are
    never cached, so fixing the file takes effect on the very next call.
    """
    if path is None or str(path).strip() == "":
        return None, _configuration_message(), {}

    resolved = Path(path).expanduser()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        # Let load_librarian_directory produce the precise error message
        # (not-found vs permission vs other OS error).
        directory, message = load_librarian_directory(path)
        return directory, message, {}

    cache_key = str(resolved)
    cached = _directory_cache.get(cache_key)
    if cached is not None and cached.mtime_ns == mtime_ns:
        return cached.directory, None, cached.specificity

    directory, message = load_librarian_directory(path)
    if directory is None:
        _directory_cache.pop(cache_key, None)
        return None, message, {}

    specificity = _term_specificity(directory)
    _directory_cache[cache_key] = _DirectoryCacheEntry(mtime_ns, directory, specificity)
    return directory, None, specificity


_IDENTIFIER_PATTERNS = (
    re.compile(r"\b10\.\d{4,9}/\S+"),  # DOI
    re.compile(r"doi\.org/", re.IGNORECASE),
    re.compile(r"\bdoi\s*:", re.IGNORECASE),
    re.compile(r"\bisbn\s*:?\s*[\d\- ]{9,17}[\dxX]\b", re.IGNORECASE),
    re.compile(r"\bissn\s*:?\s*\d{4}-?\d{3}[\dxX]\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{3}[\dxX]\b"),  # bare ISSN
    re.compile(r"\balma\d{6,}\b", re.IGNORECASE),  # Alma MMS record id
    re.compile(r"\bcdi_\w+", re.IGNORECASE),  # CDI record id
)


def looks_like_identifier(query: str) -> bool:
    """True when the query is a record identifier rather than a topic.

    Embedding a DOI or ISBN produces noise, and keyword-matching one against
    subject profiles is meaningless, so librarian recommendations skip
    identifier-shaped queries entirely (both matching paths).
    """
    text = query.strip()
    if not text:
        return False
    # A bare ISBN-10/13 (possibly hyphenated/spaced) as the whole query.
    compact = re.sub(r"[\s\-]", "", text)
    if re.fullmatch(r"\d{9,13}|\d{9,12}[xX]", compact):
        return True
    return any(pattern.search(text) for pattern in _IDENTIFIER_PATTERNS)


_SNOWBALL = snowballstemmer.stemmer("english")

# en-GB -> en-US suffix fold applied before stemming. Snowball (like Porter)
# only recognises the -ize derivational family, so without this fold the
# en-AU spellings the profiles use ("anonymisation", "digitised") never align
# with each other or with the en-US spellings CDI record metadata uses.
# Ordered longest-first; the first matching suffix wins.
_BRITISH_SUFFIX_FOLDS = (
    ("isations", "izations"),
    ("isation", "ization"),
    ("ising", "izing"),
    ("isers", "izers"),
    ("ised", "ized"),
    ("iser", "izer"),
    ("ises", "izes"),
    ("ise", "ize"),
    ("ysing", "yzing"),
    ("ysed", "yzed"),
    ("yses", "yzes"),
    ("yse", "yze"),
    ("oguing", "oging"),
    ("ogued", "oged"),
    ("ogues", "ogs"),
    ("ogue", "og"),
    ("ourally", "orally"),
    ("oural", "oral"),
    ("ouring", "oring"),
    ("oured", "ored"),
    ("ours", "ors"),
    ("our", "or"),
)

# Word stems where the -our fold would corrupt the token in any inflection:
# "detour"/"detoured" are not en-GB spellings of "detor"/"detored". Short
# -our words (hour, four, tour, ...) are already protected by the length
# guard in _fold_british.
_OUR_FOLD_EXCEPTION_STEMS = (
    "detour",
    "contour",
    "velour",
    "troubadour",
    "paramour",
)


def _fold_british(token: str) -> str:
    """Fold en-GB derivational suffixes to their en-US forms."""
    if len(token) < 6 or token.startswith(_OUR_FOLD_EXCEPTION_STEMS):
        return token
    for british, american in _BRITISH_SUFFIX_FOLDS:
        if token.endswith(british):
            return token[: -len(british)] + american
    return token


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    """Reduce a token to its Snowball (Porter2) stem, en-GB folded first.

    Snowball aligns whole derivational families ("anonymising" /
    "anonymisation" / "anonymized" all reduce to "anonym"), not just the
    plural and -ing/-ed inflections the previous hand-rolled stripper knew.
    It is applied symmetrically to both profile terms and record/query text,
    so even an over-aggressive stem still matches as long as both sides
    reduce alike, and the IDF specificity map keeps collapsed stems from
    gaining unearned weight. Short tokens (<= 3 chars) are left untouched to
    avoid mangling acronyms such as "INK", "RDR", or "ESG"; the cache keeps
    the heavier algorithm as cheap as the old suffix stripper in practice.
    """
    if len(token) <= 3:
        return token
    return _SNOWBALL.stemWord(_fold_british(token))


@lru_cache(maxsize=4096)
def _normalise_text(value: str) -> str:
    return " ".join(_stem(token) for token in _TOKEN_RE.findall(value.casefold()))


# Universal filler words carry no routing signal from ANY evidence field,
# query included: "I need research support" says nothing about which subject
# librarian to consult, yet real profiles do list words like "research" and
# phrases like "Research support" as terms. Unlike _GENERIC_METADATA_TERMS
# (which are merely too weak to trust from noisy record metadata but still
# meaningful in a direct query, e.g. "policy"), a term composed entirely of
# filler words never scores anywhere. Stored normalised so singular and
# plural profile spellings match.
_FILLER_TERMS = frozenset(
    _normalise_text(term)
    for term in (
        "research",
        "support",
        "help",
        "consultation",
        "consultations",
        "service",
        "services",
        "resource",
        "resources",
        "assistance",
        "information",
        "librarian",
        "library",
        "question",
        "questions",
        "guide",
        "guides",
        "general",
    )
)


def _is_filler_term(term: str) -> bool:
    """True when every token of the term is a filler word.

    Catches both bare fillers ("research") and filler-only phrases real
    profiles list ("Research support", "Research consultation"), while
    keeping phrases where any token adds signal ("legal research",
    "research data management").
    """
    tokens = _normalise_text(term).split()
    return bool(tokens) and all(token in _FILLER_TERMS for token in tokens)


# Function words and question scaffolding, used only to measure how much of
# a query a matched term actually explains -- never for matching itself.
# Normalised so stemming stays symmetric with query text.
_STOPWORDS = frozenset(
    _normalise_text(word)
    for word in (
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "for",
        "to", "with", "about", "from", "by", "into", "using",
        "is", "are", "was", "be", "do", "does", "did",
        "can", "could", "would", "should", "will",
        "i", "me", "my", "we", "our", "you", "your", "it",
        "this", "that", "these", "those",
        "how", "what", "which", "who", "where", "when",
        "need", "needs", "want", "wants", "looking", "look",
        "find", "get", "please", "some", "any",
    )
)


def _content_token_count(text: str) -> int:
    """Count query tokens that carry topical signal (non-stopword, non-filler)."""
    return sum(
        1
        for token in _normalise_text(text).split()
        if token not in _STOPWORDS and token not in _FILLER_TERMS
    )


# Query-evidence dampening floor: even a term covering a sliver of a long
# query keeps 40% of its weight, so a strong subject/alias hit still counts
# but needs corroboration (metadata or specificity) to clear min_score.
_MIN_QUERY_COVERAGE_FACTOR = 0.4


def _query_coverage_factor(term: str, query_content_tokens: int) -> float:
    """Scale query evidence by how much of the query the term explains.

    A one-word term matched inside a twelve-word question is far weaker
    evidence than the same term as the entire query, but both previously
    earned identical query weight. sqrt softens the penalty so specific
    multi-word terms in medium queries are barely affected.
    """
    if query_content_tokens <= 0:
        return 1.0
    term_tokens = len(_normalise_text(term).split())
    coverage = min(1.0, term_tokens / query_content_tokens)
    return max(_MIN_QUERY_COVERAGE_FACTOR, math.sqrt(coverage))


def _contains_term(text: str, term: str, *, allow_reverse: bool = False) -> bool:
    term_norm = _normalise_text(term)
    if len(term_norm) < 2:
        return False
    text_norm = _normalise_text(text)
    if not text_norm:
        return False
    if " " in term_norm:
        if f" {term_norm} " in f" {text_norm} ":
            return True
        if not allow_reverse:
            return False
        # Reverse containment is allowed only for the user's query: the query
        # may be a shorter phrase contained in a longer configured profile term
        # (e.g. "deep research" against "AI deep research"). Applying this to
        # record metadata creates false positives such as "Information services"
        # matching "legal information services".
        if not (" " in text_norm and f" {text_norm} " in f" {term_norm} "):
            return False
        # Guard against qualifier-stripping: the sub-phrase must carry
        # signal of its own ("information services" inside "legal
        # information services" drops exactly the word that made the term
        # specific) and must cover at least half the term's tokens (so
        # "data analysis" cannot claim a long specialised phrase).
        text_tokens = text_norm.split()
        if all(
            token in _FILLER_TERMS or token in _STOPWORDS
            for token in text_tokens
        ):
            return False
        return 2 * len(text_tokens) >= len(term_norm.split())
    return term_norm in set(text_norm.split())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _format_list(values: list[str], *, empty: str = _UNCONFIGURED) -> str:
    return ", ".join(values) if values else empty


def _format_human_list(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    if len(values) == 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _format_best_for_sentence(values: list[str]) -> str:
    if not values:
        return "No configured areas of support."
    return f"Consult for {_format_human_list(values)}."


def _format_similar_profile_topics(librarian: LibrarianProfile) -> str:
    topics = _unique(
        [
            *librarian.best_for,
            *librarian.subjects,
            *librarian.aliases,
            *librarian.keywords,
        ]
    )[:5]
    return _format_list(topics)


def _profile_link_target(librarian: LibrarianProfile) -> str:
    """Return a link target so displayed names are always Markdown links."""
    if librarian.url.strip():
        return librarian.url.strip()
    if librarian.email.strip():
        return f"mailto:{librarian.email.strip()}"
    return "#"


def _format_linked_name(librarian: LibrarianProfile) -> str:
    return f"[{librarian.name}]({_profile_link_target(librarian)})"


def _format_match_evidence(match: LibrarianMatch) -> str:
    return (
        f"matched terms: {_format_list(match.matched_terms, empty='none')}; "
        f"evidence fields: {_format_list(match.evidence_fields, empty='none')}"
    )


def _record_texts(record: PrimoRecord) -> dict[str, list[str]]:
    return {
        "title": [record.title],
        "subjects": record.subjects,
        "keywords": record.keywords,
        "description": [record.description, record.snippet],
        "resource_type": [record.resource_type],
        "source": [
            record.source_label,
            record.publisher,
            record.journal_title,
            record.is_part_of,
        ],
    }


def _record_field_texts(records: list[PrimoRecord]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {
        "title": [],
        "subjects": [],
        "keywords": [],
        "description": [],
        "resource_type": [],
        "source": [],
    }
    for record in records:
        for field, values in _record_texts(record).items():
            fields[field].extend(values)
    return {name: _unique(values) for name, values in fields.items()}


def _librarian_terms(librarian: LibrarianProfile) -> Iterable[str]:
    """Yield every term a librarian lists across all profile fields."""
    yield from librarian.subjects
    yield from librarian.aliases
    yield from librarian.keywords
    yield from librarian.best_for
    yield from librarian.schools
    yield from librarian.resource_types


# Ceiling on the IDF multiplier. Being unique within the directory is not the
# same as being topically specific: without a cap, one idiosyncratic word on a
# single profile is amplified without bound as the directory grows
# (1 + ln(n/1) is ~4.4x at 30 profiles), letting an accidental term dominate.
_MAX_SPECIFICITY = 3.0


def _term_specificity(directory: LibrarianDirectory) -> dict[str, float]:
    """Inverse-document-frequency weight per normalised term.

    Terms that nearly every librarian lists (e.g. broad "research") carry
    little routing signal, so they get a multiplier near 1.0. Terms unique to
    one librarian are discriminative and are amplified -- this is what lets a
    short, specialised profile (e.g. "Digital Preservation") outrank a long,
    keyword-padded one when a distinctive term actually matches.
    """
    n = len(directory.librarians)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for librarian in directory.librarians:
        seen: set[str] = set()
        for term in _librarian_terms(librarian):
            norm = _normalise_text(term)
            if norm and norm not in seen:
                seen.add(norm)
                doc_freq[norm] = doc_freq.get(norm, 0) + 1
    return {
        term: min(_MAX_SPECIFICITY, 1.0 + math.log(n / df))
        for term, df in doc_freq.items()
    }


def _score_term(
    term: str,
    texts_by_field: dict[str, list[str]],
    weights: dict[str, float],
    query_content_tokens: int = 0,
) -> tuple[float, list[str]]:
    if _is_filler_term(term):
        return 0.0, []
    score = 0.0
    evidence_fields: list[str] = []
    is_generic = _normalise_text(term) in _GENERIC_METADATA_TERMS
    for field, texts in texts_by_field.items():
        if is_generic and field in _NOISY_METADATA_FIELDS:
            continue
        weight = weights.get(field, 0.0)
        if weight <= 0:
            continue
        if any(
            _contains_term(text, term, allow_reverse=field == "query")
            for text in texts
        ):
            if field == "query":
                weight *= _query_coverage_factor(term, query_content_tokens)
            score += weight
            evidence_fields.append(field)
    return score, evidence_fields


def _is_generic_metadata_only_match(
    matched_terms: list[str], evidence_fields: list[str]
) -> bool:
    if not matched_terms or "query" in evidence_fields:
        return False
    return all(
        _normalise_text(term) in _GENERIC_METADATA_TERMS
        for term in matched_terms
    )


def _is_specific_multi_word_term(term: str) -> bool:
    return (
        " " in _normalise_text(term)
        and _normalise_text(term) not in _GENERIC_METADATA_TERMS
    )


def _metadata_supporting_record_count(
    matched_terms: list[str], records: list[PrimoRecord]
) -> int:
    count = 0
    for record in records:
        fields = _record_texts(record)
        if any(
            any(
                _contains_term(text, term)
                for text in fields[field]
            )
            for term in matched_terms
            for field in _HIGH_SIGNAL_METADATA_FIELDS
        ):
            count += 1
    return count


def _is_strong_metadata_match(
    matched_terms: list[str],
    evidence_fields: list[str],
    records: list[PrimoRecord],
) -> bool:
    if "query" in evidence_fields:
        return True
    if not (_HIGH_SIGNAL_METADATA_FIELDS & set(evidence_fields)):
        return False
    if _metadata_supporting_record_count(matched_terms, records) < 2:
        return False

    unique_terms = _unique(matched_terms)
    if len(unique_terms) >= 2:
        return True
    return any(_is_specific_multi_word_term(term) for term in unique_terms)


def is_excluded(librarian: LibrarianProfile, query: str) -> bool:
    """True when a curator deny-list term appears in the query.

    Checked on both matching paths so an exclusion cannot be resurrected by
    the semantic fallback. Only the query is inspected -- record metadata is
    too incidental to justify suppressing a librarian.
    """
    return any(_contains_term(query, term) for term in librarian.excludes)


def rank_librarians(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None = None,
    *,
    specificity: dict[str, float] | None = None,
) -> list[LibrarianMatch]:
    """Score every configured librarian, best first, with no threshold.

    Candidates must still clear the quality gates (curator deny list,
    generic metadata-only suppression, the strong-metadata requirement).
    The confidence threshold is applied by callers, so below-threshold
    near-misses stay inspectable: a no_match outcome can then show its
    closest candidates with real evidence instead of discarding them.

    ``specificity`` lets callers reuse a precomputed IDF weight map (see
    ``load_librarian_directory_cached``) instead of paying the O(librarians x
    terms) cost on every call; it is recomputed here when omitted.
    """
    records = records or []
    texts_by_field = {"query": [query], **_record_field_texts(records)}
    query_content_tokens = _content_token_count(query)
    if specificity is None:
        specificity = _term_specificity(directory)

    matches: list[LibrarianMatch] = []
    for librarian in directory.librarians:
        if is_excluded(librarian, query):
            continue
        score = 0.0
        matched_terms: list[str] = []
        evidence_fields: list[str] = []
        # Real profiles repeat terms -- within one list, across field groups,
        # or as case/plural variants that normalise identically ("Altmetric"
        # and "altmetrics"). Each concept may earn score only once; the first
        # group in the fixed order below wins for cross-group repeats.
        scored_norms: set[str] = set()

        term_groups: list[tuple[list[str], dict[str, float]]] = [
            (
                librarian.subjects,
                {
                    "query": 7.0,
                    "subjects": 8.0,
                    "keywords": 5.0,
                    "title": 3.0,
                    "description": 3.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.aliases,
                {
                    "query": 8.0,
                    "subjects": 7.0,
                    "keywords": 5.0,
                    "title": 4.0,
                    "description": 4.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.keywords,
                {
                    "query": 4.0,
                    "subjects": 4.0,
                    "keywords": 4.0,
                    "title": 2.0,
                    "description": 2.0,
                    "source": 1.0,
                },
            ),
            (
                librarian.best_for,
                {
                    "query": 8.0,
                    "subjects": 6.0,
                    "keywords": 6.0,
                    "title": 4.0,
                    "description": 4.0,
                    "resource_type": 3.0,
                    "source": 3.0,
                },
            ),
            (
                librarian.schools,
                {
                    "query": 3.0,
                    "subjects": 2.0,
                    "keywords": 2.0,
                    "source": 2.0,
                },
            ),
            (
                librarian.resource_types,
                {
                    "query": 2.0,
                    "resource_type": 4.0,
                },
            ),
        ]

        for terms, weights in term_groups:
            for term in terms:
                norm = _normalise_text(term)
                if norm in scored_norms:
                    continue
                term_score, term_fields = _score_term(
                    term, texts_by_field, weights, query_content_tokens
                )
                if term_score <= 0:
                    continue
                scored_norms.add(norm)
                term_score *= specificity.get(norm, 1.0)
                score += term_score
                matched_terms.append(term)
                evidence_fields.extend(term_fields)

        if _is_generic_metadata_only_match(
            matched_terms, evidence_fields
        ) or not _is_strong_metadata_match(matched_terms, evidence_fields, records):
            continue

        if score > 0:
            matches.append(
                LibrarianMatch(
                    librarian=librarian,
                    score=score,
                    matched_terms=_unique(matched_terms),
                    evidence_fields=_unique(evidence_fields),
                )
            )

    matches.sort(key=lambda match: (-match.score, match.librarian.name.casefold()))
    return matches


def recommend_librarians(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None = None,
    *,
    limit: int = 2,
    min_score: float = 5.0,
    specificity: dict[str, float] | None = None,
) -> list[LibrarianMatch]:
    """Rank configured librarians against a query and Primo record metadata.

    Only candidates at or above ``min_score`` are returned; use
    ``rank_librarians`` to also see below-threshold near-misses.
    """
    candidates = rank_librarians(directory, query, records, specificity=specificity)
    capped_limit = min(max(1, limit), _MAX_RECOMMENDATIONS)
    return [match for match in candidates if match.score >= min_score][:capped_limit]


# Subjects shown per profile in the directory listing. Real profiles list
# dozens of subjects; the listing exists for routing and contact lookup, so
# it shows a representative sample rather than flooding the caller's context.
_MAX_LISTED_SUBJECTS = 12


def format_librarian_directory(directory: LibrarianDirectory) -> str:
    """Format the complete configured directory for MCP responses.

    Aliases and keywords are deliberately omitted: they exist for matching,
    are often machine-expanded, and would drown the fields a caller needs to
    route a question (best-for areas, subjects, schools, contact).
    """
    lines = [
        "## Configured librarians:",
        "",
        f"{len(directory.librarians)} librarian profile(s) are configured.",
    ]
    for i, librarian in enumerate(directory.librarians, start=1):
        lines.append(f"{i}. Name: {_format_linked_name(librarian)}")
        lines.append(f"   Title: {librarian.title or _UNCONFIGURED}")
        lines.append(f"   Contact: {librarian.email or _UNCONFIGURED}")
        schools = _unique(librarian.schools)
        if schools:
            lines.append(f"   Schools: {_format_list(schools)}")
        if librarian.best_for:
            lines.append(
                f"   Best for: {_format_best_for_sentence(_unique(librarian.best_for))}"
            )
        subjects = _unique(librarian.subjects)
        if subjects:
            shown = subjects[:_MAX_LISTED_SUBJECTS]
            suffix = (
                f" (+{len(subjects) - len(shown)} more)"
                if len(subjects) > len(shown)
                else ""
            )
            lines.append(f"   Subjects: {', '.join(shown)}{suffix}")
    lines.append(_RECOMMENDATION_FOOTER)
    return "\n".join(lines)


def is_semantic_match(match: LibrarianMatch) -> bool:
    """True when a match came from the embedding fallback, not keywords."""
    return match.evidence_fields == ["semantic"]


def format_librarian_recommendations(
    matches: list[LibrarianMatch],
    query: str,
    *,
    configuration_message: str | None = None,
    semantic_error: str | None = None,
    semantic_skipped: str | None = None,
    skip_reason: str | None = None,
    near_misses: Sequence[LibrarianMatch] = (),
) -> str:
    """Format librarian recommendations for MCP responses.

    Matches from the embedding fallback (detected per match, since keyword
    and semantic results can now be mixed) surface their cosine similarity
    so callers can reason about confidence. ``semantic_error`` distinguishes
    "the semantic fallback errored" from a genuine no-match and
    ``semantic_skipped`` explains a deliberate skip (e.g. query too short);
    ``skip_reason`` reports why recommendation was skipped entirely (e.g.
    identifier query).

    ``near_misses`` (shown only on no_match) are the closest candidates
    that scored below the confidence threshold. They are rendered WITH
    their evidence so that any librarian a caller still chooses to mention
    always carries evidence -- never presented as a validated match.
    """
    if configuration_message:
        return (
            f"{_SECTION_HEADING}\n\n"
            "Status: unavailable\n"
            f"Message: Librarian recommendations unavailable: {configuration_message}"
        )

    if skip_reason:
        return (
            f"{_SECTION_HEADING}\n\n"
            "Status: skipped\n"
            f"Query: {query}\n"
            f"Message: {skip_reason}"
        )

    if semantic_error:
        error_note = (
            "Note: the semantic fallback errored and was skipped "
            f"({semantic_error}); only exact keyword matching ran."
        )
    elif semantic_skipped:
        error_note = (
            f"Note: the semantic fallback was skipped: {semantic_skipped}; "
            "only exact keyword matching ran."
        )
    else:
        error_note = None

    if not matches:
        lines = [
            _SECTION_HEADING,
            "",
            "Status: no_match",
            f"Query: {query}",
            f'Message: No librarian recommendation met the confidence threshold for "{query}".',
        ]
        if error_note:
            lines.append(error_note)
        if near_misses:
            lines.append(
                "Closest configured profiles (scored below the confidence "
                "threshold; NOT validated recommendations):"
            )
            for i, match in enumerate(near_misses, start=1):
                librarian = match.librarian
                lines.append(f"{i}. Name: {_format_linked_name(librarian)}")
                lines.append(f"   Title: {librarian.title or _UNCONFIGURED}")
                lines.append(f"   Contact: {librarian.email or _UNCONFIGURED}")
                lines.append(
                    f"   Evidence: {_format_match_evidence(match)} "
                    "(below the confidence threshold)"
                )
            lines.append(
                "If you still refer the user to one of these, present them "
                "as the closest configured contact rather than a validated "
                "recommendation, and always include the evidence shown above."
            )
        else:
            lines.append(
                "No configured profile matched even weakly. If the user "
                "still wants a contact, use primo_list_librarians and "
                "present the result as directory information; never present "
                "a librarian as recommended without showing evidence."
            )
        return "\n".join(lines)

    status = (
        "matched (semantic fallback)"
        if all(is_semantic_match(match) for match in matches)
        else "matched"
    )
    lines = [_SECTION_HEADING, "", f"Status: {status}"]
    if error_note:
        lines.append(error_note)
    for i, match in enumerate(matches, start=1):
        librarian = match.librarian
        semantic = is_semantic_match(match)
        if semantic:
            evidence = (
                f"Matched by semantic similarity (cosine {match.score:.2f}). "
                "No exact keyword match was found"
            )
        else:
            evidence = _format_match_evidence(match)
        lines.append(f"{i}. Name: {_format_linked_name(librarian)}")
        lines.append(f"   Title: {librarian.title or _UNCONFIGURED}")
        lines.append(f"   Contact: {librarian.email or _UNCONFIGURED}")
        if semantic:
            lines.append(
                f"   Similar profile topics: {_format_similar_profile_topics(librarian)}"
            )
        elif librarian.best_for:
            lines.append(f"   Best for: {_format_best_for_sentence(librarian.best_for)}")
        lines.append(f"   Evidence: {evidence}")

    lines.append(_RECOMMENDATION_FOOTER)
    return "\n".join(lines)
