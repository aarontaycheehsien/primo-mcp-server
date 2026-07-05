"""Combined keyword + semantic librarian recommendation pipeline.

Shared by the MCP server and the offline evaluation harness
(``evaluate_recommendations``) so both rank librarians with exactly the
same logic; keeping the pipeline in one place is what makes benchmark
numbers trustworthy statements about server behaviour.
"""

from __future__ import annotations

from typing import NamedTuple

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_embeddings import semantic_fallback
from primo_mcp_server.librarians import (
    _MAX_RECOMMENDATIONS,
    LibrarianDirectory,
    LibrarianMatch,
    rank_librarians,
)
from primo_mcp_server.models import PrimoRecord

# How many below-threshold candidates a no_match outcome carries. Enough to
# offer the closest contact (plus one alternative) with real evidence, few
# enough that a no_match cannot be mistaken for a recommendation list.
_MAX_NEAR_MISSES = 2


class RecommendationOutcome(NamedTuple):
    """Ranked matches plus the semantic path's error/skip status."""

    matches: list[LibrarianMatch]
    semantic_error: str | None = None
    semantic_skipped: str | None = None
    # Top-scoring candidates below the confidence threshold, populated only
    # when matches is empty. They exist so a caller who still routes the
    # user to a librarian always has real evidence to show; they are never
    # validated recommendations.
    near_misses: tuple[LibrarianMatch, ...] = ()


async def recommend_with_fallback(
    directory: LibrarianDirectory,
    query: str,
    records: list[PrimoRecord] | None,
    config: PrimoConfig,
    *,
    limit: int = 2,
    specificity: dict[str, float] | None = None,
    embedding_timeout: float | None = None,
) -> RecommendationOutcome:
    """Rank librarians by keywords, second-guessed by the semantic path.

    Deterministic keyword matching runs first. The semantic path runs when
    keywords find nothing OR when the best keyword score falls below the
    second-guess threshold, so a marginal keyword win (one generic stemmed
    term) cannot suppress a strong semantic match. Keyword matches stay
    primary and are never displaced; passing semantic candidates for other
    librarians are appended within the limit. Embedding cost is still paid
    only when keywords are weak or absent.

    Identifier-shaped queries are the caller's concern: skipping them (and
    explaining the skip) happens before this pipeline runs.
    """
    candidates = rank_librarians(
        directory,
        query,
        records or [],
        specificity=specificity,
    )
    capped_limit = min(max(1, limit), _MAX_RECOMMENDATIONS)
    matches = [
        candidate
        for candidate in candidates
        if candidate.score >= config.librarian_min_score
    ][:capped_limit]
    semantic_error: str | None = None
    semantic_skipped: str | None = None
    semantic_near_miss: LibrarianMatch | None = None
    best_keyword_score = matches[0].score if matches else 0.0
    if config.librarian_semantic_fallback and (
        not matches
        or best_keyword_score < config.librarian_semantic_second_guess_score
    ):
        semantic = await semantic_fallback(
            directory,
            query,
            records,
            config,
            limit=limit,
            timeout=embedding_timeout,
        )
        semantic_error = semantic.error
        semantic_skipped = semantic.skipped
        semantic_near_miss = semantic.near_miss
        keyword_ids = {match.librarian.id for match in matches}
        matches = (
            matches
            + [
                match
                for match in semantic.matches
                if match.librarian.id not in keyword_ids
            ]
        )[:capped_limit]

    # When nothing cleared the threshold on either path, keep the closest
    # candidates so the no_match output can show why the best were not good
    # enough. Keyword near-misses come first (matched terms explain more
    # than a bare cosine); the semantic near-miss fills a remaining slot.
    near_misses: tuple[LibrarianMatch, ...] = ()
    if not matches:
        combined = list(candidates[:_MAX_NEAR_MISSES])
        if semantic_near_miss is not None and all(
            near.librarian.id != semantic_near_miss.librarian.id
            for near in combined
        ):
            combined.append(semantic_near_miss)
        near_misses = tuple(combined[:_MAX_NEAR_MISSES])
    return RecommendationOutcome(
        matches, semantic_error, semantic_skipped, near_misses
    )
