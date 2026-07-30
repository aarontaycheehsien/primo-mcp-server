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
    recommend_librarians,
)
from primo_mcp_server.models import PrimoRecord


class RecommendationOutcome(NamedTuple):
    """Ranked matches plus the semantic path's error/skip status."""

    matches: list[LibrarianMatch]
    semantic_error: str | None = None
    semantic_skipped: str | None = None


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
    matches = recommend_librarians(
        directory,
        query,
        records or [],
        limit=limit,
        min_score=config.librarian_min_score,
        specificity=specificity,
    )
    semantic_error: str | None = None
    semantic_skipped: str | None = None
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
        keyword_ids = {match.librarian.id for match in matches}
        matches = (
            matches
            + [
                match
                for match in semantic.matches
                if match.librarian.id not in keyword_ids
            ]
        )[: min(max(1, limit), _MAX_RECOMMENDATIONS)]
    return RecommendationOutcome(matches, semantic_error, semantic_skipped)
