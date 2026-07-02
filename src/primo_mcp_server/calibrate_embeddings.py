"""Calibration CLI for the semantic librarian fallback.

Prints the cosine similarity distribution of one or more test queries
against the configured librarian directory, so a deploying institution can
set PRIMO_LIBRARIAN_SEMANTIC_MIN_SIMILARITY and
PRIMO_LIBRARIAN_SEMANTIC_MARGIN empirically for their own directory instead
of trusting the SMU-tuned defaults.

Usage:
    python -m primo_mcp_server.calibrate_embeddings "query one" "query two"

Requires PRIMO_LIBRARIANS_FILE, PRIMO_LIBRARIAN_SEMANTIC_FALLBACK=true, and
PRIMO_EMBEDDING_API_KEY (via environment or .env). Profile embeddings are
cached the same way the server caches them, so repeated runs only embed the
queries.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from primo_mcp_server.config import PrimoConfig
from primo_mcp_server.librarian_embeddings import _accepted, score_profiles
from primo_mcp_server.librarians import load_librarian_directory


async def _run(queries: list[str]) -> int:
    config = PrimoConfig()
    directory, message = load_librarian_directory(config.librarians_file)
    if message or directory is None:
        print(message, file=sys.stderr)
        return 1
    if not config.embedding_api_key:
        print("PRIMO_EMBEDDING_API_KEY is not configured.", file=sys.stderr)
        return 1

    print(
        f"Directory: {config.librarians_file} "
        f"({len(directory.librarians)} profiles)\n"
        f"Model: {config.embedding_model}"
        + (
            f" @ {config.embedding_dimensions} dims"
            if config.embedding_dimensions
            else ""
        )
        + f"\nFloor: {config.librarian_semantic_min_similarity}  "
        f"Margin: {config.librarian_semantic_margin} "
        f"(applies at >= {config.librarian_semantic_margin_min_profiles} profiles)  "
        f"Top gap: {config.librarian_semantic_min_top_gap} (below that)\n"
        f"Min topical query tokens (server gate; not applied here): "
        f"{config.librarian_semantic_min_query_tokens}"
    )

    for query in queries:
        similarities = await score_profiles(directory, query, config)
        if not similarities:
            print(f'\nQuery: "{query}"\n  No embeddable profiles.')
            continue
        ranked = sorted(similarities, key=lambda s: -s.similarity)
        mean = sum(s.similarity for s in similarities) / len(similarities)
        accepted_ids = {s.librarian.id for s in _accepted(similarities, config)}

        print(f'\nQuery: "{query}"')
        print(f"  mean={mean:.4f}", end="")
        if len(ranked) >= 2:
            print(
                f"  top1={ranked[0].similarity:.4f}"
                f"  top1-top2 gap={ranked[0].similarity - ranked[1].similarity:.4f}"
                f"  top1-mean={ranked[0].similarity - mean:.4f}"
            )
        else:
            print()
        for sim, librarian in ranked:
            marker = "ACCEPT" if librarian.id in accepted_ids else "      "
            print(f"  {marker}  {sim:.4f}  {librarian.name} ({librarian.id})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m primo_mcp_server.calibrate_embeddings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("queries", nargs="+", help="Test queries to score.")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.queries))


if __name__ == "__main__":
    raise SystemExit(main())
