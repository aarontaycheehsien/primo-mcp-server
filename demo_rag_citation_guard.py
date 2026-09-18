"""A minimal RAG pipeline showing how *code* (not the LLM) enforces that
citations only point to retrieved sources.

Pipeline (matches the infographic):
    1. Retrieve source records          (deterministic code)
    2. Draft answer with [R#] citations (LLM step -- stubbed here)
    3. Extract structured citation IDs  (deterministic code)
    4. Validate IDs against retrieved   (deterministic code)
       -> invalid ID? regenerate the answer with feedback
    5. Build references from validated IDs only (deterministic code)

Honest limits (also from the infographic):
    - Only structured [R#] tags are protected; prose citations and
      uncited claims bypass this check.
    - Validation proves the ID *was retrieved*, not that the source
      actually supports the claim it is attached to.

Run:  uv run python demo_rag_citation_guard.py
No dependencies, no API keys. Swap `fake_llm_draft` for a real LLM call.
"""

import json
import re
import sys

# --------------------------------------------------------------------------
# A tiny "index". In a real system this is your search API / vector store.
# --------------------------------------------------------------------------
CORPUS = [
    {
        "id": "D1",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al.",
        "year": 2020,
        "text": "Retrieval-augmented generation (RAG) combines a retriever with a "
                "generator so answers are grounded in retrieved documents.",
    },
    {
        "id": "D2",
        "title": "Hallucination in Large Language Models: A Survey",
        "authors": "Ji et al.",
        "year": 2023,
        "text": "Large language models frequently hallucinate, producing fluent "
                "but unsupported or fabricated statements, including fake citations.",
    },
    {
        "id": "D3",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Karpukhin et al.",
        "year": 2020,
        "text": "Dense passage retrieval uses learned embeddings to find relevant "
                "passages for open-domain question answering.",
    },
    {
        "id": "D4",
        "title": "A Recipe for Sourdough Bread",
        "authors": "Baker",
        "year": 2019,
        "text": "Mix flour, water, salt, and starter. Fold, proof, and bake in a "
                "dutch oven at high heat.",
    },
]


# --------------------------------------------------------------------------
# STEP 1 -- Retrieve source records (deterministic code)
# --------------------------------------------------------------------------
def retrieve(query: str, k: int = 3) -> list[dict]:
    """Toy keyword retrieval: rank documents by word overlap with the query."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for doc in CORPUS:
        doc_words = set(re.findall(r"\w+", (doc["title"] + " " + doc["text"]).lower()))
        score = len(query_words & doc_words)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# --------------------------------------------------------------------------
# STEP 2 -- Draft answer (the ONLY LLM step)
# --------------------------------------------------------------------------
def fake_llm_draft(query: str, records: list[dict], feedback: str | None) -> str:
    """Stand-in for a real LLM call.

    A real implementation sends the retrieved records plus an instruction like:
      "Answer using ONLY the sources below. Cite them inline as [R1], [R2]...
       Do not cite anything not listed."
    and, on retry, appends the validator's feedback.

    To make the demo interesting, the first draft hallucinates [R999] --
    a citation to a source that was never retrieved. When the validator's
    feedback comes back, the "model" produces a clean draft.
    """
    if feedback is None:
        return (
            "RAG grounds answers in retrieved documents [R1]. This matters because "
            "LLMs are prone to hallucinating unsupported claims and even fake "
            "citations [R2]. One study found RAG eliminates hallucination entirely "
            "[R999]."
        )
    return (
        "RAG grounds answers in retrieved documents by combining a retriever with "
        "a generator [R1]. This matters because LLMs are prone to hallucinating "
        "unsupported claims and even fake citations [R2]."
    )


# --------------------------------------------------------------------------
# STEP 3 -- Extract structured citation IDs (deterministic code)
# --------------------------------------------------------------------------
def extract_citation_ids(answer: str) -> list[str]:
    """Pull every [R#] tag out of the draft, preserving first-seen order."""
    seen: list[str] = []
    for tag in re.findall(r"\[R(\d+)\]", answer):
        if tag not in seen:
            seen.append(tag)
    return seen


# --------------------------------------------------------------------------
# STEP 4 -- Validate IDs against retrieved records (deterministic code)
# --------------------------------------------------------------------------
def validate_ids(cited: list[str], records: list[dict]) -> tuple[list[str], list[str]]:
    """Split cited IDs into (valid, invalid) against the retrieved set."""
    allowed = {str(i + 1) for i in range(len(records))}  # R1..Rn map to records
    valid = [c for c in cited if c in allowed]
    invalid = [c for c in cited if c not in allowed]
    return valid, invalid


# --------------------------------------------------------------------------
# STEP 5 -- Build references deterministically from validated IDs only
# --------------------------------------------------------------------------
def build_references(valid_ids: list[str], records: list[dict]) -> str:
    """The reference list comes from OUR records, never from LLM text."""
    lines = []
    for rid in sorted(valid_ids, key=int):
        doc = records[int(rid) - 1]
        lines.append(f"[R{rid}] {doc['authors']} ({doc['year']}). {doc['title']}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The guard loop that ties it together
# --------------------------------------------------------------------------
def answer_with_enforced_citations(query: str, max_attempts: int = 3) -> str:
    records = retrieve(query)
    print(f"STEP 1  Retrieved {len(records)} records for: {query!r}")
    for i, doc in enumerate(records, start=1):
        print(f"        R{i} = {doc['id']}: {doc['title']}")

    feedback = None
    for attempt in range(1, max_attempts + 1):
        print(f"\nSTEP 2  LLM draft (attempt {attempt})")
        draft = fake_llm_draft(query, records, feedback)
        print(f"        {draft}")

        cited = extract_citation_ids(draft)
        print(f"STEP 3  Code extracted citation IDs: {['R' + c for c in cited]}")

        valid, invalid = validate_ids(cited, records)
        if invalid:
            bad = ", ".join("R" + c for c in invalid)
            print(f"STEP 4  INVALID -> {bad} was never retrieved. Regenerating...")
            feedback = (
                f"Your draft cited [{bad}], which is not among the retrieved "
                f"sources R1-R{len(records)}. Rewrite the answer citing only "
                f"those sources."
            )
            continue

        print(f"STEP 4  All cited IDs are valid: {['R' + c for c in valid]}")
        references = build_references(valid, records)
        print("STEP 5  References built deterministically from validated IDs.")
        return draft + "\n\nReferences:\n" + references

    raise RuntimeError(f"LLM failed to produce valid citations in {max_attempts} attempts")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "why does retrieval augmented generation reduce hallucination"
    final = answer_with_enforced_citations(query)
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final)
