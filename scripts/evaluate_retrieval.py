"""Retrieval evaluation harness: hit-rate@k and MRR over a labeled QA set.

Usage:
    python scripts/evaluate_retrieval.py examples/sample_docs examples/eval_set.json

The eval set is a JSON list of {"question": ..., "expected_source": ...,
"expected_substring": ...}. A retrieval counts as a hit if any retrieved chunk
comes from the expected source AND contains the expected substring.

This deliberately evaluates *retrieval* in isolation from generation: if the
right chunk never reaches the LLM, no prompt engineering can save the answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragdoc.config import Settings
from ragdoc.pipeline import RagPipeline


def evaluate(docs_path: str, eval_path: str, top_k: int = 4) -> dict:
    pipeline = RagPipeline(settings=Settings(top_k=top_k))
    pipeline.ingest([docs_path])
    cases = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    hits, reciprocal_ranks = 0, []
    for case in cases:
        results = pipeline.retrieve(case["question"])
        rank = next(
            (
                i
                for i, r in enumerate(results, start=1)
                if case["expected_source"] in r.chunk.source
                and case["expected_substring"].lower() in r.chunk.text.lower()
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            print("  MISS:", repr(case["question"]))

    n = len(cases)
    return {
        "cases": n,
        f"hit_rate@{top_k}": round(hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
    }


if __name__ == "__main__":
    docs = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_docs"
    eval_set = sys.argv[2] if len(sys.argv) > 2 else "examples/eval_set.json"
    print(json.dumps(evaluate(docs, eval_set), indent=2))
