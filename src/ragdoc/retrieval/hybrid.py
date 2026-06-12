"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion (RRF).

RRF (Cormack, Clarke & Buettcher, 2009) combines rankings using only ranks:

    score(d) = sum over retrievers r of  1 / (k + rank_r(d))

Why ranks instead of scores? Cosine similarities and BM25 scores live on
incomparable scales, and any weighted-sum scheme needs per-corpus calibration.
RRF needs none, is robust to one retriever misfiring, and consistently ranks
documents found by *both* retrievers above documents found by only one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragdoc.embeddings.base import Embedder
from ragdoc.ingestion.chunker import Chunk
from ragdoc.retrieval.bm25 import BM25Index
from ragdoc.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float          # fused RRF score
    found_by: tuple[str, ...]  # which retrievers surfaced it ("dense", "bm25")


class HybridRetriever:
    def __init__(self, embedder: Embedder, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()

    def __len__(self) -> int:
        return len(self.vector_store)

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        self.vector_store.add(chunks, self.embedder.embed(texts))
        self.bm25.add(texts)

    def retrieve(self, query: str, top_k: int = 4, candidate_k: int = 20) -> list[RetrievalResult]:
        query_vec = self.embedder.embed([query])[0]
        dense_hits = self.vector_store.search(query_vec, k=candidate_k)
        bm25_hits = self.bm25.search(query, k=candidate_k)

        fused: dict[int, float] = {}
        provenance: dict[int, set[str]] = {}
        for name, hits in (("dense", dense_hits), ("bm25", bm25_hits)):
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                provenance.setdefault(idx, set()).add(name)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                chunk=self.vector_store.chunks[idx],
                score=score,
                found_by=tuple(sorted(provenance[idx])),
            )
            for idx, score in ranked
        ]
