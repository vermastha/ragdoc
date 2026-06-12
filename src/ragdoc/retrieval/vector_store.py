"""In-memory vector store with cosine search and disk persistence.

Why not FAISS/Chroma? For corpora that fit in RAM (up to a few hundred
thousand chunks), brute-force matrix multiplication on normalized vectors is
exact, fast, and dependency-free. The class boundary is the seam where an ANN
index (FAISS HNSW, pgvector, etc.) would slot in once exact search becomes the
bottleneck — a deliberate "simplest thing that scales far enough" choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragdoc.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None  # (n, dim), rows are unit vectors
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        vectors = vectors.astype(np.float32)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_index, cosine_score)] for the top-k chunks."""
        if self._matrix is None or len(self.chunks) == 0:
            return []
        scores = self._matrix @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(int(i), float(scores[i])) for i in top]

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)
        payload = [
            {"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in self.chunks
        ]
        (directory / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        store = cls()
        chunks_file = directory / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"No index found at {directory}")
        payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        store.chunks = [Chunk(**item) for item in payload]
        vectors_file = directory / "vectors.npy"
        if vectors_file.exists():
            store._matrix = np.load(vectors_file)
        return store
