"""Dense embeddings via sentence-transformers (optional dependency)."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model; normalizes output to unit vectors
    so cosine similarity reduces to a dot product in the vector store."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install \"ragdoc[embeddings]\""
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
