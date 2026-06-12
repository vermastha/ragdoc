"""Dependency-free fallback embedder using hashed character n-grams.

This is *not* a semantic embedder — it captures lexical overlap only. It
exists so the pipeline, tests, and CI run with zero model downloads, and so
the system degrades gracefully when sentence-transformers is unavailable.
The Embedder protocol makes the two backends interchangeable.
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self._ngram_range = ngram_range

    def _ngrams(self, text: str):
        text = " ".join(text.lower().split())
        lo, hi = self._ngram_range
        for n in range(lo, hi + 1):
            for i in range(max(len(text) - n + 1, 0)):
                yield text[i : i + n]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in self._ngrams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0  # signed hashing reduces collisions
                matrix[row, bucket] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
