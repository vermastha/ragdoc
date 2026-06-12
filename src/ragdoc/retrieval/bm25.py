"""BM25 (Okapi) lexical retriever, implemented from scratch.

Dense retrieval is strong on paraphrase ("How do I cancel?" matching
"terminating your subscription") but notoriously weak on exact tokens —
error codes, function names, product SKUs, legal section numbers. BM25 is
the mirror image. ragdoc runs both and fuses the rankings (see hybrid.py).

Implemented directly (~60 lines) rather than pulling in rank_bm25: the
algorithm is small, and owning it makes the scoring fully inspectable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()  # document frequency per term
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._doc_freqs)

    def add(self, texts: list[str]) -> None:
        for text in texts:
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            for term in freqs:
                self._df[term] += 1
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0

    def _idf(self, term: str) -> float:
        n, df = len(self._doc_freqs), self._df.get(term, 0)
        # BM25+-style floor at 0 avoids negative IDF for very common terms.
        return max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 0.0)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return [(doc_index, bm25_score)] sorted by score, descending."""
        query_terms = tokenize(query)
        if not query_terms or not self._doc_freqs:
            return []
        scores = [0.0] * len(self._doc_freqs)
        for term in query_terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self._doc_lens[i] / self._avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]
