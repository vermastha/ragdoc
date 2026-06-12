"""RagPipeline: the public facade. Ingest -> chunk -> index -> retrieve -> answer.

Every stage is injected, so each can be swapped or mocked independently:

    pipeline = RagPipeline(embedder=HashingEmbedder(), answerer=ExtractiveAnswerer())
    pipeline.ingest(["docs/"])
    answer = pipeline.ask("What is the refund policy?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ragdoc.config import Settings
from ragdoc.embeddings.base import Embedder
from ragdoc.embeddings.hashing import HashingEmbedder
from ragdoc.generation.answerer import Answer, auto_answerer
from ragdoc.ingestion.chunker import chunk_documents
from ragdoc.ingestion.loader import load_many
from ragdoc.retrieval.hybrid import HybridRetriever, RetrievalResult


def default_embedder(settings: Settings) -> Embedder:
    """Best available embedder: sentence-transformers if installed, else hashing."""
    try:
        from ragdoc.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.embedding_model)
    except ImportError:
        return HashingEmbedder()


class RagPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        answerer=None,
    ) -> None:
        self.settings = settings or Settings()
        self.embedder = embedder or default_embedder(self.settings)
        self.answerer = answerer or auto_answerer(self.settings.llm_model)
        self.retriever = HybridRetriever(self.embedder, rrf_k=self.settings.rrf_k)

    # -- indexing -------------------------------------------------------------

    def ingest(self, paths: Iterable[str | Path]) -> int:
        """Load, chunk, and index documents. Returns the number of new chunks."""
        docs = load_many(paths)
        chunks = chunk_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.index(chunks)
        return len(chunks)

    def save_index(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or self.settings.index_dir)
        self.retriever.vector_store.save(directory)
        return directory

    def load_index(self, directory: str | Path | None = None) -> int:
        """Rehydrate a saved index. BM25 is rebuilt from chunk text (cheap, and
        avoids persisting a second artifact that can drift out of sync)."""
        from ragdoc.retrieval.vector_store import VectorStore

        directory = Path(directory or self.settings.index_dir)
        store = VectorStore.load(directory)
        self.retriever.vector_store = store
        self.retriever.bm25.add([c.text for c in store.chunks])
        return len(store)

    # -- querying -------------------------------------------------------------

    def retrieve(self, question: str) -> list[RetrievalResult]:
        return self.retriever.retrieve(
            question,
            top_k=self.settings.top_k,
            candidate_k=self.settings.candidate_k,
        )

    def ask(self, question: str) -> Answer:
        return self.answerer.answer(question, self.retrieve(question))
