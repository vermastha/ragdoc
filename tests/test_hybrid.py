from ragdoc.ingestion.chunker import Chunk
from ragdoc.retrieval.hybrid import HybridRetriever


def chunks_from(texts):
    return [Chunk(t, "doc.txt", i) for i, t in enumerate(texts)]


def test_hybrid_finds_exact_token_dense_would_miss(embedder):
    retriever = HybridRetriever(embedder)
    retriever.index(
        chunks_from(
            [
                "General information about our products and services.",
                "Troubleshooting guide: error XK-99 indicates a failed sensor.",
                "Company history and mission statement.",
            ]
        )
    )
    results = retriever.retrieve("XK-99", top_k=1)
    assert results[0].chunk.chunk_id == 1


def test_chunk_found_by_both_retrievers_ranks_above_single_source(embedder):
    retriever = HybridRetriever(embedder)
    retriever.index(
        chunks_from(
            [
                "Refund policy: customers may request a refund within 30 days of purchase.",
                "Unrelated text about office plants and watering schedules.",
                "Another unrelated paragraph about parking arrangements.",
            ]
        )
    )
    results = retriever.retrieve("refund within 30 days", top_k=3)
    top = results[0]
    assert top.chunk.chunk_id == 0
    assert set(top.found_by) == {"dense", "bm25"}


def test_retrieve_on_empty_index_returns_empty(embedder):
    retriever = HybridRetriever(embedder)
    assert retriever.retrieve("anything") == []
