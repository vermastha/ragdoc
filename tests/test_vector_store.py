import numpy as np

from ragdoc.ingestion.chunker import Chunk
from ragdoc.retrieval.vector_store import VectorStore


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_search_returns_nearest_by_cosine():
    store = VectorStore()
    chunks = [Chunk(f"chunk {i}", "s", i) for i in range(3)]
    store.add(chunks, np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])]))
    hits = store.search(unit([1, 0.1]), k=2)
    assert hits[0][0] == 0
    assert hits[0][1] > hits[1][1]


def test_save_and_load_roundtrip(tmp_path, embedder):
    store = VectorStore()
    chunks = [Chunk("alpha", "a.txt", 0), Chunk("beta", "b.txt", 0)]
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.chunks == chunks
    query = embedder.embed(["alpha"])[0]
    assert loaded.search(query, k=1)[0][0] == store.search(query, k=1)[0][0]


def test_load_missing_index_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope")
