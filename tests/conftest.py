import pytest

from ragdoc.embeddings.hashing import HashingEmbedder


@pytest.fixture()
def embedder():
    """Offline embedder so the suite needs no model downloads or network."""
    return HashingEmbedder(dim=128)
