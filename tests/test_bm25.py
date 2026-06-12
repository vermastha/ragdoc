from ragdoc.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! err-42") == ["hello", "world", "err", "42"]


def test_exact_term_match_ranks_first():
    index = BM25Index()
    index.add(
        [
            "The refund policy allows returns within 30 days.",
            "Our shipping times vary by region.",
            "Error code E1234 means the disk is full.",
        ]
    )
    hits = index.search("what does error code E1234 mean", k=3)
    assert hits[0][0] == 2


def test_rare_terms_outweigh_common_terms():
    index = BM25Index()
    index.add(
        [
            "the the the the common words document",
            "quantum entanglement experiments in the lab",
        ]
    )
    hits = index.search("quantum entanglement", k=2)
    assert hits[0][0] == 1


def test_empty_query_returns_nothing():
    index = BM25Index()
    index.add(["some document"])
    assert index.search("!!!", k=5) == []
