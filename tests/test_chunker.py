from ragdoc.ingestion.chunker import chunk_document, split_sentences
from ragdoc.ingestion.loader import Document


def make_doc(text: str) -> Document:
    return Document(text=text, source="test.txt")


def test_short_document_is_single_chunk():
    chunks = chunk_document(make_doc("One sentence. Another sentence."), chunk_size=200)
    assert len(chunks) == 1
    assert chunks[0].source == "test.txt"


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} contains some words." for i in range(100))
    chunks = chunk_document(make_doc(text), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 50 for c in chunks)  # small slack for joins


def test_overlap_carries_content_between_chunks():
    text = " ".join(f"Fact {i} is stated here clearly." for i in range(60))
    chunks = chunk_document(make_doc(text), chunk_size=300, chunk_overlap=120)
    for prev, nxt in zip(chunks, chunks[1:]):
        # The next chunk must start with text already seen at the end of prev.
        first_sentence = nxt.text.split(".")[0]
        assert first_sentence in prev.text


def test_oversized_sentence_is_hard_split():
    text = "x" * 5000  # no sentence boundaries at all
    chunks = chunk_document(make_doc(text), chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 900 for c in chunks)


def test_no_sentence_cut_mid_way():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    sentences = split_sentences(text)
    assert sentences == ["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."]


def test_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(make_doc("Hello."), chunk_size=100, chunk_overlap=100)
