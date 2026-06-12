"""Chunking: split documents into overlapping, retrieval-sized pieces.

Strategy: greedy sentence packing. We split on sentence boundaries (falling
back to paragraph and hard splits for pathological inputs), then pack
sentences into chunks up to `chunk_size` characters with `chunk_overlap`
characters carried over between consecutive chunks.

Why sentence-aware instead of fixed-window? Fixed windows routinely cut
sentences in half, which hurts both embedding quality (truncated semantics)
and the readability of retrieved context shown to the LLM. Overlap exists so
that facts straddling a chunk boundary are fully contained in at least one
chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragdoc.ingestion.loader import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text."""

    text: str
    source: str
    chunk_id: int  # position within its source document


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, treating blank lines as hard boundaries."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(_SENTENCE_SPLIT.split(paragraph))
    return sentences


def chunk_document(doc: Document, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_sentences(doc.text)
    # Guard against single "sentences" longer than the chunk size (tables,
    # minified text, OCR noise): hard-split them so packing always terminates.
    units: list[str] = []
    for s in sentences:
        if len(s) <= chunk_size:
            units.append(s)
        else:
            units.extend(s[i : i + chunk_size] for i in range(0, len(s), chunk_size))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append(Chunk(text=" ".join(current), source=doc.source, chunk_id=len(chunks)))
        # Carry trailing sentences forward as overlap for the next chunk.
        carried: list[str] = []
        carried_len = 0
        for s in reversed(current):
            if carried_len + len(s) > chunk_overlap:
                break
            carried.insert(0, s)
            carried_len += len(s) + 1
        current = carried
        current_len = carried_len

    for unit in units:
        if current_len + len(unit) + 1 > chunk_size and current:
            flush()
        current.append(unit)
        current_len += len(unit) + 1
    if current:
        chunks.append(Chunk(text=" ".join(current), source=doc.source, chunk_id=len(chunks)))

    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 150
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return out
