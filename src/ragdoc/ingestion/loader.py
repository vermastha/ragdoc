"""Document loading: turn files into plain text + metadata.

Supports .txt, .md, and .pdf. The loader is deliberately simple — each format
returns a single text blob; structure-aware parsing (headings, tables) is a
documented extension point, not premature complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    """A loaded source document."""

    text: str
    source: str  # file path or identifier, used for citations


def load_file(path: str | Path) -> Document:
    """Load a single file into a Document. Raises ValueError on unsupported types."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return Document(text=path.read_text(encoding="utf-8", errors="replace"), source=str(path))
    if ext == ".pdf":
        return Document(text=_read_pdf(path), source=str(path))
    raise ValueError(f"Unsupported file type {ext!r}: {path}")


def load_path(path: str | Path) -> list[Document]:
    """Load a file, or recursively load all supported files in a directory."""
    path = Path(path)
    if path.is_file():
        return [load_file(path)]
    if path.is_dir():
        docs = [
            load_file(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not docs:
            raise ValueError(f"No supported documents found under {path}")
        return docs
    raise FileNotFoundError(path)


def load_many(paths: Iterable[str | Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        docs.extend(load_path(p))
    return docs


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # local import keeps base import light

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)
