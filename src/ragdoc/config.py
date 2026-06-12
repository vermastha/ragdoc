"""Central configuration with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Tunable knobs for the pipeline.

    Defaults are chosen for a laptop-friendly footprint; everything can be
    overridden via environment variables or constructor args.
    """

    # Chunking
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # characters of overlap between chunks

    # Retrieval
    top_k: int = 4                 # chunks handed to the LLM
    candidate_k: int = 20          # candidates fetched per retriever before fusion
    rrf_k: int = 60                # RRF damping constant (standard value from the paper)

    # Models
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAGDOC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("RAGDOC_LLM_MODEL", "claude-sonnet-4-5")
    )

    # Persistence
    index_dir: str = field(default_factory=lambda: os.getenv("RAGDOC_INDEX_DIR", ".ragdoc_index"))
