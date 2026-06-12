"""Embedder protocol: any object with embed() works.

Keeping this a Protocol (structural typing) rather than an ABC means tests and
alternative backends do not need to inherit from anything in this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Maps a batch of texts to a (n, dim) float32 matrix of unit vectors."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
