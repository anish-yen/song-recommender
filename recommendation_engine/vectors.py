"""Vector primitives shared by the encoder, the index, and retrieval.

Kept separate from ``similarity`` so that validation rules have no dependency on
ranking policy, and from ``embeddings`` so they can be tested without torch.
"""

from __future__ import annotations

import numpy as np

from .config import EmbeddingSpace
from .errors import EmbeddingError

#: Vectors shorter than this are treated as degenerate: dividing by such a norm
#: amplifies floating-point noise into a meaningless direction.
MIN_NORM = 1e-6


def validate_vector(vector: np.ndarray, space: EmbeddingSpace, *, label: str) -> np.ndarray:
    """Return ``vector`` as a 1-D float32 array, or raise ``EmbeddingError``.

    ``label`` identifies the offending vector in the message (a track id, or the
    query text), because "invalid embedding" alone is not actionable when you are
    indexing several hundred files.
    """
    array = np.asarray(vector)
    if array.ndim != 1:
        raise EmbeddingError(
            f"{label}: expected a 1-D embedding, got shape {array.shape}"
        )
    if array.shape[0] != space.dim:
        raise EmbeddingError(
            f"{label}: expected {space.dim} dimensions, got {array.shape[0]}"
        )
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise EmbeddingError(f"{label}: embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(array))
    if norm < MIN_NORM:
        raise EmbeddingError(f"{label}: embedding norm {norm:.3g} is effectively zero")
    return array


def l2_normalize(vector: np.ndarray, space: EmbeddingSpace, *, label: str) -> np.ndarray:
    """Validate then scale to unit length, so cosine similarity is a dot product."""
    array = validate_vector(vector, space, label=label)
    return array / float(np.linalg.norm(array))


def pool_excerpts(
    excerpt_vectors: np.ndarray, space: EmbeddingSpace, *, label: str
) -> np.ndarray:
    """Collapse per-excerpt embeddings into one unit-length track vector.

    Normalising before averaging stops a single loud excerpt with a large raw
    norm from dominating the track's direction; the final normalisation makes the
    result directly comparable with query vectors.
    """
    matrix = np.asarray(excerpt_vectors)
    if matrix.ndim != 2:
        raise EmbeddingError(
            f"{label}: expected a 2-D (excerpts, dim) array, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise EmbeddingError(f"{label}: no excerpt embeddings to pool")
    unit = np.stack(
        [
            l2_normalize(row, space, label=f"{label} excerpt {i}")
            for i, row in enumerate(matrix)
        ]
    )
    return l2_normalize(unit.mean(axis=0), space, label=f"{label} pooled")


def stack_matrix(
    vectors: list[np.ndarray], space: EmbeddingSpace, *, labels: list[str]
) -> np.ndarray:
    """Build a normalised ``(n, dim)`` float32 matrix from per-track vectors."""
    if len(vectors) != len(labels):
        raise ValueError("vectors and labels must be the same length")
    if not vectors:
        return np.zeros((0, space.dim), dtype=np.float32)
    return np.stack(
        [l2_normalize(v, space, label=label) for v, label in zip(vectors, labels)]
    )
