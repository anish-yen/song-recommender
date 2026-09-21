"""Error types shared across the recommendation engine.

A single hierarchy means callers (the CLI today, FastAPI routes later) can map
engine failures onto responses without importing every submodule.
"""

from __future__ import annotations


class RecommenderError(Exception):
    """Base class for every error raised by the recommendation engine."""


class ManifestError(RecommenderError):
    """The catalog manifest is malformed, or references audio that is unusable."""


class AudioError(RecommenderError):
    """Audio could not be decoded, or is unusable (silent, non-finite, empty)."""


class EmbeddingError(RecommenderError):
    """An embedding is invalid: wrong dimension, non-finite, or zero-length."""


class EmbeddingSpaceMismatch(RecommenderError):
    """Vectors from different embedding spaces were about to be compared.

    Cosine similarity between two unrelated coordinate systems produces a
    number, not a meaning, so this is always an error rather than a warning.
    """


class IndexCorruptError(RecommenderError):
    """A persisted index is missing, unreadable, or internally inconsistent."""


class UnknownTrackError(RecommenderError):
    """A seed track id is not present in the catalog."""
