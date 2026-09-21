"""Content-based music recommendation: mood search and similar-song retrieval.

The public surface is deliberately small so that the API layer can depend on it
without reaching into internals:

    from recommendation_engine import ClapEncoder, build_index, load_index
    from recommendation_engine import search, similar_to

``ClapEncoder`` is imported lazily because it pulls in torch and transformers;
retrieval, indexing, and manifest handling stay importable without them.
"""

from __future__ import annotations

from .config import DEFAULT_SPACE, EmbeddingSpace
from .errors import (
    AudioError,
    EmbeddingError,
    EmbeddingSpaceMismatch,
    IndexCorruptError,
    ManifestError,
    RecommenderError,
    UnknownTrackError,
)
from .index import BuildReport, build_index, load_index, save_index
from .manifest import Track, load_manifest, parse_tracks
from .similarity import Catalog, Recommendation, cosine_scores, search, similar_to

__all__ = [
    "AudioError",
    "BuildReport",
    "Catalog",
    "ClapEncoder",
    "DEFAULT_SPACE",
    "EmbeddingError",
    "EmbeddingSpace",
    "EmbeddingSpaceMismatch",
    "IndexCorruptError",
    "ManifestError",
    "Recommendation",
    "RecommenderError",
    "Track",
    "UnknownTrackError",
    "build_index",
    "cosine_scores",
    "load_index",
    "load_manifest",
    "parse_tracks",
    "save_index",
    "search",
    "similar_to",
]


def __getattr__(name: str):
    """Defer the torch import until someone actually needs inference."""
    if name == "ClapEncoder":
        from .embeddings import ClapEncoder

        return ClapEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
