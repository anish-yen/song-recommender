"""Retrieval and ranking over stored embeddings.

Exact NumPy cosine similarity. This is intentionally the simplest correct
implementation: when the pgvector port lands it must agree with these numbers, so
this module doubles as the reference oracle for that work.

``similarity`` in the output is a cosine similarity in ``[-1, 1]``. It is a
ranking score, not a probability and not a "percentage match" — do not render it
as one.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import EmbeddingSpace
from .errors import EmbeddingSpaceMismatch, UnknownTrackError
from .vectors import MIN_NORM, validate_vector


@dataclasses.dataclass(frozen=True)
class Recommendation:
    """One ranked result. ``evidence`` stays empty unless something supports it."""

    track_id: str
    title: str
    artist: str
    similarity: float
    provider_url: str | None = None
    #: Filled by ``evidence.py``: the measured match, filters that actually kept the
    #: track, and model estimates marked with their ``kind``. Never from raw
    #: embedding dimensions, and never from an LLM.
    evidence: dict[str, object] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "similarity": self.similarity,
        }
        if self.provider_url:
            payload["provider_url"] = self.provider_url
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


@dataclasses.dataclass(frozen=True)
class Catalog:
    """An in-memory searchable catalog: aligned ids, metadata, and unit vectors."""

    space: EmbeddingSpace
    track_ids: tuple[str, ...]
    metadata: dict[str, dict[str, object]]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.track_ids):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors for {len(self.track_ids)} track ids"
            )
        if self.vectors.shape[0] and self.vectors.shape[1] != self.space.dim:
            raise ValueError(
                f"vectors have {self.vectors.shape[1]} dimensions, "
                f"space declares {self.space.dim}"
            )
        if len(set(self.track_ids)) != len(self.track_ids):
            raise ValueError("catalog contains duplicate track ids")

    def __len__(self) -> int:
        return len(self.track_ids)

    def position_of(self, track_id: str) -> int:
        try:
            return self.track_ids.index(track_id)
        except ValueError as exc:
            raise UnknownTrackError(
                f"track {track_id!r} is not in the catalog ({len(self)} tracks indexed)"
            ) from exc

    def vector_of(self, track_id: str) -> np.ndarray:
        return self.vectors[self.position_of(track_id)]


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``query`` and every row of ``matrix``.

    Norms are divided out explicitly rather than assuming unit-length inputs, so
    this stays correct as a reference implementation regardless of the caller.
    """
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {query.shape}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"query has {query.shape[0]} dimensions, matrix rows have {matrix.shape[1]}"
        )

    query_norm = float(np.linalg.norm(query))
    if query_norm < MIN_NORM:
        raise ValueError("query vector has effectively zero norm")
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms < MIN_NORM):
        raise ValueError("catalog contains a vector with effectively zero norm")

    scores = (matrix @ query) / (row_norms * query_norm)
    # Floating-point error can push an identical pair slightly past 1.0; clipping
    # keeps the contract that results are valid cosine similarities.
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer, got {type(k).__name__}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k


def _require_same_space(catalog: Catalog, query_space: EmbeddingSpace | None) -> None:
    if query_space is None:
        return
    if query_space.fingerprint() != catalog.space.fingerprint():
        raise EmbeddingSpaceMismatch(
            "query and catalog come from different embedding spaces; "
            f"query={query_space.describe()} catalog={catalog.space.describe()}. "
            "Rebuild the index with the current configuration."
        )


def rank_by_score(
    catalog: Catalog,
    scores: np.ndarray,
    *,
    k: int,
    exclude: frozenset[str] = frozenset(),
    max_per_artist: int | None = None,
) -> list[Recommendation]:
    """Turn raw scores into ranked results.

    Order is descending score, with ties broken by ``track_id`` ascending. The
    tie-break is not cosmetic: without it, two tracks with identical scores would
    swap places depending on NumPy's sort internals, so a rerun of the same query
    could show a different ordering.
    """
    k = _validate_k(k)
    if max_per_artist is not None:
        if isinstance(max_per_artist, bool) or not isinstance(max_per_artist, int):
            raise ValueError("max_per_artist must be an integer or None")
        if max_per_artist < 1:
            raise ValueError(f"max_per_artist must be at least 1, got {max_per_artist}")

    candidates = [
        (self_id, float(score))
        for self_id, score in zip(catalog.track_ids, scores)
        if self_id not in exclude
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))

    results: list[Recommendation] = []
    per_artist: dict[str, int] = {}
    for track_id, score in candidates:
        if len(results) == k:
            break
        meta = catalog.metadata.get(track_id, {})
        artist = str(meta.get("artist", ""))
        if max_per_artist is not None:
            key = artist.casefold()
            if per_artist.get(key, 0) >= max_per_artist:
                continue
            per_artist[key] = per_artist.get(key, 0) + 1
        results.append(
            Recommendation(
                track_id=track_id,
                title=str(meta.get("title", "")),
                artist=artist,
                similarity=score,
                provider_url=meta.get("provider_url") or None,  # type: ignore[arg-type]
            )
        )
    return results


def search(
    catalog: Catalog,
    query_vector: np.ndarray,
    *,
    k: int = 10,
    query_space: EmbeddingSpace | None = None,
    exclude: frozenset[str] = frozenset(),
    max_per_artist: int | None = None,
) -> list[Recommendation]:
    """Rank the catalog against an already-encoded query vector.

    Returns fewer than ``k`` results when the catalog holds fewer usable tracks,
    and an empty list for an empty catalog — neither is an error.
    """
    k = _validate_k(k)
    _require_same_space(catalog, query_space)
    if len(catalog) == 0:
        return []
    vector = validate_vector(query_vector, catalog.space, label="query")
    scores = cosine_scores(vector, catalog.vectors)
    return rank_by_score(
        catalog, scores, k=k, exclude=exclude, max_per_artist=max_per_artist
    )


def similar_to(
    catalog: Catalog,
    track_id: str,
    *,
    k: int = 10,
    exclude: frozenset[str] = frozenset(),
    max_per_artist: int | None = None,
) -> list[Recommendation]:
    """Nearest neighbours of a catalog track, excluding the seed itself.

    Raises ``UnknownTrackError`` for an unindexed seed rather than returning an
    empty list, so the caller can tell "no such track" from "nothing similar".
    ``exclude`` removes further tracks (e.g. ones a filter rejected) before the
    ``k`` cut, so filtering never shortens a list that had enough candidates.
    """
    k = _validate_k(k)
    seed_vector = catalog.vector_of(track_id)
    scores = cosine_scores(seed_vector, catalog.vectors)
    return rank_by_score(
        catalog,
        scores,
        k=k,
        exclude=frozenset(exclude) | {track_id},
        max_per_artist=max_per_artist,
    )
