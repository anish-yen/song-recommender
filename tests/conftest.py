"""Shared fixtures.

Nothing here imports torch or transformers: the default suite must run on a clean
checkout with no model weights, so inference is stood in for by ``FakeEncoder``.
The real checkpoint is exercised only by ``tests/test_real_model.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.similarity import Catalog

#: Small space for hand-checkable arithmetic. Using 4 dimensions instead of the
#: real 512 is what lets the cosine expectations below be written by hand.
SPACE_4D = EmbeddingSpace(dim=4, model_revision="test-revision")


@pytest.fixture
def space() -> EmbeddingSpace:
    return SPACE_4D


@dataclasses.dataclass(frozen=True)
class FakeEmbedding:
    """Structural stand-in for ``embeddings.AudioEmbedding``."""

    vector: np.ndarray
    excerpt_count: int


class FakeEncoder:
    """Deterministic encoder: the vector is a function of the file's bytes.

    Recording ``embedded_paths`` lets the cache tests assert on what was actually
    re-embedded, which is the behaviour that matters — not how many times a mock
    was called.
    """

    def __init__(self, space: EmbeddingSpace = SPACE_4D) -> None:
        self.space = space
        self.embedded_paths: list[str] = []

    def _vector_for(self, payload: bytes) -> np.ndarray:
        digest = hashlib.sha256(payload).digest()
        raw = np.frombuffer(digest[: self.space.dim * 4], dtype=np.uint32)
        vector = (raw.astype(np.float64) / np.iinfo(np.uint32).max) - 0.5
        vector = vector[: self.space.dim].astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_audio_file(self, path) -> FakeEmbedding:
        self.embedded_paths.append(str(path))
        return FakeEmbedding(
            vector=self._vector_for(Path(path).read_bytes()), excerpt_count=1
        )

    def embed_text(self, texts) -> np.ndarray:
        return np.stack([self._vector_for(text.encode("utf-8")) for text in texts])


@pytest.fixture
def fake_encoder(space: EmbeddingSpace) -> FakeEncoder:
    return FakeEncoder(space)


def make_catalog(
    entries: list[tuple[str, str, list[float]]],
    *,
    space: EmbeddingSpace = SPACE_4D,
) -> Catalog:
    """Build a catalog from ``(track_id, artist, vector)`` triples."""
    track_ids = tuple(entry[0] for entry in entries)
    metadata = {
        track_id: {
            "track_id": track_id,
            "title": f"Title {track_id}",
            "artist": artist,
            "provider_url": f"https://example.test/{track_id}",
        }
        for track_id, artist, _ in entries
    }
    vectors = (
        np.array([entry[2] for entry in entries], dtype=np.float32)
        if entries
        else np.zeros((0, space.dim), dtype=np.float32)
    )
    return Catalog(
        space=space, track_ids=track_ids, metadata=metadata, vectors=vectors
    )


@pytest.fixture
def catalog() -> Catalog:
    """Four tracks whose cosine similarities against ``[1,0,0,0]`` are exact.

    ``a`` is identical (1.0), ``b`` orthogonal (0.0), ``c`` at 45 degrees
    (1/sqrt(2)), ``d`` opposite (-1.0).
    """
    return make_catalog(
        [
            ("a", "Artist One", [1.0, 0.0, 0.0, 0.0]),
            ("b", "Artist Two", [0.0, 1.0, 0.0, 0.0]),
            ("c", "Artist One", [2**-0.5, 2**-0.5, 0.0, 0.0]),
            ("d", "Artist Three", [-1.0, 0.0, 0.0, 0.0]),
        ]
    )


def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> Path:
    """Write a float32 WAV (mono if 1-D, else one column per channel)."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def sine(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def write_manifest(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tracks": entries}, indent=2), encoding="utf-8")
    return path
