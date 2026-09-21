"""Persisting the catalog's vectors, and rebuilding them only when needed.

Storage is deliberately boring and inspectable: a plain ``.npy`` float array plus a
JSON sidecar. No pickle (loading one executes arbitrary code) and no vector
database — this index is a stand-in for the pgvector table that integration will
own, and keeping it dumb makes the eventual swap a data migration, not a rewrite.

Re-embedding a catalog is the expensive step, so a vector is reused only when both
the audio file's content hash and the embedding space's fingerprint are unchanged.
Either one changing means the stored number no longer describes the current input.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import EmbeddingSpace
from .errors import AudioError, EmbeddingError, IndexCorruptError, RecommenderError
from .manifest import Track
from .similarity import Catalog

#: Bumped when the on-disk layout changes incompatibly.
INDEX_VERSION = 1

VECTORS_FILE = "vectors.npy"
METADATA_FILE = "index.json"

_HASH_CHUNK = 1 << 20


def file_digest(path) -> str:
    """sha256 of a file's bytes.

    Content hashing rather than mtime/size: re-encoding a file or replacing it with
    a different master keeps the size identical often enough that size would miss
    real changes, and touching a file would trigger pointless re-embedding.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise AudioError(f"{path}: audio file not found") from exc
    return digest.hexdigest()


@dataclasses.dataclass
class BuildReport:
    """What a build actually did — printed by the CLI, asserted on by tests."""

    embedded: list[str] = dataclasses.field(default_factory=list)
    reused: list[str] = dataclasses.field(default_factory=list)
    failed: list[tuple[str, str]] = dataclasses.field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.embedded)} embedded", f"{len(self.reused)} reused"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def save_index(
    index_dir,
    *,
    space: EmbeddingSpace,
    entries: list[dict[str, object]],
    vectors: np.ndarray,
) -> None:
    """Write ``vectors.npy`` and ``index.json`` into ``index_dir``."""
    if vectors.ndim != 2 or (vectors.shape[0] and vectors.shape[1] != space.dim):
        raise ValueError(
            f"vectors shape {vectors.shape} does not match space dim {space.dim}"
        )
    if vectors.shape[0] != len(entries):
        raise ValueError(
            f"{vectors.shape[0]} vectors for {len(entries)} entries"
        )

    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTORS_FILE, vectors.astype(np.float32, copy=False))
    payload = {
        "index_version": INDEX_VERSION,
        "space": space.to_dict(),
        "space_fingerprint": space.fingerprint(),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tracks": entries,
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def read_index_metadata(index_dir) -> dict[str, object]:
    """Load and sanity-check ``index.json``."""
    path = Path(index_dir) / METADATA_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IndexCorruptError(
            f"no index at {path}; build one first with 'build-index'"
        ) from exc
    except json.JSONDecodeError as exc:
        raise IndexCorruptError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise IndexCorruptError(f"{path}: expected a JSON object")
    version = payload.get("index_version")
    if version != INDEX_VERSION:
        raise IndexCorruptError(
            f"{path}: index_version {version!r} is not supported "
            f"(this build writes {INDEX_VERSION}); rebuild the index"
        )
    for key in ("space", "space_fingerprint", "tracks"):
        if key not in payload:
            raise IndexCorruptError(f"{path}: missing required key {key!r}")
    if not isinstance(payload["tracks"], list):
        raise IndexCorruptError(f"{path}: 'tracks' must be a list")

    try:
        space = EmbeddingSpace.from_dict(dict(payload["space"]))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IndexCorruptError(f"{path}: unusable embedding space ({exc})") from exc
    if space.fingerprint() != payload["space_fingerprint"]:
        raise IndexCorruptError(
            f"{path}: recorded fingerprint does not match the recorded space; "
            "the file has been edited by hand — rebuild the index"
        )
    payload["space"] = space
    return payload


def load_index(index_dir) -> Catalog:
    """Load a persisted index into a searchable ``Catalog``."""
    directory = Path(index_dir)
    payload = read_index_metadata(directory)
    space: EmbeddingSpace = payload["space"]  # type: ignore[assignment]
    entries: list[dict[str, object]] = payload["tracks"]  # type: ignore[assignment]

    vectors_path = directory / VECTORS_FILE
    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise IndexCorruptError(f"{vectors_path} is missing; rebuild the index") from exc
    except ValueError as exc:
        raise IndexCorruptError(f"{vectors_path} is unreadable: {exc}") from exc

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise IndexCorruptError(
            f"{vectors_path}: expected a 2-D array, got shape {vectors.shape}"
        )
    if vectors.shape[0] != len(entries):
        raise IndexCorruptError(
            f"{directory}: {vectors.shape[0]} vectors but {len(entries)} track entries"
        )

    track_ids: list[str] = []
    metadata: dict[str, dict[str, object]] = {}
    for position, entry in enumerate(entries):
        track_id = entry.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            raise IndexCorruptError(f"{directory}: entry {position} has no track_id")
        if track_id in metadata:
            raise IndexCorruptError(f"{directory}: duplicate track_id {track_id!r}")
        track_ids.append(track_id)
        metadata[track_id] = dict(entry)

    try:
        return Catalog(
            space=space,
            track_ids=tuple(track_ids),
            metadata=metadata,
            vectors=vectors,
        )
    except ValueError as exc:
        raise IndexCorruptError(f"{directory}: {exc}") from exc


def _reusable_vectors(index_dir, space: EmbeddingSpace) -> dict[str, tuple[str, np.ndarray]]:
    """Map ``track_id -> (audio_sha256, vector)`` from an existing compatible index.

    Returns nothing when there is no index, when it is unreadable, or when it was
    built in a different embedding space — in every one of those cases the stored
    vectors cannot be trusted, and a full rebuild is the correct response.
    """
    try:
        catalog = load_index(index_dir)
    except IndexCorruptError:
        return {}
    if catalog.space.fingerprint() != space.fingerprint():
        return {}

    reusable: dict[str, tuple[str, np.ndarray]] = {}
    for position, track_id in enumerate(catalog.track_ids):
        digest = catalog.metadata[track_id].get("audio_sha256")
        if isinstance(digest, str) and digest:
            reusable[track_id] = (digest, catalog.vectors[position])
    return reusable


def build_index(
    tracks: list[Track],
    encoder,
    index_dir,
    *,
    reuse: bool = True,
) -> tuple[Catalog, BuildReport]:
    """Embed ``tracks`` (reusing unchanged vectors) and persist the result.

    A track whose audio cannot be embedded is reported in ``BuildReport.failed``
    and left out of the index. It is never stored with a placeholder vector: a
    fabricated vector would quietly pollute every future search.
    """
    space: EmbeddingSpace = encoder.space
    cached = _reusable_vectors(index_dir, space) if reuse else {}

    report = BuildReport()
    entries: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []

    for track in tracks:
        try:
            digest = file_digest(track.audio_path)
        except AudioError as exc:
            report.failed.append((track.track_id, str(exc)))
            continue

        hit = cached.get(track.track_id)
        if hit is not None and hit[0] == digest:
            entry = track.metadata()
            entry["audio_sha256"] = digest
            entries.append(entry)
            vectors.append(np.asarray(hit[1], dtype=np.float32))
            report.reused.append(track.track_id)
            continue

        try:
            embedding = encoder.embed_audio_file(track.audio_path)
        except (AudioError, EmbeddingError) as exc:
            report.failed.append((track.track_id, str(exc)))
            continue

        entry = track.metadata()
        entry["audio_sha256"] = digest
        entry["excerpt_count"] = embedding.excerpt_count
        entries.append(entry)
        vectors.append(np.asarray(embedding.vector, dtype=np.float32))
        report.embedded.append(track.track_id)

    if not entries:
        detail = "; ".join(f"{tid}: {msg}" for tid, msg in report.failed) or "no tracks"
        raise RecommenderError(f"index build produced no usable vectors ({detail})")

    matrix = np.stack(vectors).astype(np.float32)
    save_index(index_dir, space=space, entries=entries, vectors=matrix)
    return load_index(index_dir), report
