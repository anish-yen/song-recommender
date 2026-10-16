"""Persisting the catalog's vectors, and rebuilding them only when needed.

Storage is deliberately boring and inspectable: a plain ``.npy`` float array plus a
JSON sidecar. No pickle (loading one executes arbitrary code) and no vector
database — this index is a stand-in for the pgvector table that integration will
own, and keeping it dumb makes the eventual swap a data migration, not a rewrite.

Re-embedding a catalog is the expensive step, so a vector is reused only when both
the audio file's content hash and the embedding space's fingerprint are unchanged.
Either one changing means the stored number no longer describes the current input.

Each build is published as one immutable *generation*: a ``gen-*`` folder holding
``vectors.npy`` and an ``index.json`` that records the vectors file's sha256. A
one-line ``CURRENT`` pointer is then swapped in with ``os.replace``, which is
atomic, so a reader always sees a complete old generation or a complete new one —
never new vectors beside old metadata.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .config import EmbeddingSpace
from .errors import AudioError, EmbeddingError, IndexCorruptError, RecommenderError
from .manifest import Track
from .similarity import Catalog

#: Bumped when the on-disk layout changes incompatibly.
#: 2: generation folders behind an atomic ``CURRENT`` pointer, vectors checksum.
INDEX_VERSION = 2

VECTORS_FILE = "vectors.npy"
METADATA_FILE = "index.json"
POINTER_FILE = "CURRENT"
GENERATION_PREFIX = "gen-"
#: Generations kept after a publish: the new one plus its predecessor, so a reader
#: that resolved the old pointer just before the swap can still finish loading.
KEEP_GENERATIONS = 2
#: Persisted vectors are unit length; anything further off was not written by us.
UNIT_NORM_TOLERANCE = 1e-3

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
    """Publish ``vectors`` and ``entries`` as a new generation of ``index_dir``.

    Everything is validated and written into a fresh generation folder first; only
    then does the ``CURRENT`` pointer move. A failure at any earlier point leaves
    the previous generation as the one readers load.
    """
    if vectors.ndim != 2 or (vectors.shape[0] and vectors.shape[1] != space.dim):
        raise ValueError(
            f"vectors shape {vectors.shape} does not match space dim {space.dim}"
        )
    if vectors.shape[0] != len(entries):
        raise ValueError(
            f"{vectors.shape[0]} vectors for {len(entries)} entries"
        )
    matrix = vectors.astype(np.float32, copy=False)
    problem = _vector_problem(matrix)
    if problem:
        raise ValueError(f"refusing to save index: {problem}")

    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    generation = f"{GENERATION_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}"
    staging = directory / generation
    staging.mkdir()
    try:
        np.save(staging / VECTORS_FILE, matrix)
        payload = {
            "index_version": INDEX_VERSION,
            "space": space.to_dict(),
            "space_fingerprint": space.fingerprint(),
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "vectors_sha256": file_digest(staging / VECTORS_FILE),
            "tracks": entries,
        }
        (staging / METADATA_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        _write_pointer(directory, generation)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _prune_generations(directory, keep=generation)


def _write_pointer(directory: Path, generation: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".current-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(generation + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, directory / POINTER_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _prune_generations(directory: Path, *, keep: str) -> None:
    """Drop old generations, keeping ``keep`` and its newest predecessors."""
    older = sorted(
        (p for p in directory.iterdir()
         if p.is_dir() and p.name.startswith(GENERATION_PREFIX) and p.name != keep),
        key=lambda p: p.name,
    )
    for stale in older[: max(0, len(older) - (KEEP_GENERATIONS - 1))]:
        shutil.rmtree(stale, ignore_errors=True)


def current_generation(index_dir) -> Path:
    """The folder holding the published ``vectors.npy`` and ``index.json``."""
    directory = Path(index_dir)
    try:
        name = (directory / POINTER_FILE).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # No pointer: either nothing was built, or a pre-generation (v1) layout
        # whose index.json sits directly in the folder. read_index_metadata
        # reports both clearly.
        return directory
    if not name.startswith(GENERATION_PREFIX) or Path(name).name != name:
        raise IndexCorruptError(f"{directory / POINTER_FILE}: invalid generation {name!r}")
    return directory / name


def _vector_problem(matrix: np.ndarray) -> str | None:
    """Why ``matrix`` cannot be a persisted catalog, or ``None`` if it can."""
    if not np.all(np.isfinite(matrix)):
        rows = np.flatnonzero(~np.all(np.isfinite(matrix), axis=1))
        return f"non-finite values in row(s) {rows[:5].tolist()}"
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE)
    if bad.size:
        return f"row(s) {bad[:5].tolist()} are not unit length (norms {norms[bad[:5]].round(4).tolist()})"
    return None


def read_index_metadata(index_dir) -> dict[str, object]:
    """Load and sanity-check the current generation's ``index.json``."""
    path = current_generation(index_dir) / METADATA_FILE
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
    for key in ("space", "space_fingerprint", "vectors_sha256", "tracks"):
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
    directory = current_generation(index_dir)
    payload = read_index_metadata(index_dir)
    space: EmbeddingSpace = payload["space"]  # type: ignore[assignment]
    entries: list[dict[str, object]] = payload["tracks"]  # type: ignore[assignment]

    vectors_path = directory / VECTORS_FILE
    try:
        digest = file_digest(vectors_path)
    except AudioError as exc:  # file_digest's not-found error
        raise IndexCorruptError(f"{vectors_path} is missing; rebuild the index") from exc
    if digest != payload["vectors_sha256"]:
        raise IndexCorruptError(
            f"{vectors_path}: checksum does not match {METADATA_FILE}; "
            "the vectors belong to a different build — rebuild the index"
        )
    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except (ValueError, EOFError, OSError) as exc:
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
    problem = _vector_problem(vectors)
    if problem:
        raise IndexCorruptError(f"{vectors_path}: {problem}; rebuild the index")

    track_ids: list[str] = []
    metadata: dict[str, dict[str, object]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise IndexCorruptError(
                f"{directory}: entry {position} is {type(entry).__name__}, not an object"
            )
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
