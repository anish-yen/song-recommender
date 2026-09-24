"""The catalog manifest: the contract between ingestion and this engine.

Ingestion produces a manifest; this module is the only thing that parses it.
Optional provenance fields record the source and usage terms when ingestion supplies
them. Evaluation catalogs must supply both; the parser does not enforce that policy.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .errors import ManifestError

#: Fields a manifest entry must provide.
REQUIRED_FIELDS = ("track_id", "title", "artist", "audio_path")


@dataclasses.dataclass(frozen=True)
class Track:
    """One catalog entry. ``audio_path`` is resolved against the manifest's folder."""

    track_id: str
    title: str
    artist: str
    audio_path: Path
    #: Where the audio came from (dataset name, "self-recorded", ...). Optional
    #: in the schema but recorded whenever ingestion supplies it.
    source: str | None = None
    #: Licence or usage terms for this specific file.
    license: str | None = None
    #: A link the frontend can send listeners to.
    provider_url: str | None = None

    def metadata(self) -> dict[str, object]:
        """Serialisable form, used by the index and by API responses."""
        return {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "audio_path": str(self.audio_path),
            "source": self.source,
            "license": self.license,
            "provider_url": self.provider_url,
        }


def _require_str(entry: dict[str, object], field: str, where: str) -> str:
    value = entry.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ManifestError(f"{where}: field '{field}' is required and must be non-empty")
    if not isinstance(value, str):
        raise ManifestError(
            f"{where}: field '{field}' must be a string, got {type(value).__name__}"
        )
    return value.strip()


def _optional_str(entry: dict[str, object], field: str, where: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(
            f"{where}: field '{field}' must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    return stripped or None


def parse_tracks(entries: list[dict[str, object]], *, base_dir: Path) -> list[Track]:
    """Validate manifest entries into ``Track`` records.

    Raises on duplicate track ids and on audio files that do not exist: both are
    silent-corruption risks. A duplicate id would make similar-song lookups
    ambiguous, and a missing file would otherwise surface much later as a
    confusing decode error in the middle of a long index build.
    """
    tracks: list[Track] = []
    seen: dict[str, int] = {}
    for position, entry in enumerate(entries):
        where = f"manifest entry {position}"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where}: expected an object, got {type(entry).__name__}")

        track_id = _require_str(entry, "track_id", where)
        if track_id in seen:
            raise ManifestError(
                f"{where}: duplicate track_id {track_id!r} "
                f"(first seen at entry {seen[track_id]})"
            )
        seen[track_id] = position

        # JSON has no comment syntax, so underscore-prefixed keys are treated as
        # annotations and ignored. Everything else must be a field we know, so a
        # typo'd key is never silently dropped.
        supplied = {key for key in entry if not key.startswith("_")}
        unknown = supplied - set(REQUIRED_FIELDS) - {"source", "license", "provider_url"}
        if unknown:
            raise ManifestError(
                f"{where}: unknown field(s) {', '.join(sorted(unknown))}"
            )

        raw_path = Path(_require_str(entry, "audio_path", where))
        audio_path = raw_path if raw_path.is_absolute() else base_dir / raw_path
        if not audio_path.is_file():
            raise ManifestError(
                f"{where} ({track_id}): audio file not found at {audio_path}"
            )

        tracks.append(
            Track(
                track_id=track_id,
                title=_require_str(entry, "title", where),
                artist=_require_str(entry, "artist", where),
                audio_path=audio_path,
                source=_optional_str(entry, "source", where),
                license=_optional_str(entry, "license", where),
                provider_url=_optional_str(entry, "provider_url", where),
            )
        )

    if not tracks:
        raise ManifestError("manifest contains no tracks")
    return tracks


def load_manifest(path: str | Path) -> list[Track]:
    """Read a JSON manifest: either a bare list, or ``{"tracks": [...]}``."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc

    if isinstance(raw, dict):
        entries = raw.get("tracks")
        if entries is None:
            raise ManifestError(
                f"manifest {manifest_path}: object form must have a 'tracks' key"
            )
    else:
        entries = raw
    if not isinstance(entries, list):
        raise ManifestError(f"manifest {manifest_path}: 'tracks' must be a list")

    return parse_tracks(entries, base_dir=manifest_path.parent.resolve())
