"""Write an engine manifest from the FMA dataset (github.com/mdeff/fma).

    uv run python -m ingestion.fma --metadata data/audio/fma/fma_metadata \\
        --audio data/audio/fma/fma_small --per-genre 5 --out data/manifest.json

Sampling is deterministic: within each top-level genre, the lowest track ids whose
audio file exists are taken first. ``--per-genre 0`` keeps every available track.
FMA metadata is CC BY 4.0; each clip keeps the license its artist chose, which is
copied into the manifest's ``license`` field.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

SOURCE = "FMA: A Dataset For Music Analysis (Defferrard et al., ISMIR 2017)"


def read_tracks(tracks_csv: Path) -> dict[int, dict[str, str]]:
    """``tracks.csv`` has a three-row header: group, field, then ``track_id``.

    Columns are returned as ``"group.field"`` keys, e.g. ``"track.genre_top"``.
    """
    with open(tracks_csv, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        groups, fields, _ = next(reader), next(reader), next(reader)
        keys = [f"{g}.{f}" for g, f in zip(groups, fields)]
        return {int(row[0]): dict(zip(keys[1:], row[1:])) for row in reader if row}


def read_track_urls(raw_tracks_csv: Path) -> dict[int, str]:
    """FMA page per track, from ``raw_tracks.csv`` (absent -> no provider links)."""
    if not raw_tracks_csv.exists():
        return {}
    with open(raw_tracks_csv, newline="", encoding="utf-8") as handle:
        return {
            int(r["track_id"]): r["track_url"]
            for r in csv.DictReader(handle)
            if r.get("track_url", "").startswith("http")
        }


def audio_path(audio_root: Path, track_id: int) -> Path:
    """FMA stores clips as ``<first three digits>/<six-digit id>.mp3``."""
    stem = f"{track_id:06d}"
    return audio_root / stem[:3] / f"{stem}.mp3"


def build_entries(
    tracks: dict[int, dict[str, str]],
    urls: dict[int, str],
    *,
    audio_root: Path,
    manifest_dir: Path,
    subset: str = "small",
    per_genre: int = 5,
) -> tuple[list[dict[str, str]], int]:
    """Manifest entries grouped by genre, plus how many listed clips were missing."""
    by_genre: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing = 0
    for track_id in sorted(tracks):
        meta = tracks[track_id]
        genre = meta.get("track.genre_top", "")
        if meta.get("set.subset") != subset or not genre:
            continue
        if per_genre and len(by_genre[genre]) >= per_genre:
            continue
        audio = audio_path(audio_root, track_id).resolve()
        if not audio.exists():
            missing += 1
            continue
        entry = {
            "track_id": f"fma-{track_id:06d}",
            "title": meta.get("track.title", "").strip() or f"FMA track {track_id}",
            "artist": meta.get("artist.name", "").strip() or "Unknown artist",
            "audio_path": os.path.relpath(audio, manifest_dir.resolve()),
            "source": f"{SOURCE}; genre_top={genre}",
            "license": meta.get("track.license", "").strip() or "unknown (see FMA)",
        }
        if track_id in urls:
            entry["provider_url"] = urls[track_id]
        by_genre[genre].append(entry)
    return [e for genre in sorted(by_genre) for e in by_genre[genre]], missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingestion.fma", description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True, type=Path, help="unpacked fma_metadata/")
    parser.add_argument("--audio", required=True, type=Path, help="unpacked fma_small/ (or medium...)")
    parser.add_argument("--subset", default="small", help="small, medium, or large")
    parser.add_argument("--per-genre", type=int, default=5, help="tracks per genre; 0 = all")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    entries, missing = build_entries(
        read_tracks(args.metadata / "tracks.csv"),
        read_track_urls(args.metadata / "raw_tracks.csv"),
        audio_root=args.audio, manifest_dir=args.out.parent,
        subset=args.subset, per_genre=args.per_genre,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"tracks": entries}, indent=2, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry["source"].rsplit("=", 1)[1]] += 1
    summary = ", ".join(f"{g}: {n}" for g, n in sorted(counts.items()))
    print(f"{len(entries)} tracks -> {args.out} ({summary}); {missing} listed but missing audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
