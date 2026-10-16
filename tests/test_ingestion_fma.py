"""FMA → manifest conversion: header parsing, subset/genre sampling, missing audio."""

from __future__ import annotations

import csv

from ingestion.fma import build_entries, read_tracks
from recommendation_engine.manifest import parse_tracks


def write_tracks_csv(path, rows):
    """Mimic FMA's three-row header (group, field, track_id)."""
    columns = [("artist", "name"), ("set", "subset"), ("track", "genre_top"),
               ("track", "license"), ("track", "title")]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [g for g, _ in columns])
        writer.writerow([""] + [f for _, f in columns])
        writer.writerow(["track_id"] + [""] * len(columns))
        for track_id, artist, subset, genre, title in rows:
            writer.writerow([track_id, artist, subset, genre, "CC BY 4.0", title])


def touch_clip(root, track_id):
    stem = f"{track_id:06d}"
    path = root / stem[:3] / f"{stem}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really audio")


def test_samples_lowest_ids_per_genre_within_subset(tmp_path):
    write_tracks_csv(tmp_path / "tracks.csv", [
        (2, "AWOL", "small", "Hip-Hop", "Food"),
        (3, "AWOL", "small", "Hip-Hop", "Electric Ave"),
        (5, "AWOL", "small", "Hip-Hop", "This World"),
        (10, "Kurt Vile", "small", "Pop", "Freeway"),
        (20, "Big Band", "medium", "Pop", "Not in small"),
        (30, "Untagged", "small", "", "No genre"),
        (40, "Gone", "small", "Pop", "Clip missing"),
    ])
    audio = tmp_path / "fma_small"
    for track_id in (2, 3, 5, 10, 20, 30):
        touch_clip(audio, track_id)

    entries, missing = build_entries(
        read_tracks(tmp_path / "tracks.csv"), {2: "https://freemusicarchive.org/x"},
        audio_root=audio, manifest_dir=tmp_path, per_genre=2,
    )

    assert [e["track_id"] for e in entries] == ["fma-000002", "fma-000003", "fma-000010"]
    assert missing == 1
    assert entries[0]["provider_url"] == "https://freemusicarchive.org/x"
    assert "provider_url" not in entries[1]
    assert entries[0]["audio_path"] == "fma_small/000/000002.mp3"
    assert entries[2]["source"].endswith("genre_top=Pop")


def test_entries_pass_the_engine_manifest_validation(tmp_path):
    write_tracks_csv(tmp_path / "tracks.csv", [(7, "Artist", "small", "Rock", "Song")])
    touch_clip(tmp_path / "fma_small", 7)
    entries, _ = build_entries(
        read_tracks(tmp_path / "tracks.csv"), {},
        audio_root=tmp_path / "fma_small", manifest_dir=tmp_path, per_genre=0,
    )
    tracks = parse_tracks(entries, base_dir=tmp_path)
    assert tracks[0].track_id == "fma-000007"
    assert tracks[0].license == "CC BY 4.0"
