"""The seed-song grading sheet: genre labels, the chance baseline, and sheet shape."""

from __future__ import annotations

import json

import pytest

from recommendation_engine.seed_eval import dataset_genre, genre_agreement, read_seeds, sheet_rows

from .conftest import make_catalog


def fma_catalog():
    catalog = make_catalog([
        ("r1", "A", [1.0, 0.0, 0.0, 0.0]),
        ("r2", "B", [0.9, 0.1, 0.0, 0.0]),
        ("r3", "C", [0.8, 0.2, 0.0, 0.0]),
        ("j1", "D", [0.0, 1.0, 0.0, 0.0]),
        ("j2", "E", [0.1, 0.9, 0.0, 0.0]),
    ])
    for track_id, meta in catalog.metadata.items():
        genre = "Rock" if track_id.startswith("r") else "Jazz"
        meta["source"] = f"FMA (fma_small); genre_top={genre}"
    return catalog


def test_dataset_genre_reads_the_ingestion_label():
    assert dataset_genre({"source": "FMA (fma_small); genre_top=Hip-Hop"}) == "Hip-Hop"
    assert dataset_genre({"source": "self-recorded"}) is None
    assert dataset_genre({}) is None


def test_sheet_has_k_ungraded_rows_per_seed_in_rank_order():
    rows = sheet_rows(fma_catalog(), ["r1", "j1"], k=2)
    assert [(r["seed_id"], r["rank"], r["track_id"]) for r in rows] == [
        ("r1", 1, "r2"), ("r1", 2, "r3"), ("j1", 1, "j2"), ("j1", 2, "r3")]
    assert all(r["grade"] == "" and r["why"].startswith("Sounds like") for r in rows)


def test_genre_agreement_against_hand_computed_chance():
    catalog = fma_catalog()
    rows = sheet_rows(catalog, ["r1", "j1"], k=2)
    observed, chance = genre_agreement(catalog, rows)
    # r1 -> r2, r3 (both Rock); j1 -> j2 (Jazz), r3 (Rock): 3 of 4 share the seed's genre.
    assert observed == pytest.approx(3 / 4)
    # Another random track shares Rock with p = 2/4, Jazz with p = 1/4; two rows each.
    assert chance == pytest.approx((2 * 2 / 4 + 2 * 1 / 4) / 4)


def test_no_genre_labels_means_no_agreement_figure():
    catalog = make_catalog([("a", "x", [1.0, 0.0, 0.0, 0.0]), ("b", "y", [0.0, 1.0, 0.0, 0.0])])
    assert genre_agreement(catalog, sheet_rows(catalog, ["a"], k=1)) is None


def test_read_seeds_accepts_a_list_or_an_object(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(["x", "y"]))
    (tmp_path / "b.json").write_text(json.dumps({"seeds": ["z"]}))
    (tmp_path / "c.json").write_text(json.dumps({"seeds": [1]}))
    assert read_seeds(tmp_path / "a.json") == ["x", "y"]
    assert read_seeds(tmp_path / "b.json") == ["z"]
    with pytest.raises(ValueError):
        read_seeds(tmp_path / "c.json")
