"""The design doc's 20 seed-song tests, as a grading sheet.

Picks the seeds (or reads a list you wrote down before looking at any results),
runs similar-song retrieval for each, and writes one CSV row per result with an
empty ``grade`` column for a listener to fill with good / acceptable / wrong
(EVALUATION.md, step 4). It also prints how often a result shares its seed's FMA
genre, next to the rate chance would give — a sanity check, not a quality grade.

    uv run python -m recommendation_engine.seed_eval --pick 20 -k 5 --out seed_grades.csv
    uv run python -m recommendation_engine.seed_eval --seeds data/seeds.json --backend pgvector

``data/seeds.json`` is ``["fma-000002", ...]`` or ``{"seeds": [...]}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from .errors import IndexCorruptError, RecommenderError
from .evidence import explained_similar, short_reason
from .similarity import Catalog, similar_to

COLUMNS = ["seed_id", "seed_title", "seed_artist", "seed_genre", "rank", "track_id",
           "title", "artist", "genre", "similarity", "why", "grade", "notes"]

_GENRE = re.compile(r"genre_top=([^;]+)")


def dataset_genre(meta: dict[str, object]) -> str | None:
    """The FMA ``genre_top`` the ingestion step records in ``source``, if any.

    This is the dataset's own label, not a model estimate, which is why it can be
    used to sanity-check retrieval.
    """
    match = _GENRE.search(str(meta.get("source") or ""))
    return match.group(1).strip() if match else None


def read_seeds(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = payload.get("seeds") if isinstance(payload, dict) else payload
    if not isinstance(seeds, list) or not all(isinstance(s, str) for s in seeds):
        raise ValueError(f"{path}: expected a list of track ids or {{\"seeds\": [...]}}")
    return seeds


def genre_agreement(catalog: Catalog, rows: list[dict[str, object]]) -> tuple[float, float] | None:
    """(observed share of results in the seed's genre, share expected by chance).

    Chance is the probability that a random *other* catalog track has the seed's
    genre, averaged over the seeds actually graded. ``None`` without genre labels.
    """
    counts = Counter(g for g in (dataset_genre(m) for m in catalog.metadata.values()) if g)
    labelled = [r for r in rows if r["seed_genre"] and r["genre"]]
    if not labelled or not counts:
        return None
    total = sum(counts.values())
    observed = sum(r["seed_genre"] == r["genre"] for r in labelled) / len(labelled)
    chance = sum((counts[r["seed_genre"]] - 1) / (total - 1) for r in labelled) / len(labelled)
    return observed, chance


def sheet_rows(catalog: Catalog, seeds: list[str], *, k: int, attributes=None,
               neighbours=similar_to, max_per_artist: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        catalog.position_of(seed)  # UnknownTrackError names the bad seed
        seed_meta = catalog.metadata[seed]
        out = explained_similar(catalog, seed, attributes=attributes, k=k,
                                max_per_artist=max_per_artist, neighbours=neighbours)
        for rank, rec in enumerate(out.results, start=1):
            meta = catalog.metadata[rec.track_id]
            rows.append({
                "seed_id": seed, "seed_title": seed_meta.get("title", ""),
                "seed_artist": seed_meta.get("artist", ""),
                "seed_genre": dataset_genre(seed_meta) or "",
                "rank": rank, "track_id": rec.track_id, "title": rec.title,
                "artist": rec.artist, "genre": dataset_genre(meta) or "",
                "similarity": f"{rec.similarity:.4f}", "why": short_reason(rec.evidence),
                "grade": "", "notes": "",
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    from .index import load_index
    from .pgvector_store import PgVectorStore, pick_seeds

    parser = argparse.ArgumentParser(prog="python -m recommendation_engine.seed_eval",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--index-dir", default=os.environ.get("INDEX_DIR", ".index"))
    chosen = parser.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--seeds", type=Path, help="JSON list of seed track ids")
    chosen.add_argument("--pick", type=int, help="pick this many seeds spread over the index")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-per-artist", type=int, default=2,
                        help="artist cap, as the product endpoint uses (0 = none)")
    parser.add_argument("--backend", choices=["numpy", "pgvector"], default="numpy")
    parser.add_argument("--dsn", help="Postgres URL for --backend pgvector ($PGVECTOR_URL)")
    parser.add_argument("--out", type=Path, default=Path("seed_grades.csv"))
    args = parser.parse_args(argv)

    try:
        catalog = load_index(args.index_dir)
        try:
            from .attributes import load_attributes
            attributes = load_attributes(args.index_dir, catalog)
        except IndexCorruptError:
            attributes = None  # reasons fall back to the similarity match alone
        seeds = read_seeds(args.seeds) if args.seeds else pick_seeds(catalog, args.pick)
        neighbours = similar_to
        if args.backend == "pgvector":
            store = PgVectorStore.from_url(args.dsn or os.environ.get("PGVECTOR_URL", ""),
                                           catalog.space)
            store.verify_matches(catalog)
            neighbours = store.similar_to
        rows = sheet_rows(catalog, seeds, k=args.k, attributes=attributes,
                          neighbours=neighbours, max_per_artist=args.max_per_artist or None)
    except (RecommenderError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len(seeds)} seeds to {args.out}; "
          "fill the grade column with good / acceptable / wrong")
    agreement = genre_agreement(catalog, rows)
    if agreement:
        observed, chance = agreement
        print(f"results sharing the seed's FMA genre: {observed:.0%} (chance: {chance:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
