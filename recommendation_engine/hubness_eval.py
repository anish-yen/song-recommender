"""Does hub correction cut cross-query repeats, and what does it cost?

Runs the real service path (filters off, artist cap 2, MMR on, as the discover page
does) at several hub weights and prints, per weight:

* mood search over ``hubness_eval_queries.json``: how many top-k lists the most
  repeated track appears in, the top repeaters, distinct tracks across all lists,
  and mean pairwise overlap between lists;
* similar songs over a fixed sample of seeds: the same repetition numbers, plus how
  often a result shares the seed's FMA genre (a relevance proxy that needs no
  listening) when the catalog records ``genre_top`` in each track's ``source``;
* cost: the mean raw cosine of the results and the deepest raw-cosine rank reached.

This measures repetition and a proxy, not quality. A weight earns its place only if
the graded listening check (EVALUATION.md) is no worse with it on.

    uv run python -m recommendation_engine.hubness --index-dir .index
    uv run python -m recommendation_engine.hubness_eval --index-dir .index
    uv run python -m recommendation_engine.hubness_eval --index-dir .index --no-text
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .hubness import load_hubness
from .similarity import cosine_scores

DEFAULT_QUERIES = Path(__file__).with_name("hubness_eval_queries.json")
_GENRE = re.compile(r"genre_top=([^;]+)")


def _queries(path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("queries") if isinstance(payload, dict) else payload
    return [q["query"] if isinstance(q, dict) else q for q in items]


def _genre(meta: dict) -> str | None:
    found = _GENRE.search(str(meta.get("source", "")))
    return found.group(1).strip() if found else None


def _raw_rank(scores: np.ndarray, ids: tuple[str, ...], track_id: str, skip: str | None) -> int:
    """1-based rank of ``track_id`` by raw cosine, ignoring ``skip`` (the seed)."""
    i = ids.index(track_id)
    better = scores > scores[i]
    if skip is not None:
        better[ids.index(skip)] = False
    return int(better.sum()) + 1


def _repetition(lists: list[list[str]], catalog, top: int = 5) -> dict[str, object]:
    counts = Counter(t for ids in lists for t in ids)
    overlaps = [len(set(a) & set(b)) / max(1, min(len(a), len(b)))
                for a, b in itertools.combinations(lists, 2)]
    return {
        "lists": len(lists),
        "max_appearances": max(counts.values()) if counts else 0,
        "top_repeaters": [
            f"{catalog.metadata[t].get('title', t)} — {catalog.metadata[t].get('artist', '')}: {n}"
            for t, n in counts.most_common(top)],
        "distinct_tracks": len(counts),
        "mean_pairwise_overlap": round(float(np.mean(overlaps)), 4) if overlaps else 0.0,
    }


def run(rec, *, weights, queries, seeds, k, max_per_artist=2) -> list[dict[str, object]]:
    catalog = rec.catalog
    ids = catalog.track_ids
    query_vectors = {q: rec.encode(q) for q in queries} if queries else {}
    report = []
    for weight in weights:
        rec.hub_weight = weight
        row: dict[str, object] = {"weight": weight}

        if queries:
            lists, cosines, depth = [], [], []
            for q in queries:
                answer = rec.recommend(query=q, k=k, max_per_artist=max_per_artist)
                scores = cosine_scores(query_vectors[q], catalog.vectors)
                lists.append([r.track_id for r in answer.results])
                cosines += [r.similarities["query"] for r in answer.results]
                depth += [_raw_rank(scores, ids, r.track_id, None) for r in answer.results]
            row["mood"] = {**_repetition(lists, catalog),
                           "mean_cosine": round(float(np.mean(cosines)), 4),
                           "median_raw_rank": float(np.median(depth)),
                           "deepest_raw_rank": int(max(depth))}

        if seeds:
            lists, cosines, depth, same_genre, judged = [], [], [], 0, 0
            for seed in seeds:
                answer = rec.recommend(seed_id=seed, k=k, max_per_artist=max_per_artist)
                scores = cosine_scores(catalog.vector_of(seed), catalog.vectors)
                lists.append([r.track_id for r in answer.results])
                seed_genre = _genre(catalog.metadata[seed])
                for r in answer.results:
                    cosines.append(r.similarities["seed"])
                    depth.append(_raw_rank(scores, ids, r.track_id, seed))
                    genre = _genre(catalog.metadata[r.track_id])
                    if seed_genre and genre:
                        judged += 1
                        same_genre += genre == seed_genre
            row["similar"] = {**_repetition(lists, catalog),
                              "mean_cosine": round(float(np.mean(cosines)), 4),
                              "median_raw_rank": float(np.median(depth)),
                              "deepest_raw_rank": int(max(depth)),
                              "same_genre_share": round(same_genre / judged, 4) if judged else None}
        report.append(row)
    return report


def _print(report: list[dict[str, object]]) -> None:
    for section in ("mood", "similar"):
        rows = [r for r in report if section in r]
        if not rows:
            continue
        print(f"\n== {section} ({rows[0][section]['lists']} lists)")
        keys = [k for k in rows[0][section] if k not in ("lists", "top_repeaters")]
        print("weight  " + "  ".join(f"{k:>20}" for k in keys))
        for r in rows:
            print(f"{r['weight']:<6}  " + "  ".join(f"{str(r[section][k]):>20}" for k in keys))
        for r in rows:
            print(f"  weight {r['weight']} top repeaters: " + "; ".join(r[section]["top_repeaters"]))


def main(argv: list[str] | None = None) -> int:
    from api.service import Recommender
    from recommendation_engine.attributes import load_attributes
    from recommendation_engine.errors import IndexCorruptError
    from recommendation_engine.index import load_index

    parser = argparse.ArgumentParser(prog="python -m recommendation_engine.hubness_eval")
    parser.add_argument("--index-dir", default=".index")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--weights", default="0,0.25,0.5,0.75")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=500,
                        help="seed tracks for the similar path, evenly spaced through the catalog")
    parser.add_argument("--no-text", action="store_true", help="skip mood search (no model load)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args(argv)

    catalog = load_index(args.index_dir)
    hubness = load_hubness(args.index_dir, catalog)  # fail loudly if not built
    try:
        attributes = load_attributes(args.index_dir, catalog)
    except IndexCorruptError:
        attributes = None
    encoder = None
    if not args.no_text:
        if hubness.text is None:
            parser.error("hubness.json has no text penalties; rebuild without --audio-only "
                         "or pass --no-text")
        from recommendation_engine.embeddings import ClapEncoder
        encoder = ClapEncoder(space=catalog.space, device=args.device)

    rec = Recommender(catalog, attributes, encoder, hubness=hubness, hub_weight=0.0)
    step = max(1, len(catalog) // max(1, args.seeds))
    seeds = list(catalog.track_ids[::step][:args.seeds]) if args.seeds else []
    weights = [float(w) for w in args.weights.split(",")]
    report = run(rec, weights=weights, queries=[] if args.no_text else _queries(args.queries),
                 seeds=seeds, k=args.k)
    _print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
