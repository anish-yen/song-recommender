"""Command-line entry point for the recommender core.

    python -m recommendation_engine.demo build-index --manifest data/manifest.json
    python -m recommendation_engine.demo search "late-night beats" -k 5
    python -m recommendation_engine.demo similar <track_id> -k 5

This exists so the engine is demonstrable before the API and database land. The
FastAPI routes should call the same functions rather than shelling out to this.
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import RecommenderError
from .index import build_index, load_index
from .manifest import load_manifest
from .similarity import search, similar_to

DEFAULT_INDEX_DIR = ".index"


def _print_results(results, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        print("no results")
        return
    width = max(len(r.track_id) for r in results)
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>2}. {result.similarity:+.4f}  "
            f"{result.track_id:<{width}}  {result.title} — {result.artist}"
        )


def cmd_build_index(args: argparse.Namespace) -> int:
    from .embeddings import ClapEncoder

    tracks = load_manifest(args.manifest)
    encoder = ClapEncoder(device=args.device)
    print(f"embedding space: {encoder.space.describe()}")
    print(f"catalog: {len(tracks)} track(s) from {args.manifest}")

    catalog, report = build_index(
        tracks, encoder, args.index_dir, reuse=not args.force
    )
    print(f"index written to {args.index_dir}: {report.summary()}")
    for track_id, message in report.failed:
        print(f"  FAILED {track_id}: {message}", file=sys.stderr)
    print(f"searchable tracks: {len(catalog)}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .embeddings import ClapEncoder

    catalog = load_index(args.index_dir)
    encoder = ClapEncoder(space=catalog.space, device=args.device)
    query_vector = encoder.embed_text([args.query])[0]
    results = search(
        catalog,
        query_vector,
        k=args.k,
        query_space=encoder.space,
        max_per_artist=args.max_per_artist,
    )
    _print_results(results, as_json=args.json)
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    catalog = load_index(args.index_dir)
    results = similar_to(
        catalog, args.track_id, k=args.k, max_per_artist=args.max_per_artist
    )
    _print_results(results, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recommendation_engine.demo",
        description="Mood search and similar-song retrieval over CLAP embeddings.",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"where the embedding index lives (default: {DEFAULT_INDEX_DIR})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="embed a manifest's audio")
    build.add_argument("--manifest", required=True, help="path to the JSON manifest")
    build.add_argument(
        "--device",
        default="cpu",
        help="cpu (default), auto, mps, or cuda; 'auto' opts in to an accelerator",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="re-embed every track instead of reusing unchanged vectors",
    )
    build.set_defaults(func=cmd_build_index)

    search_cmd = subparsers.add_parser("search", help="rank tracks against a mood query")
    search_cmd.add_argument("query", help='e.g. "late-night beats without heavy vocals"')
    search_cmd.add_argument("-k", type=int, default=10, help="results to return")
    search_cmd.add_argument("--device", default="cpu", help="device for text encoding")
    search_cmd.add_argument(
        "--max-per-artist",
        type=int,
        default=None,
        help="cap results from any one artist",
    )
    search_cmd.add_argument("--json", action="store_true", help="emit JSON")
    search_cmd.set_defaults(func=cmd_search)

    similar = subparsers.add_parser("similar", help="find tracks like a seed track")
    similar.add_argument("track_id", help="a track_id already in the index")
    similar.add_argument("-k", type=int, default=10, help="results to return")
    similar.add_argument(
        "--max-per-artist",
        type=int,
        default=None,
        help="cap results from any one artist",
    )
    similar.add_argument("--json", action="store_true", help="emit JSON")
    similar.set_defaults(func=cmd_similar)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RecommenderError as exc:
        # Engine errors are expected failure modes (bad manifest, no index, silent
        # audio); a traceback would bury the actionable message.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
