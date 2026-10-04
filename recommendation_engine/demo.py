"""Command-line entry point for the recommender core.

    python -m recommendation_engine.demo build-index --manifest data/manifest.json
    python -m recommendation_engine.demo search "late-night beats" -k 5
    python -m recommendation_engine.demo similar <track_id> -k 5
    python -m recommendation_engine.demo search "study music" --vocals instrumental

Each result carries evidence (see ``evidence.py``) when ``attributes.json`` is
present: the similarity match, any filter that kept the track, model estimates
(vocals, genre), and measurements (tempo, energy).

This exists so the engine is demonstrable before the API and database land. The
FastAPI routes should call the same functions rather than shelling out to this.
"""

from __future__ import annotations

import argparse
import json
import sys

from .attributes import load_attributes
from .errors import IndexCorruptError, RecommenderError
from .evidence import (
    ExplainedResults,
    explained_search,
    explained_similar,
    short_reason,
    summarize,
)
from .filters import ENERGY_BANDS, AttributeFilters
from .index import build_index, load_index
from .manifest import load_manifest

DEFAULT_INDEX_DIR = ".index"


def _print_results(explained: ExplainedResults, *, as_json: bool, verbose: bool) -> None:
    results = explained.results
    if as_json:
        print(json.dumps({
            "results": [{**r.to_dict(), "reason": short_reason(r.evidence)} for r in results],
            "hidden_by_filters": {"mismatch": explained.hidden_mismatch,
                                  "unknown_estimate": explained.hidden_unknown},
        }, indent=2))
        return
    if explained.hidden_mismatch or explained.hidden_unknown:
        print(f"filters hid {explained.hidden_mismatch} track(s) that did not match and "
              f"{explained.hidden_unknown} whose estimate is unknown")
    if not results:
        print("no results")
        return
    width = max(len(r.track_id) for r in results)
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>2}. {result.similarity:+.4f}  "
            f"{result.track_id:<{width}}  {result.title} — {result.artist}"
        )
        lines = summarize(result.evidence) if verbose else [short_reason(result.evidence)]
        for line in lines:
            print(f"      {line}")


def _filters(args: argparse.Namespace) -> AttributeFilters:
    filters = AttributeFilters(
        vocals=args.vocals, genre=args.genre, mood=args.mood, energy=args.energy,
        tempo_min=args.tempo_min, tempo_max=args.tempo_max,
    )
    filters.validate()
    return filters


def _attributes(args: argparse.Namespace, catalog, filters: AttributeFilters):
    """Load estimates. Filters require them; plain queries degrade to match-only reasons."""
    try:
        return load_attributes(args.index_dir, catalog)
    except IndexCorruptError as exc:
        if filters.active():
            raise
        if not args.json:
            print(f"note: reasons limited to the similarity match ({exc})", file=sys.stderr)
        return None


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

    filters = _filters(args)
    catalog = load_index(args.index_dir)
    attributes = _attributes(args, catalog, filters)
    encoder = ClapEncoder(space=catalog.space, device=args.device)
    query_vector = encoder.embed_text([args.query])[0]
    explained = explained_search(
        catalog,
        query_vector,
        query_text=args.query,
        attributes=attributes,
        filters=filters,
        k=args.k,
        query_space=encoder.space,
        max_per_artist=args.max_per_artist,
    )
    _print_results(explained, as_json=args.json, verbose=args.verbose)
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    filters = _filters(args)
    catalog = load_index(args.index_dir)
    attributes = _attributes(args, catalog, filters)
    explained = explained_similar(
        catalog, args.track_id, attributes=attributes, filters=filters,
        k=args.k, max_per_artist=args.max_per_artist,
    )
    _print_results(explained, as_json=args.json, verbose=args.verbose)
    return 0


def _add_filter_flags(subparser: argparse.ArgumentParser) -> None:
    group = subparser.add_argument_group(
        "filters (tracks whose value is unknown are hidden and counted)"
    )
    group.add_argument("--vocals", help="vocal or instrumental (model estimate)")
    group.add_argument("--genre", help="e.g. jazz, electronic, country (model estimate)")
    group.add_argument("--mood", help="calm, sad, happy, energetic, angry, or dark "
                       "(model estimate; the weakest attribute)")
    group.add_argument("--energy", choices=ENERGY_BANDS,
                       help="catalog-relative third: low, medium, or high (measured)")
    group.add_argument("--tempo-min", type=float, help="lowest BPM to keep (measured)")
    group.add_argument("--tempo-max", type=float, help="highest BPM to keep (measured)")


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
    search_cmd.add_argument("-v", "--verbose", action="store_true",
                            help="print the full evidence instead of one reason line")
    _add_filter_flags(search_cmd)
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
    similar.add_argument("-v", "--verbose", action="store_true",
                         help="print the full evidence instead of one reason line")
    _add_filter_flags(similar)
    similar.set_defaults(func=cmd_similar)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RecommenderError, ValueError) as exc:
        # Engine errors are expected failure modes (bad manifest, no index, silent
        # audio, an unknown filter value); a traceback would bury the actionable message.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
