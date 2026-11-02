"""Judge query interpretation on the fixed evaluation sets.

    uv run python -m recommendation_engine.intent_eval --index-dir .index
    uv run python -m recommendation_engine.intent_eval --interpret-only

Two questions, answered separately:

1. **Recognition** (no index needed; CLAP encodes the labelled sounds on first use):
   does each activity query read as the energy band it was written for, does each
   everyday situation get a reading, and how much do the control queries (already
   sound descriptions) get read into?
2. **Ranking** (needs an index with ``attributes.json``): with interpretation on
   versus off, how do the top-k's mean energy percentile and the share of results
   with a measured tempo in the read range change?

The queries were written before the interpretation code ran (design doc: write the
tests before tuning), and labelled searches that copy them are dropped from
``intent_labels.json`` (``intent_teacher.lookalikes``). The second set names
situations (``accept``); the reader does not use those names, so whether the matched
search fits is printed for a person to judge.
Listener grading is still needed before shipping; these numbers say whether the
mapping does what it claims, not whether people like the results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .errors import RecommenderError
from .evidence import explained_search
from .filters import AttributeFilters
from .intents import QueryInterpreter, read_query

QUERIES_FILE = Path(__file__).with_name("intent_eval_queries.json")


def load_queries(path=QUERIES_FILE) -> list[dict[str, object]]:
    """Either format: ``expect`` (an energy band, first set) or ``accept`` (situation
    names, later sets; ``[]`` means nothing should be inferred)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [q for q in payload["queries"] if "accept" in q or q.get("expect") in ("high", "low", "none")]


def recognition(interpreter: QueryInterpreter, queries) -> list[dict[str, object]]:
    rows = []
    for item in queries:
        reading = interpreter.interpret(item["query"])
        group = _group(item)
        if group in ("high", "low", "none"):
            expect = item["expect"]
            correct = reading.energy == (None if expect == "none" else expect)
        elif group == "situation":
            expect = "/".join(item["accept"])
            correct = None  # judged by a person from similar_to
        else:
            expect = "nothing"
            correct = reading.energy is None and reading.sound is None
        rows.append({
            "query": item["query"], "expect": expect, "group": group, "correct": correct,
            "energy": reading.energy, "tempo": reading.tempo,
            "similar_to": reading.similar_to, "sound": reading.sound is not None,
        })
    return rows


def _group(item) -> str:
    if "accept" in item:
        return "situation" if item["accept"] else "control"
    return item["expect"]


def _top_stats(results, attributes, tempo) -> tuple[float, float | None]:
    energy = [attributes[r.track_id]["energy"]["value"] for r in results]
    in_range = None
    if tempo is not None:
        low, high = tempo
        bpms = [attributes[r.track_id]["tempo"]["bpm"] for r in results]
        known = [b for b in bpms if b is not None]
        if known:
            in_range = sum((low is None or b >= low) and (high is None or b <= high)
                           for b in known) / len(known)
    return float(np.mean(energy)) if energy else float("nan"), in_range


def ranking(catalog, attributes, encoder, interpreter, rows, k: int) -> None:
    def encode(text):
        return encoder.embed_text([text])[0]

    for row in rows:
        query = row["query"]
        plain = explained_search(catalog, encode(query), query_text=query,
                                 attributes=attributes, k=k, query_space=encoder.space)
        vector, applied = read_query(query, AttributeFilters(), encode=encode,
                                     interpreter=interpreter)
        mapped = explained_search(
            catalog, vector, query_text=query, attributes=attributes, k=k,
            query_space=encoder.space, boosts=applied.boosts,
            similar_to=applied.similar_to)
        tempo = row["tempo"]
        row["energy_off"], row["tempo_off"] = _top_stats(plain.results, attributes, tempo)
        row["energy_on"], row["tempo_on"] = _top_stats(mapped.results, attributes, tempo)
        before = {r.track_id for r in plain.results}
        row["overlap"] = sum(r.track_id in before for r in mapped.results)


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def _tempo(tempo) -> str:
    if tempo is None:
        return "-"
    low, high = tempo
    return f"{'' if low is None else f'{low:g}'}-{'' if high is None else f'{high:g}'}"


def report(rows, *, ranked: bool, k: int) -> str:
    lines = []
    header = f"{'query':<36} {'expect':<14} {'energy':<7} {'tempo':<8} ok  similar to"
    if ranked:
        header += f"   energy off->on   tempo-fit off->on  overlap/{k}"
    lines.append(header)
    for r in rows:
        ok = "?" if r["correct"] is None else "ok" if r["correct"] else "x"
        line = (f"{r['query']:<36} {r['expect']:<14} {r['energy'] or '-':<7} "
                f"{_tempo(r['tempo']):<8} {ok:<3} {r['similar_to'] or '-'}")
        if ranked:
            line += (f"   {_fmt(r["energy_off"])}->{_fmt(r['energy_on'])}"
                     f"     {_fmt(r["tempo_off"])}->{_fmt(r['tempo_on'])}     {r['overlap']}")
        lines.append(line)
    lines.append("")
    for group, label in (("high", "read as high energy"), ("low", "read as low energy"),
                         ("none", "read as no energy level")):
        members = [r for r in rows if r["group"] == group]
        if members:
            lines.append(f"{group:<9} {sum(r['correct'] for r in members)}/{len(members)} {label}")
            if ranked and group in ("high", "low"):
                off = np.mean([r["energy_off"] for r in members])
                on = np.mean([r["energy_on"] for r in members])
                lines.append(f"          mean top-{k} energy percentile {off:.2f} -> {on:.2f}")
    situations = [r for r in rows if r["group"] == "situation"]
    if situations:
        read = sum(r["sound"] for r in situations)
        lines.append(f"situation {read}/{len(situations)} given a reading "
                     "(whether each fits is for a person to judge)")
    controls = [r for r in rows if r["group"] == "control"]
    if controls:
        lines.append(f"control   {sum(r['correct'] for r in controls)}/{len(controls)} read into "
                     f"nothing; {sum(r['energy'] is not None for r in controls)} given an energy "
                     f"boost, {sum(r['sound'] for r in controls)} a sound")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m recommendation_engine.intent_eval",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--index-dir", default=".index")
    parser.add_argument("--queries", default=str(QUERIES_FILE))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="device for CLAP text encoding")
    parser.add_argument("--interpret-only", action="store_true",
                        help="recognition only; no index needed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from .config import DEFAULT_SPACE
    from .embeddings import ClapEncoder

    try:
        catalog = attributes = None
        space = DEFAULT_SPACE
        if not args.interpret_only:
            from .attributes import load_attributes
            from .index import load_index

            catalog = load_index(args.index_dir)
            attributes = load_attributes(args.index_dir, catalog)
            space = catalog.space
        encoder = ClapEncoder(space=space, device=args.device)
        interpreter = QueryInterpreter.load(encoder.embed_text, encoder.space.describe())
        rows = recognition(interpreter, load_queries(args.queries))
        if catalog is not None:
            ranking(catalog, attributes, encoder, interpreter, rows, args.k)
    except RecommenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(report(rows, ranked=not args.interpret_only, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
