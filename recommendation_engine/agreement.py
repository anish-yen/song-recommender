"""Check attribute estimates against hand labels.

The README allows an attribute in a filter or reason only once it has been checked:
listen to 15-20 tracks, write down what you hear, and compare. No model is loaded;
this reads ``attributes.json`` and a labels file.

    python -m recommendation_engine.agreement --labels data/labels.json

Labels file (``null`` or a missing key means "not labelled"; keys starting with
``_`` are comments)::

    {"tracks": {"gymnopedie-no-1": {"vocals": "instrumental", "genre": "classical",
                                    "mood": ["calm"], "energy": "low"}}}

Mood counts as agreeing when any hand-labelled mood is among the predicted moods;
a track with no detected mood is reported as unknown.
Energy compares catalog-relative bands, so label it relative to *this* catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attributes import LABELS, load_attributes
from .errors import RecommenderError
from .filters import ENERGY_BANDS, energy_band

ATTRIBUTES = ("vocals", "genre", "mood", "energy")


def _allowed(labels: dict[str, list[str]]) -> dict[str, set[str]]:
    return {
        "vocals": set(labels["vocals"]),
        "genre": set(labels["genre"]),
        "mood": set(labels["mood"]),
        "energy": set(ENERGY_BANDS),
    }


def compare_with_labels(
    predicted: dict[str, dict[str, object]],
    labels: dict[str, dict[str, object]],
    vocabulary: dict[str, list[str]] = LABELS,
) -> dict[str, dict[str, list]]:
    """Per attribute: ``agree`` / ``disagree`` / ``unknown`` lists of track entries.

    A ``choice`` estimate of ``None`` is reported as ``unknown``, not as a
    disagreement: abstaining is the threshold doing its job, and is counted apart.
    """
    allowed = _allowed(vocabulary)
    missing = sorted(set(labels) - set(predicted))
    if missing:
        raise ValueError(f"labelled tracks not in the index: {', '.join(missing)}")

    report: dict[str, dict[str, list]] = {
        name: {"agree": [], "disagree": [], "unknown": []} for name in ATTRIBUTES
    }
    for track_id in sorted(labels):
        hand = labels[track_id]
        unknown_keys = set(hand) - set(ATTRIBUTES)
        if unknown_keys:
            raise ValueError(f"{track_id}: unknown label keys {sorted(unknown_keys)}")
        attrs = predicted[track_id]
        for name in ATTRIBUTES:
            wanted = hand.get(name)
            if wanted is None or wanted == []:
                continue
            wanted_set = set(wanted) if isinstance(wanted, list) else {wanted}
            bad = wanted_set - allowed[name]
            if bad:
                raise ValueError(f"{track_id}: {name} label(s) {sorted(bad)} are not "
                                 f"one of: {', '.join(sorted(allowed[name]))}")
            if name in ("vocals", "genre"):
                got = attrs[name]["label"]
                if got is None:
                    report[name]["unknown"].append((track_id, wanted, None))
                    continue
                agrees = got in wanted_set
            elif name == "mood":
                got = list(attrs.get("mood", {}).get("labels", []))
                if not got:
                    report[name]["unknown"].append((track_id, wanted, None))
                    continue
                agrees = bool(wanted_set & set(got))
            else:
                got = energy_band(attrs["energy"]["value"])
                agrees = got in wanted_set
            report[name]["agree" if agrees else "disagree"].append((track_id, wanted, got))
    return report


def read_labels(path) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tracks"), dict):
        raise ValueError(f"{path}: expected an object with a 'tracks' mapping")
    return {
        tid: {k: v for k, v in entry.items() if not k.startswith("_")}
        for tid, entry in payload["tracks"].items()
        if not tid.startswith("_") and isinstance(entry, dict)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recommendation_engine.agreement",
        description="Compare attributes.json with hand labels.",
    )
    parser.add_argument("--index-dir", default=".index")
    parser.add_argument("--labels", required=True)
    args = parser.parse_args(argv)

    from .index import load_index

    try:
        catalog = load_index(args.index_dir)
        report = compare_with_labels(
            load_attributes(args.index_dir, catalog), read_labels(args.labels)
        )
    except (RecommenderError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for name in ATTRIBUTES:
        entry = report[name]
        decided = len(entry["agree"]) + len(entry["disagree"])
        if not decided and not entry["unknown"]:
            print(f"{name:<7} no labels")
            continue
        rate = f"{len(entry['agree'])}/{decided} agree" if decided else "0 decided"
        print(f"{name:<7} {rate}, {len(entry['unknown'])} unknown")
        for track_id, wanted, got in entry["disagree"]:
            print(f"          {track_id}: you said {wanted}, model said {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
