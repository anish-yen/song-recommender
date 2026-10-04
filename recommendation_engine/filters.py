"""Hard filters over per-track attributes.

A filter keeps a track only when its attribute *positively* matches. A track whose
estimate is unknown (a label of ``None``) is excluded too, but counted separately
from mismatches, so a response can say "3 tracks hidden because their vocals
estimate is unknown" instead of silently shrinking the list.

Vocals, genre, and mood are model estimates and will occasionally be wrong; tempo
and energy are measurements (see ``attributes.py``). A track with no detected mood
is *unknown* for a mood filter, not a mismatch: the tagger did not say "not calm".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from .attributes import ESTIMATE_KIND, LABELS, MEASURED_KIND

#: Energy is a within-catalog percentile, so bands are catalog thirds, not
#: absolute loudness: "high" means "among the most energetic third of this catalog".
ENERGY_BANDS = ("low", "medium", "high")

#: Beat trackers are unreliable outside this range, and so are requests for it.
TEMPO_LIMITS = (30.0, 300.0)


def energy_band(percentile: float) -> str:
    if percentile < 1 / 3:
        return "low"
    if percentile < 2 / 3:
        return "medium"
    return "high"


@dataclasses.dataclass(frozen=True)
class AttributeFilters:
    """Requested attribute values; ``None`` means "do not filter on this"."""

    vocals: str | None = None
    genre: str | None = None
    mood: str | None = None
    energy: str | None = None
    tempo_min: float | None = None
    tempo_max: float | None = None

    def active(self) -> dict[str, object]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    def validate(self, labels: dict[str, list[str]] = LABELS) -> None:
        """Reject values the attributes can never contain, rather than matching nothing."""
        allowed = {
            "vocals": set(labels["vocals"]),
            "genre": set(labels["genre"]),
            "mood": set(labels["mood"]),
            "energy": set(ENERGY_BANDS),
        }
        for name in ("vocals", "genre", "mood", "energy"):
            value = getattr(self, name)
            if value is None:
                continue
            if not allowed[name]:
                raise ValueError(f"{name} filter is unavailable: no {name} estimates exist")
            if value not in allowed[name]:
                raise ValueError(
                    f"unknown {name} filter {value!r}; "
                    f"choose from: {', '.join(sorted(allowed[name]))}"
                )
        low, high = TEMPO_LIMITS
        for name in ("tempo_min", "tempo_max"):
            value = getattr(self, name)
            if value is not None and not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g} BPM, got {value}")
        if (self.tempo_min is not None and self.tempo_max is not None
                and self.tempo_min > self.tempo_max):
            raise ValueError(f"tempo_min {self.tempo_min} exceeds tempo_max {self.tempo_max}")


@dataclasses.dataclass(frozen=True)
class FilterOutcome:
    """Which tracks passed, and why the others were hidden."""

    #: track_id -> one evidence entry per filter the track passed.
    passed: dict[str, list[dict[str, object]]]
    mismatched: frozenset[str]
    unknown: frozenset[str]

    @property
    def excluded(self) -> frozenset[str]:
        return self.mismatched | self.unknown


def _check(name: str, wanted, attrs: dict) -> tuple[str, dict[str, object] | None]:
    """Return ``("pass" | "mismatch" | "unknown", evidence entry or None)``."""
    entry = attrs.get(name)
    if not isinstance(entry, dict):
        return "unknown", None
    if name == "vocals":
        label = entry.get("label")
        if label is None:
            return "unknown", None
        if label != wanted:
            return "mismatch", None
        return "pass", {"attribute": name, "value": wanted, "kind": ESTIMATE_KIND,
                        "vocal_score": entry["vocal_score"]}
    if name == "genre":
        label = entry.get("label")
        if label is None:
            return "unknown", None
        if label != wanted:
            return "mismatch", None
        return "pass", {"attribute": name, "value": wanted, "kind": ESTIMATE_KIND,
                        "score": entry["score"]}
    if name == "mood":
        labels = entry.get("labels")
        if not labels:
            return "unknown", None
        if wanted not in labels:
            return "mismatch", None
        return "pass", {"attribute": name, "value": wanted, "kind": ESTIMATE_KIND,
                        "score": entry["scores"][wanted]}
    if name == "energy":
        value = entry.get("value")
        if value is None:
            return "unknown", None
        if energy_band(value) != wanted:
            return "mismatch", None
        return "pass", {"attribute": name, "value": wanted, "kind": MEASURED_KIND,
                        "percentile": value}
    # tempo: ``wanted`` is the (min, max) range, either end open.
    bpm = entry.get("bpm")
    if bpm is None:
        return "unknown", None
    low, high = wanted
    if (low is not None and bpm < low) or (high is not None and bpm > high):
        return "mismatch", None
    return "pass", {"attribute": name, "value": [low, high], "kind": MEASURED_KIND, "bpm": bpm}


def apply_filters(
    track_ids: Iterable[str],
    attributes: dict[str, dict[str, object]],
    filters: AttributeFilters,
) -> FilterOutcome:
    """Evaluate ``filters`` for every track. A mismatch outranks an unknown."""
    filters.validate()
    wanted: dict[str, object] = {
        name: getattr(filters, name)
        for name in ("vocals", "genre", "mood", "energy")
        if getattr(filters, name) is not None
    }
    if filters.tempo_min is not None or filters.tempo_max is not None:
        wanted["tempo"] = (filters.tempo_min, filters.tempo_max)

    passed: dict[str, list[dict[str, object]]] = {}
    mismatched: set[str] = set()
    unknown: set[str] = set()
    for track_id in track_ids:
        attrs = attributes.get(track_id, {})
        verdicts = [_check(name, value, attrs) for name, value in wanted.items()]
        states = {state for state, _ in verdicts}
        if "mismatch" in states:
            mismatched.add(track_id)
        elif "unknown" in states:
            unknown.add(track_id)
        else:
            passed[track_id] = [entry for _, entry in verdicts if entry is not None]
    return FilterOutcome(passed, frozenset(mismatched), frozenset(unknown))
