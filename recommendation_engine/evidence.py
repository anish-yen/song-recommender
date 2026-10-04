"""Evidence-based reasons for each recommendation.

Evidence is structured data, not prose, so the wording a frontend shows can never
drift from what was measured. ``short_reason`` is the one-line wording for a result
list; ``summarize`` is the full version.

Rules, from the project's README:

1. ``match`` is always present and always true: the result's rank and cosine
   similarity against the query or seed.
2. An attribute is a *reason* (``filters_passed``) only when a filter the user
   applied actually kept the track.
3. Agreement is reported as agreement, never as the cause of the ranking:
   ``query_agreement`` when a word in the search names a value the tagger also
   found, ``shared_with_seed`` when a similar song shares the seed's attributes.
   Both come from AST and measurement, not from the CLAP search embedding.
4. Model estimates (vocals, genre, mood; ``kind: ast-audioset``) are worded as
   estimates. Measurements (tempo, energy; ``kind: measured``) are stated as facts.
   Unknown values are omitted, never guessed.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from .attributes import ESTIMATE_KIND, LABELS, MEASURED_KIND
from .config import EmbeddingSpace
from .errors import RecommenderError
from .filters import AttributeFilters, FilterOutcome, apply_filters, energy_band
from .similarity import Catalog, Recommendation, search, similar_to

#: Two tempos within this fraction of the seed's count as "similar tempo".
TEMPO_CLOSE = 0.08

#: Search words that name an attribute value: every label, plus a few unambiguous
#: synonyms. Matching is whole-word and case-insensitive; nothing else is parsed.
SYNONYMS: dict[str, tuple[str, str]] = {
    "vocals": ("vocals", "vocal"), "singing": ("vocals", "vocal"),
    "singer": ("vocals", "vocal"), "sung": ("vocals", "vocal"),
    "rapping": ("vocals", "vocal"), "rapper": ("vocals", "vocal"),
    "rap": ("genre", "hip hop"), "hiphop": ("genre", "hip hop"),
    "edm": ("genre", "electronic"), "electronica": ("genre", "electronic"),
    "relaxing": ("mood", "calm"), "relaxed": ("mood", "calm"), "chill": ("mood", "calm"),
    "peaceful": ("mood", "calm"), "gentle": ("mood", "calm"),
    "melancholy": ("mood", "sad"), "melancholic": ("mood", "sad"),
    "cheerful": ("mood", "happy"), "joyful": ("mood", "happy"),
    "exciting": ("mood", "energetic"), "aggressive": ("mood", "angry"),
    "scary": ("mood", "dark"), "eerie": ("mood", "dark"), "spooky": ("mood", "dark"),
}
#: A word within this many words after a negation is skipped: "without heavy
#: vocals" must not claim agreement with vocal tracks.
NEGATIONS = frozenset({"no", "not", "without", "non", "never"})
NEGATION_WINDOW = 3


def _vocabulary() -> dict[str, tuple[str, str]]:
    words = {
        label.replace(" ", ""): (attribute, label)
        for attribute in ("vocals", "genre", "mood")
        for label in LABELS[attribute]
    }
    return {**words, **SYNONYMS}


def query_terms(query: str) -> list[dict[str, str]]:
    """Attribute values a search names, in order, skipping negated ones."""
    text = re.sub(r"\bhip[\s-]+hop\b", "hiphop", query.lower())
    words = re.findall(r"[a-z&]+", text)
    vocabulary = _vocabulary()
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, word in enumerate(words):
        if word not in vocabulary:
            continue
        if NEGATIONS & set(words[max(0, i - NEGATION_WINDOW) : i]):
            continue
        attribute, value = vocabulary[word]
        if (attribute, value) in seen:
            continue
        seen.add((attribute, value))
        term = "hip hop" if word == "hiphop" else word
        found.append({"term": term, "attribute": attribute, "value": value})
    return found


def _agrees(term: dict[str, str], estimates: dict[str, object]) -> bool:
    if term["attribute"] == "mood":
        return term["value"] in estimates.get("mood", [])
    return estimates.get(term["attribute"]) == term["value"]


@dataclasses.dataclass(frozen=True)
class ExplainedResults:
    results: list[Recommendation]
    #: Tracks a filter hid because their estimate did not match.
    hidden_mismatch: int = 0
    #: Tracks a filter hid because their estimate is unknown.
    hidden_unknown: int = 0


def track_estimates(attrs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Known model estimates for one track; unknown ones are left out entirely."""
    out: dict[str, object] = {}
    for name in ("vocals", "genre"):
        entry = attrs.get(name)
        if isinstance(entry, dict) and entry.get("label") is not None:
            out[name] = entry["label"]
    mood = attrs.get("mood")
    if isinstance(mood, dict) and mood.get("labels"):
        out["mood"] = list(mood["labels"])
    return out


def track_measurements(attrs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Measured facts for one track."""
    out: dict[str, object] = {}
    tempo = attrs.get("tempo")
    if isinstance(tempo, dict) and tempo.get("bpm") is not None:
        out["tempo_bpm"] = tempo["bpm"]
    energy = attrs.get("energy")
    if isinstance(energy, dict) and energy.get("value") is not None:
        out["energy"] = {"band": energy_band(energy["value"]), "percentile": energy["value"]}
    return out


def shared_with_seed(
    seed: tuple[dict[str, object], dict[str, object]],
    result: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    """What the seed and the result agree on: agreement, not causation.

    Takes ``(estimates, measurements)`` pairs and returns ``{"estimates": ...,
    "measured": ...}``, dropping an empty half.
    """
    (seed_est, seed_meas), (est, meas) = seed, result
    shared_est: dict[str, object] = {}
    for name in ("vocals", "genre"):
        if name in seed_est and seed_est[name] == est.get(name):
            shared_est[name] = est[name]
    moods = [m for m in est.get("mood", []) if m in seed_est.get("mood", [])]
    if moods:
        shared_est["mood"] = moods
    shared_meas: dict[str, object] = {}
    if "tempo_bpm" in seed_meas and "tempo_bpm" in meas:
        a, b = seed_meas["tempo_bpm"], meas["tempo_bpm"]
        if abs(a - b) <= TEMPO_CLOSE * a:
            shared_meas["tempo_bpm"] = [a, b]
    if ("energy" in seed_meas and "energy" in meas
            and seed_meas["energy"]["band"] == meas["energy"]["band"]):
        shared_meas["energy"] = meas["energy"]["band"]
    shared: dict[str, object] = {}
    if shared_est:
        shared["estimates"] = {"kind": ESTIMATE_KIND, **shared_est}
    if shared_meas:
        shared["measured"] = {"kind": MEASURED_KIND, **shared_meas}
    return shared


def _with_evidence(
    results: list[Recommendation],
    *,
    match: dict[str, object],
    candidates: int,
    attributes: dict[str, dict[str, object]] | None,
    outcome: FilterOutcome | None,
    seed_attrs: tuple[dict[str, object], dict[str, object]] | None = None,
    terms: list[dict[str, str]] | None = None,
) -> list[Recommendation]:
    explained = []
    for rank, rec in enumerate(results, start=1):
        evidence: dict[str, object] = {
            "match": {**match, "rank": rank, "of": candidates,
                      "similarity": round(rec.similarity, 4)},
        }
        if outcome is not None and outcome.passed.get(rec.track_id):
            evidence["filters_passed"] = list(outcome.passed[rec.track_id])
        if attributes is not None:
            attrs = attributes.get(rec.track_id, {})
            estimates, measurements = track_estimates(attrs), track_measurements(attrs)
            if estimates:
                evidence["estimates"] = {"kind": ESTIMATE_KIND, **estimates}
            if measurements:
                evidence["measured"] = {"kind": MEASURED_KIND, **measurements}
            if seed_attrs is not None:
                shared = shared_with_seed(seed_attrs, (estimates, measurements))
                if shared:
                    evidence["shared_with_seed"] = shared
            filtered = {(e["attribute"], e["value"]) for e in evidence.get("filters_passed", [])
                        if isinstance(e["value"], str)}  # a tempo filter's value is a range
            agreed = [t for t in terms or [] if _agrees(t, estimates)
                      and (t["attribute"], t["value"]) not in filtered]
            if agreed:
                evidence["query_agreement"] = {"kind": ESTIMATE_KIND, "terms": agreed}
        explained.append(dataclasses.replace(rec, evidence=evidence))
    return explained


def _filter(
    catalog: Catalog,
    attributes: dict[str, dict[str, object]] | None,
    filters: AttributeFilters,
) -> FilterOutcome | None:
    if not filters.active():
        return None
    if attributes is None:
        raise RecommenderError(
            "attribute filters need attributes; run "
            "'python -m recommendation_engine.attributes' first"
        )
    return apply_filters(catalog.track_ids, attributes, filters)


def explained_search(
    catalog: Catalog,
    query_vector: np.ndarray,
    *,
    query_text: str,
    attributes: dict[str, dict[str, object]] | None,
    filters: AttributeFilters = AttributeFilters(),
    k: int = 10,
    query_space: EmbeddingSpace | None = None,
    max_per_artist: int | None = None,
) -> ExplainedResults:
    """Mood search with filters applied before the ``k`` cut, plus evidence."""
    outcome = _filter(catalog, attributes, filters)
    excluded = outcome.excluded if outcome else frozenset()
    results = search(catalog, query_vector, k=k, query_space=query_space,
                     exclude=excluded, max_per_artist=max_per_artist)
    explained = _with_evidence(
        results, match={"kind": "query", "query": query_text},
        candidates=len(catalog) - len(excluded), attributes=attributes, outcome=outcome,
        terms=query_terms(query_text),
    )
    return ExplainedResults(
        explained,
        hidden_mismatch=len(outcome.mismatched) if outcome else 0,
        hidden_unknown=len(outcome.unknown) if outcome else 0,
    )


def explained_similar(
    catalog: Catalog,
    track_id: str,
    *,
    attributes: dict[str, dict[str, object]] | None,
    filters: AttributeFilters = AttributeFilters(),
    k: int = 10,
    max_per_artist: int | None = None,
) -> ExplainedResults:
    """Similar songs with filters applied before the ``k`` cut, plus evidence."""
    catalog.position_of(track_id)  # unknown seed -> UnknownTrackError before filtering
    outcome = _filter(catalog, attributes, filters)
    excluded = (outcome.excluded if outcome else frozenset()) - {track_id}
    results = similar_to(catalog, track_id, k=k, exclude=excluded,
                         max_per_artist=max_per_artist)
    seed_title = str(catalog.metadata.get(track_id, {}).get("title", track_id))
    seed_attrs = None
    if attributes is not None:
        seed = attributes.get(track_id, {})
        seed_attrs = (track_estimates(seed), track_measurements(seed))
    explained = _with_evidence(
        results,
        match={"kind": "seed", "seed_track_id": track_id, "seed_title": seed_title},
        candidates=len(catalog) - len(excluded) - 1,
        attributes=attributes, outcome=outcome, seed_attrs=seed_attrs,
    )
    return ExplainedResults(
        explained,
        hidden_mismatch=len(outcome.mismatched - {track_id}) if outcome else 0,
        hidden_unknown=len(outcome.unknown - {track_id}) if outcome else 0,
    )


# ------------------------------------------------------------------------ wording


def _describe_filter(entry: dict[str, object]) -> str:
    name, value = entry["attribute"], entry["value"]
    if name == "tempo":
        low, high = value
        if low is not None and high is not None:
            span = f"{low:g}-{high:g} BPM"
        elif low is not None:
            span = f"at least {low:g} BPM"
        else:
            span = f"at most {high:g} BPM"
        return f"{span} (measured {entry['bpm']:.0f})"
    if name == "energy":
        return f"{value} energy (measured)"
    if name == "mood":
        return f"{value} mood (model estimate)"
    return f"{value} (model estimate)"


def _describe_estimates(est: dict[str, object]) -> list[str]:
    parts = [est[k] for k in ("vocals", "genre") if k in est]
    if "mood" in est:
        parts.append(", ".join(est["mood"]))
    return parts


def _describe_measured(meas: dict[str, object]) -> list[str]:
    parts = []
    if "tempo_bpm" in meas:
        tempo = meas["tempo_bpm"]
        if isinstance(tempo, list):
            parts.append(f"similar tempo ({tempo[0]:.0f} vs {tempo[1]:.0f} BPM)")
        else:
            parts.append(f"{tempo:.0f} BPM")
    if "energy" in meas:
        energy = meas["energy"]
        band = energy["band"] if isinstance(energy, dict) else energy
        parts.append(f"{band} energy for this catalog")
    return parts


def summarize(evidence: dict[str, object]) -> list[str]:
    """Reference wording: measured facts stated plainly, estimates marked as such."""
    match = evidence["match"]
    if match["kind"] == "query":
        why = f"closest sound match to \"{match['query']}\""
    else:
        why = f"sounds closest to {match['seed_title']}"
    why += f" (rank {match['rank']} of {match['of']})"
    if "filters_passed" in evidence:
        why += "; passed your filters: " + ", ".join(
            _describe_filter(e) for e in evidence["filters_passed"]
        )
    lines = [f"Why: {why}"]
    if "query_agreement" in evidence:
        lines.append("Agrees with your search: " + ", ".join(
            _describe_term(t) for t in evidence["query_agreement"]["terms"]
        ) + " (model estimate)")
    shared = evidence.get("shared_with_seed", {})
    parts = []
    if "estimates" in shared:
        parts.append(", ".join(_describe_estimates(shared["estimates"])) + " (model estimates)")
    if "measured" in shared:
        parts.append(", ".join(_describe_measured(shared["measured"])) + " (measured)")
    if parts:
        lines.append("Shared with seed: " + " · ".join(parts))
    if "estimates" in evidence:
        lines.append("Model estimates: " + " · ".join(_describe_estimates(evidence["estimates"])))
    if "measured" in evidence:
        lines.append("Measured: " + " · ".join(_describe_measured(evidence["measured"])))
    return lines


def _describe_term(term: dict[str, str]) -> str:
    if term["term"] == term["value"]:
        return f"\"{term['term']}\""
    return f"\"{term['term']}\" ({term['value']})"


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def short_reason(evidence: dict[str, object]) -> str:
    """One line per result: the strongest true statement, in priority order.

    1. filters the user applied that kept the track;
    2. search words the tagger agrees with;
    3. attributes a similar song shares with its seed;
    4. otherwise, the similarity match alone.
    The similarity anchor ("closest sound match to ..." / "sounds like ...") is
    always included, because it is the actual ranking criterion.
    """
    match = evidence["match"]
    if match["kind"] == "query":
        anchor = f"closest sound match to \"{match['query']}\""
    else:
        anchor = f"sounds like {match['seed_title']}"

    if "filters_passed" in evidence:
        filters = ", ".join(_describe_filter(e) for e in evidence["filters_passed"])
        return _capitalize(f"{filters} · {anchor}")
    if "query_agreement" in evidence:
        terms = ", ".join(_describe_term(t) for t in evidence["query_agreement"]["terms"])
        return _capitalize(f"matches {terms} in your search (model estimate) · {anchor}")
    shared = evidence.get("shared_with_seed")
    if shared:
        parts = [anchor]
        if "estimates" in shared:
            parts.append("both " + ", ".join(_describe_estimates(shared["estimates"]))
                         + " (model estimates)")
        measured = shared.get("measured", {})
        if "tempo_bpm" in measured:
            parts.append("similar tempo")
        if "energy" in measured:
            parts.append(f"both {measured['energy']} energy")
        return _capitalize(" · ".join(parts))
    return _capitalize(anchor)
