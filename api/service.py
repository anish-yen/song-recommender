"""The logic behind the HTTP routes, with no HTTP in it.

``Recommender`` holds one loaded catalog, its attribute estimates, and one text
encoder, and answers the three questions the frontend asks: mood search, similar
songs, and the two combined. Everything ranking-related is delegated to
``recommendation_engine``; this module only chooses the path, blends scores when a
query and a seed arrive together, and shapes results for the frontend.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import numpy as np

from recommendation_engine.attributes import LABELS, load_attributes
from recommendation_engine.errors import IndexCorruptError, RecommenderError
from recommendation_engine.evidence import (
    explained_search,
    explained_similar,
    short_reason,
    track_estimates,
    track_measurements,
)
from recommendation_engine.filters import ENERGY_BANDS, TEMPO_LIMITS, AttributeFilters, apply_filters
from recommendation_engine.index import load_index
from recommendation_engine.similarity import Catalog, cosine_scores, rank_by_score

#: Upper bound on results per request; the design target is 10.
MAX_K = 50
#: Longest query accepted. CLAP's text tower truncates long input anyway, and the
#: frontend already cuts at 400 characters.
MAX_QUERY_CHARS = 400


@dataclasses.dataclass(frozen=True)
class Ranked:
    """One result: catalog metadata, what it was ranked on, and why."""

    track_id: str
    metadata: dict[str, object]
    score: float
    #: Cosine similarity per path that contributed ("query", "seed").
    similarities: dict[str, float]
    why: str
    evidence: dict[str, object]


@dataclasses.dataclass(frozen=True)
class Answer:
    mode: str  # "search" | "similar" | "blend"
    results: list[Ranked]
    hidden_mismatch: int = 0
    hidden_unknown: int = 0


def _zscore(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / spread


class Recommender:
    """One catalog plus one encoder, shared by every request in the process."""

    def __init__(self, catalog: Catalog, attributes, encoder) -> None:
        self.catalog = catalog
        self.attributes = attributes
        self.encoder = encoder
        # Torch modules are not documented as safe for concurrent forward passes;
        # text encoding is ~15 ms, so serialising it costs little.
        self._encode_lock = threading.Lock()

    @classmethod
    def from_index(cls, index_dir, *, device: str = "cpu") -> "Recommender":
        catalog = load_index(index_dir)
        try:
            attributes = load_attributes(index_dir, catalog)
        except IndexCorruptError:
            # No (or stale) attributes: search still works, filters are refused,
            # and reasons fall back to the similarity match alone.
            attributes = None
        from recommendation_engine.embeddings import ClapEncoder

        recommender = cls(catalog, attributes, ClapEncoder(space=catalog.space, device=device))
        # The first forward pass pays one-off setup (~1.4 s measured on CPU); pay it
        # at startup so the first user's search meets the latency target.
        recommender.encode("warm-up")
        return recommender

    # ------------------------------------------------------------------ lookups

    def track(self, track_id: str) -> dict[str, object]:
        self.catalog.position_of(track_id)  # UnknownTrackError if absent
        meta = dict(self.catalog.metadata[track_id])
        attrs = (self.attributes or {}).get(track_id, {})
        return {
            **meta,
            "estimates": track_estimates(attrs),
            "measured": track_measurements(attrs),
        }

    def audio_file(self, track_id: str) -> Path | None:
        """The clip to serve as a preview, only when its license is recorded."""
        self.catalog.position_of(track_id)  # UnknownTrackError if absent
        meta = self.catalog.metadata[track_id]
        path, license_ = meta.get("audio_path"), meta.get("license")
        if not isinstance(path, str) or not license_ or str(license_).startswith("unknown"):
            return None
        file = Path(path)
        return file if file.is_file() else None

    def find_tracks(self, text: str, *, limit: int = 10) -> list[dict[str, object]]:
        """Seed-picker lookup: case-insensitive match on title or artist."""
        needle = text.strip().casefold()
        if not needle:
            return []
        hits = []
        for track_id in self.catalog.track_ids:
            meta = self.catalog.metadata[track_id]
            title, artist = str(meta.get("title", "")), str(meta.get("artist", ""))
            haystack = f"{title} {artist}".casefold()
            if needle in haystack:
                # Title prefix first, then any title match, then artist-only matches.
                rank = 0 if title.casefold().startswith(needle) else (
                    1 if needle in title.casefold() else 2)
                hits.append((rank, title.casefold(), track_id, title, artist))
        hits.sort()
        return [{"track_id": t, "title": title, "artist": artist}
                for _, _, t, title, artist in hits[:limit]]

    def filter_options(self) -> dict[str, object]:
        """Values a filter can take, with how many catalog tracks carry each."""
        counts: dict[str, dict[str, int]] = {"vocals": {}, "genre": {}, "mood": {}}
        for attrs in (self.attributes or {}).values():
            estimates = track_estimates(attrs)
            for name in ("vocals", "genre"):
                if name in estimates:
                    counts[name][estimates[name]] = counts[name].get(estimates[name], 0) + 1
            for mood in estimates.get("mood", []):
                counts["mood"][mood] = counts["mood"].get(mood, 0) + 1
        available = self.attributes is not None
        return {
            "available": available,
            "vocals": [{"value": v, "count": counts["vocals"].get(v, 0)} for v in LABELS["vocals"]],
            "genre": [{"value": v, "count": counts["genre"].get(v, 0)} for v in LABELS["genre"]],
            "mood": [{"value": v, "count": counts["mood"].get(v, 0)} for v in LABELS["mood"]],
            "energy": [{"value": v} for v in ENERGY_BANDS],
            "tempo": {"min": TEMPO_LIMITS[0], "max": TEMPO_LIMITS[1]},
        }

    # ---------------------------------------------------------------- ranking

    def encode(self, text: str) -> np.ndarray:
        with self._encode_lock:
            return self.encoder.embed_text([text])[0]

    def recommend(
        self,
        *,
        query: str | None = None,
        seed_id: str | None = None,
        filters: AttributeFilters = AttributeFilters(),
        k: int = 10,
        max_per_artist: int | None = None,
    ) -> Answer:
        query = (query or "").strip()
        if not query and not seed_id:
            raise ValueError("send a query, a seed track, or both")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be an integer from 1 to {MAX_K}")
        filters.validate()
        if seed_id:
            self.catalog.position_of(seed_id)  # 404 before any encoding work

        if query and seed_id:
            return self._blend(query, seed_id, filters, k, max_per_artist)
        if seed_id:
            out = explained_similar(self.catalog, seed_id, attributes=self.attributes,
                                    filters=filters, k=k, max_per_artist=max_per_artist)
            path = "seed"
        else:
            out = explained_search(self.catalog, self.encode(query), query_text=query,
                                   attributes=self.attributes, filters=filters, k=k,
                                   query_space=self.encoder.space, max_per_artist=max_per_artist)
            path = "query"
        results = [
            Ranked(r.track_id, self.catalog.metadata[r.track_id], r.similarity,
                   {path: r.similarity}, short_reason(r.evidence), r.evidence)
            for r in out.results
        ]
        return Answer("similar" if seed_id else "search", results,
                      out.hidden_mismatch, out.hidden_unknown)

    def _blend(self, query, seed_id, filters, k, max_per_artist) -> Answer:
        """Query and seed together: sum of each path's z-scored cosine.

        Raw cosines are not comparable across paths (text-to-audio scores sit near
        0.1-0.35, audio-to-audio near 0.4-0.9), so each is standardised over the
        catalog first. Equal weighting is a starting guess, not a tuned value.
        """
        catalog = self.catalog
        query_vector = self.encode(query)
        q_scores = cosine_scores(query_vector, catalog.vectors)
        s_scores = cosine_scores(catalog.vector_of(seed_id), catalog.vectors)
        blended = _zscore(q_scores) + _zscore(s_scores)

        excluded: frozenset[str] = frozenset()
        hidden_mismatch = hidden_unknown = 0
        if filters.active():
            if self.attributes is None:
                raise RecommenderError(
                    "attribute filters need attributes; run "
                    "'python -m recommendation_engine.attributes' first")
            outcome = apply_filters(catalog.track_ids, self.attributes, filters)
            excluded = outcome.excluded
            hidden_mismatch = len(outcome.mismatched - {seed_id})
            hidden_unknown = len(outcome.unknown - {seed_id})
        ranked = rank_by_score(catalog, blended, k=k, exclude=excluded | {seed_id},
                               max_per_artist=max_per_artist)

        # Evidence for the chosen tracks, from each path's own explainer.
        everyone = len(catalog)
        q_ev = {r.track_id: r.evidence for r in explained_search(
            catalog, query_vector, query_text=query, attributes=self.attributes,
            filters=filters, k=everyone, query_space=self.encoder.space).results}
        s_ev = {r.track_id: r.evidence for r in explained_similar(
            catalog, seed_id, attributes=self.attributes, filters=filters, k=everyone).results}
        position = {t: i for i, t in enumerate(catalog.track_ids)}

        results = []
        for rec in ranked:
            evidence = {"query": q_ev.get(rec.track_id, {}), "seed": s_ev.get(rec.track_id, {}),
                        "blend": {"method": "sum of z-scored cosines", "weights": [1, 1]}}
            parts = [short_reason(e) for e in (evidence["query"], evidence["seed"]) if e]
            why = " · ".join(parts[:1] + [p[:1].lower() + p[1:] for p in parts[1:]])
            i = position[rec.track_id]
            results.append(Ranked(
                rec.track_id, catalog.metadata[rec.track_id], rec.similarity,
                {"query": float(q_scores[i]), "seed": float(s_scores[i])}, why, evidence))
        return Answer("blend", results, hidden_mismatch, hidden_unknown)
