"""Filters, evidence, and hand-label agreement over hand-written attributes.

Uses the shared 4-track ``catalog`` fixture. Against the query [1,0,0,0] it ranks
a (1.0) > c (0.707) > b (0.0) > d (-1.0); seeded from ``a`` it ranks c > b > d.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation_engine.agreement import compare_with_labels
from recommendation_engine.attributes import ESTIMATE_KIND, MEASURED_KIND
from recommendation_engine.errors import RecommenderError
from recommendation_engine.evidence import (
    explained_search,
    explained_similar,
    query_terms,
    short_reason,
    summarize,
)
from recommendation_engine.filters import AttributeFilters, apply_filters, energy_band

QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def track(vocals, genre, energy, bpm, moods=()):
    return {
        "mood": {"labels": list(moods), "scores": {m: 0.02 for m in moods}},
        "vocals": {"label": vocals, "vocal_score": 0.07 if vocals == "vocal" else 0.002,
                   "top_class": "Singing"},
        "genre": {"label": genre, "score": 0.25 if genre else 0.001, "scores": {}},
        "energy": {"value": energy, "loudness_dbfs": -12.0, "onset_rate": 2.0},
        "tempo": {"bpm": bpm, "pulse_clarity": 0.5 if bpm else 0.1},
    }


ATTRS = {
    "a": track("vocal", "jazz", 0.1, 120.0, ["calm", "sad"]),
    "b": track("instrumental", None, 0.5, 124.0, ["calm"]),
    "c": track(None, "jazz", 0.9, None),
    "d": track("instrumental", "rock", 0.2, 90.0, ["dark"]),
}


def ids(explained):
    return [r.track_id for r in explained.results]


class TestFilters:
    def test_mismatch_outranks_unknown_and_each_is_counted_apart(self):
        outcome = apply_filters(
            "abcd", ATTRS, AttributeFilters(vocals="instrumental", genre="rock")
        )
        assert set(outcome.passed) == {"d"}
        # c: vocals unknown *and* genre mismatch -> a mismatch, not an unknown.
        assert outcome.mismatched == {"a", "c"}
        assert outcome.unknown == {"b"}

    def test_tempo_range_hides_unmeasurable_tempo_as_unknown(self):
        outcome = apply_filters("abcd", ATTRS, AttributeFilters(tempo_min=100, tempo_max=130))
        assert set(outcome.passed) == {"a", "b"}
        assert outcome.mismatched == {"d"}
        assert outcome.unknown == {"c"}
        assert outcome.passed["b"] == [{"attribute": "tempo", "value": [100, 130],
                                        "kind": MEASURED_KIND, "bpm": 124.0}]

    def test_unknown_filter_value_is_rejected_not_silently_empty(self):
        with pytest.raises(ValueError, match="polka"):
            AttributeFilters(genre="polka").validate()

    def test_no_detected_mood_is_unknown_not_a_mismatch(self):
        outcome = apply_filters("abcd", ATTRS, AttributeFilters(mood="calm"))
        assert set(outcome.passed) == {"a", "b"}
        assert outcome.mismatched == {"d"}
        assert outcome.unknown == {"c"}
        assert outcome.passed["b"] == [{"attribute": "mood", "value": "calm",
                                        "kind": ESTIMATE_KIND, "score": 0.02}]

    def test_attribute_without_an_estimator_reports_it(self):
        labels = {"vocals": ["vocal"], "genre": ["jazz"], "mood": []}
        with pytest.raises(ValueError, match="unavailable"):
            AttributeFilters(mood="calm").validate(labels)

    @pytest.mark.parametrize("filters, message", [
        (AttributeFilters(tempo_min=140, tempo_max=120), "exceeds"),
        (AttributeFilters(tempo_max=1000), "between"),
    ])
    def test_impossible_tempo_ranges_are_rejected(self, filters, message):
        with pytest.raises(ValueError, match=message):
            filters.validate()

    @pytest.mark.parametrize("percentile, band", [
        (0.0, "low"), (0.33, "low"), (1 / 3, "medium"), (0.66, "medium"),
        (2 / 3, "high"), (1.0, "high"),
    ])
    def test_energy_bands_are_catalog_thirds(self, percentile, band):
        assert energy_band(percentile) == band


class TestExplainedSearch:
    def test_filters_apply_before_the_k_cut(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS,
                               filters=AttributeFilters(vocals="instrumental"), k=1)
        # a (best match) is vocal and c is unknown; b is the best track that passes.
        assert ids(out) == ["b"]
        assert (out.hidden_mismatch, out.hidden_unknown) == (1, 1)
        assert out.results[0].evidence["match"] == {
            "kind": "query", "query": "q", "rank": 1, "of": 2, "similarity": 0.0,
        }

    def test_only_applied_filters_become_reasons(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS,
                               filters=AttributeFilters(vocals="instrumental"), k=4)
        assert out.results[0].evidence["filters_passed"] == [{
            "attribute": "vocals", "value": "instrumental", "kind": ESTIMATE_KIND,
            "vocal_score": 0.002,
        }]

    def test_without_filters_attributes_are_described_not_given_as_reasons(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS, k=4)
        assert ids(out) == ["a", "c", "b", "d"]
        assert all("filters_passed" not in r.evidence for r in out.results)
        c = out.results[1].evidence
        # Unknown vocals, no detected mood, and an unmeasurable tempo are omitted.
        assert c["estimates"] == {"kind": ESTIMATE_KIND, "genre": "jazz"}
        assert c["measured"] == {"kind": MEASURED_KIND,
                                 "energy": {"band": "high", "percentile": 0.9}}

    def test_missing_attributes_degrade_to_the_match_alone(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=None, k=2)
        assert [set(r.evidence) for r in out.results] == [{"match"}, {"match"}]

    def test_filters_without_attributes_are_an_error(self, catalog):
        with pytest.raises(RecommenderError, match="need attributes"):
            explained_search(catalog, QUERY, query_text="q", attributes=None,
                             filters=AttributeFilters(vocals="vocal"))


class TestExplainedSimilar:
    def test_reports_agreement_with_the_seed_by_kind(self, catalog):
        out = explained_similar(catalog, "a", attributes=ATTRS, k=3)
        assert ids(out) == ["c", "b", "d"]
        shared = {r.track_id: r.evidence.get("shared_with_seed") for r in out.results}
        assert shared == {
            "c": {"estimates": {"kind": ESTIMATE_KIND, "genre": "jazz"}},
            # 124 is within 8% of the seed's 120 BPM; 90 is not.
            "b": {"estimates": {"kind": ESTIMATE_KIND, "mood": ["calm"]},
                  "measured": {"kind": MEASURED_KIND, "tempo_bpm": [120.0, 124.0]}},
            "d": {"measured": {"kind": MEASURED_KIND, "energy": "low"}},
        }

    def test_seed_failing_a_filter_is_not_counted_as_hidden(self, catalog):
        # Seed a is vocal; the filter would reject it, but it is never a candidate.
        out = explained_similar(catalog, "a", attributes=ATTRS,
                                filters=AttributeFilters(vocals="instrumental"))
        assert ids(out) == ["b", "d"]
        assert (out.hidden_mismatch, out.hidden_unknown) == (0, 1)
        assert out.results[0].evidence["match"]["of"] == 2


class TestSummary:
    def test_estimates_are_marked_and_measurements_stated(self, catalog):
        out = explained_search(catalog, QUERY, query_text="calm", attributes=ATTRS,
                               filters=AttributeFilters(vocals="instrumental"), k=1)
        assert summarize(out.results[0].evidence) == [
            'Why: closest sound match to "calm" (rank 1 of 2); '
            "passed your filters: instrumental (model estimate)",
            'Agrees with your search: "calm" (model estimate)',
            "Model estimates: instrumental · calm",
            "Measured: 124 BPM · medium energy for this catalog",
        ]

    def test_tempo_reason_quotes_the_measured_value(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS,
                               filters=AttributeFilters(tempo_min=100, tempo_max=130), k=1)
        assert summarize(out.results[0].evidence)[0].endswith(
            "passed your filters: 100-130 BPM (measured 120)"
        )

    def test_shared_with_seed_separates_estimates_from_measurements(self, catalog):
        out = explained_similar(catalog, "a", attributes=ATTRS, k=2)
        assert summarize(out.results[1].evidence)[1] == (
            "Shared with seed: calm (model estimates) · similar tempo (120 vs 124 BPM) (measured)"
        )



class TestQueryAgreement:
    def test_terms_include_synonyms_and_multiword_genres(self):
        terms = query_terms("Relaxing hip-hop with a female singer")
        assert [(t["term"], t["attribute"], t["value"]) for t in terms] == [
            ("relaxing", "mood", "calm"), ("hip hop", "genre", "hip hop"),
            ("singer", "vocals", "vocal"),
        ]

    @pytest.mark.parametrize("query", [
        "late-night beats without heavy vocals", "no vocals please", "not sad at all",
    ])
    def test_negated_words_never_count(self, query):
        assert query_terms(query) == []

    def test_only_terms_the_tagger_agrees_with_are_reported(self, catalog):
        out = explained_search(catalog, QUERY, query_text="calm jazz", attributes=ATTRS, k=4)
        agreement = {r.track_id: r.evidence.get("query_agreement") for r in out.results}
        # a: calm and jazz; c: jazz only (no mood); b: calm only; d: neither.
        assert [t["value"] for t in agreement["a"]["terms"]] == ["calm", "jazz"]
        assert [t["value"] for t in agreement["c"]["terms"]] == ["jazz"]
        assert [t["value"] for t in agreement["b"]["terms"]] == ["calm"]
        assert agreement["d"] is None

    def test_a_filter_is_not_repeated_as_agreement(self, catalog):
        out = explained_search(catalog, QUERY, query_text="calm", attributes=ATTRS,
                               filters=AttributeFilters(mood="calm"), k=1)
        assert "query_agreement" not in out.results[0].evidence


class TestShortReason:
    def test_filters_come_first(self, catalog):
        out = explained_search(catalog, QUERY, query_text="calm", attributes=ATTRS,
                               filters=AttributeFilters(vocals="instrumental"), k=1)
        assert short_reason(out.results[0].evidence) == (
            'Instrumental (model estimate) · closest sound match to "calm"'
        )

    def test_query_agreement_next(self, catalog):
        out = explained_search(catalog, QUERY, query_text="relaxing", attributes=ATTRS, k=1)
        assert short_reason(out.results[0].evidence) == (
            'Matches "relaxing" (calm) in your search (model estimate) · '
            'closest sound match to "relaxing"'
        )

    def test_similar_songs_name_what_they_share(self, catalog):
        out = explained_similar(catalog, "a", attributes=ATTRS, k=2)
        assert short_reason(out.results[1].evidence) == (
            "Sounds like Title a · both calm (model estimates) · similar tempo"
        )

    def test_otherwise_only_the_match(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=None, k=1)
        assert short_reason(out.results[0].evidence) == 'Closest sound match to "q"'


VOCAB = {"vocals": ["vocal", "instrumental"], "genre": ["jazz", "rock"],
         "mood": ["calm", "sad", "dark"]}


class TestAgreement:
    def test_counts_agreement_and_keeps_abstentions_apart(self):
        report = compare_with_labels(ATTRS, {
            "a": {"vocals": "vocal", "genre": "rock", "mood": ["calm"], "energy": "low"},
            "c": {"vocals": "vocal", "genre": None, "mood": ["sad"]},
            "d": {"mood": ["calm"]},
        }, vocabulary=VOCAB)
        assert [t for t, *_ in report["vocals"]["agree"]] == ["a"]
        assert [t for t, *_ in report["vocals"]["unknown"]] == ["c"]
        assert report["genre"]["disagree"] == [("a", "rock", "jazz")]
        assert report["energy"]["agree"] == [("a", "low", "low")]
        assert report["mood"] == {
            "agree": [("a", ["calm"], ["calm", "sad"])],
            "disagree": [("d", ["calm"], ["dark"])],
            "unknown": [("c", ["sad"], None)],   # no detected mood is not a miss
        }

    def test_rejects_labels_the_attributes_cannot_contain(self):
        with pytest.raises(ValueError, match="polka"):
            compare_with_labels(ATTRS, {"a": {"genre": "polka"}}, vocabulary=VOCAB)

    def test_rejects_tracks_missing_from_the_index(self):
        with pytest.raises(ValueError, match="ghost"):
            compare_with_labels(ATTRS, {"ghost": {"vocals": "vocal"}})
