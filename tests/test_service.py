"""Score meanings per mode at the service layer, over the 4-track fixture.

Against the query [1,0,0,0] the catalog ranks a (1.0) > c (0.707) > b (0.0) > d (-1.0).
No model weights are loaded: the encoder is the API tests' stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.service import BLEND_WEIGHTS, Recommender, _zscore
from recommendation_engine.filters import AttributeFilters
from recommendation_engine.similarity import cosine_scores

from .test_api import StubEncoder
from .test_evidence import ATTRS

TOWARD_A = np.array([1, 0, 0, 0], dtype=np.float32)


@pytest.fixture
def rec(catalog):
    return Recommender(catalog, ATTRS, StubEncoder())


def cosines(catalog, vector):
    return dict(zip(catalog.track_ids, cosine_scores(vector, catalog.vectors)))


def test_mood_only_scores_are_query_cosines(rec, catalog):
    answer = rec.recommend(query="toward a", k=4)
    cos = cosines(catalog, TOWARD_A)
    assert answer.mode == "search"
    assert [r.track_id for r in answer.results] == ["a", "c", "b", "d"]
    for r in answer.results:
        assert r.score == pytest.approx(cos[r.track_id])
        assert r.similarities == {"query": pytest.approx(cos[r.track_id])}


def test_seed_only_scores_are_seed_cosines_and_exclude_the_seed(rec, catalog):
    answer = rec.recommend(seed_id="a", k=4)
    cos = cosines(catalog, catalog.vector_of("a"))
    assert answer.mode == "similar"
    assert [r.track_id for r in answer.results] == ["c", "b", "d"]
    for r in answer.results:
        assert r.score == pytest.approx(cos[r.track_id])
        assert r.similarities == {"seed": pytest.approx(cos[r.track_id])}


def test_both_together_score_is_the_weighted_sum_of_z_scores(rec, catalog):
    answer = rec.recommend(query="toward a", seed_id="b", k=3)
    q = cosine_scores(TOWARD_A, catalog.vectors)
    s = cosine_scores(catalog.vector_of("b"), catalog.vectors)
    blended = dict(zip(catalog.track_ids,
                       BLEND_WEIGHTS["query"] * _zscore(q) + BLEND_WEIGHTS["seed"] * _zscore(s)))
    cos_q, cos_s = dict(zip(catalog.track_ids, q)), dict(zip(catalog.track_ids, s))
    assert answer.mode == "blend"
    assert [r.track_id for r in answer.results] == ["c", "a", "d"]
    for r in answer.results:
        assert r.score == pytest.approx(blended[r.track_id])
        assert r.similarities == {"query": pytest.approx(cos_q[r.track_id]),
                                  "seed": pytest.approx(cos_s[r.track_id])}
        assert r.evidence["blend"]["weights"] == BLEND_WEIGHTS


def test_blend_match_names_each_paths_own_rank(rec):
    answer = rec.recommend(query="toward a", seed_id="b", k=3)
    # Ranks count only tracks that could be returned (the seed b is not one):
    # query path a > c > d; seed path c > a = d, ties broken by track id.
    by_id = {r.track_id: r.evidence["match"] for r in answer.results}
    assert by_id["c"] == {"kind": "blend", "rank": 1, "of": 3, "query": "toward a",
                          "query_rank": 2, "seed_track_id": "b", "seed_title": "Title b",
                          "seed_rank": 1}
    assert by_id["a"]["query_rank"] == 1
    assert by_id["a"]["rank"] == 2


def test_blend_lifts_the_query_paths_attribute_evidence(rec):
    answer = rec.recommend(query="toward a", seed_id="b", k=3)
    first = answer.results[0].evidence  # c: genre jazz, energy 0.9, nothing else known
    assert first["estimates"] == first["query"]["estimates"]
    assert first["measured"] == first["query"]["measured"]


def test_blend_hides_filtered_tracks_and_never_returns_the_seed(rec):
    answer = rec.recommend(query="toward a", seed_id="b", k=4,
                           filters=AttributeFilters(vocals="instrumental"))
    assert [r.track_id for r in answer.results] == ["d"]
    assert (answer.hidden_mismatch, answer.hidden_unknown) == (1, 1)
    assert answer.results[0].evidence["filters_passed"][0]["value"] == "instrumental"
