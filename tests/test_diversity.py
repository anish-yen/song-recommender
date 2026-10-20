import dataclasses

import numpy as np
import pytest

from recommendation_engine.diversity import mmr_rerank
from recommendation_engine.similarity import rank_by_score, cosine_scores
from tests.conftest import make_catalog

QUERY = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def near_duplicates():
    # a1/a2/a3 are near-copies close to the query; b and c differ but are still relevant.
    return make_catalog([
        ("a1", "artist-1", [1.00, 0.00, 0.00, 0.0]),
        ("a2", "artist-2", [0.99, 0.02, 0.00, 0.0]),
        ("a3", "artist-3", [0.98, 0.03, 0.00, 0.0]),
        ("b",  "artist-4", [0.70, 0.70, 0.00, 0.0]),
        ("c",  "artist-5", [0.70, 0.00, 0.70, 0.0]),
    ])


def ranked(catalog, query, k=50):
    return rank_by_score(catalog, cosine_scores(np.array(query), catalog.vectors), k=k)


def ids(results):
    return [r.track_id for r in results]


def test_lambda_one_matches_plain_similarity(near_duplicates):
    pool = ranked(near_duplicates, QUERY)
    assert ids(mmr_rerank(near_duplicates, pool, k=5, lambda_=1.0)) == ids(pool)


def test_breaks_up_near_duplicates(near_duplicates):
    pool = ranked(near_duplicates, QUERY)
    assert ids(pool)[:3] == ["a1", "a2", "a3"]
    # relevance is pool-scaled: b and c sit at 0, so variety must be weighted heavily
    # to beat a near copy at 1.0 (this fixture has nothing between 0.707 and 1.0)
    out = ids(mmr_rerank(near_duplicates, pool, k=3, lambda_=0.2))
    assert out[0] == "a1"              # best match stays first
    assert {"b", "c"} & set(out)       # something different gets in


def test_keeps_original_similarity_scores(near_duplicates):
    pool = ranked(near_duplicates, QUERY)
    before = {r.track_id: r.similarity for r in pool}
    for r in mmr_rerank(near_duplicates, pool, k=5, lambda_=0.5):
        assert r.similarity == before[r.track_id]


def test_deterministic(near_duplicates):
    pool = ranked(near_duplicates, QUERY)
    assert ids(mmr_rerank(near_duplicates, pool, k=4)) == ids(mmr_rerank(near_duplicates, pool, k=4))


@pytest.mark.parametrize("k", [1, 5, 99])
def test_k_bounds(near_duplicates, k):
    pool = ranked(near_duplicates, QUERY)
    assert len(mmr_rerank(near_duplicates, pool, k=k)) == min(k, len(pool))


def test_rejects_bad_lambda(near_duplicates):
    with pytest.raises(ValueError):
        mmr_rerank(near_duplicates, ranked(near_duplicates, QUERY), k=3, lambda_=1.5)


def test_empty_pool(near_duplicates):
    assert mmr_rerank(near_duplicates, [], k=3) == []


def test_relevance_scale_does_not_change_the_picks(near_duplicates):
    """Blend scores are z-score sums, not cosines; an affine rescale must not matter."""
    pool = ranked(near_duplicates, QUERY)

    def rescaled(scale, shift):
        return [dataclasses.replace(r, similarity=r.similarity * scale + shift) for r in pool]

    compressed = ids(mmr_rerank(near_duplicates, rescaled(0.1, 0.0), k=4))
    stretched = ids(mmr_rerank(near_duplicates, rescaled(9.0, -3.0), k=4))
    assert compressed == stretched == ids(mmr_rerank(near_duplicates, pool, k=4))


def test_narrow_score_band_does_not_promote_a_weak_match():
    """Mood-search scores (text-to-audio cosine) sit in a narrow band; on the real
    index raw MMR promoted similarity rank 41 to position 2."""
    catalog = make_catalog([
        ("s1", "artist-1", [0.35, 0.937, 0.0, 0.0]),
        ("s2", "artist-2", [0.34, 0.673, 0.657, 0.0]),    # cos(s1, s2) = 0.75
        ("w", "artist-3", [0.27, 0.1126, 0.0, 0.956]),    # weak match, unlike s1
    ])
    pool = ranked(catalog, QUERY)
    assert ids(pool) == ["s1", "s2", "w"]
    assert ids(mmr_rerank(catalog, pool, k=2)) == ["s1", "s2"]


def test_opposite_sounding_tracks_earn_no_variety_bonus():
    catalog = make_catalog([
        ("a", "artist-1", [1.0, 0.0, 0.0, 0.0]),
        ("c", "artist-2", [0.7071, 0.7071, 0.0, 0.0]),
        ("b", "artist-3", [0.0, 1.0, 0.0, 0.0]),
        ("d", "artist-4", [-1.0, 0.0, 0.0, 0.0]),
    ])
    assert ids(mmr_rerank(catalog, ranked(catalog, QUERY), k=3)) == ["a", "c", "b"]


def test_a_distinct_relevant_track_beats_a_near_copy_at_the_default_weight():
    query = [0.7071, 0.7071, 0.0, 0.0]
    catalog = make_catalog([
        ("a1", "artist-1", [1.0, 0.20, 0.0, 0.0]),
        ("a2", "artist-2", [1.0, 0.19, 0.0, 0.0]),   # near copy of a1
        ("b1", "artist-3", [0.15, 1.0, 0.0, 0.0]),   # different, slightly less relevant
        ("f", "artist-4", [0.0, 0.0, 1.0, 0.0]),
    ])
    pool = ranked(catalog, query)
    assert ids(pool)[:3] == ["a1", "a2", "b1"]
    assert ids(mmr_rerank(catalog, pool, k=3)) == ["a1", "b1", "a2"]
