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
    out = ids(mmr_rerank(near_duplicates, pool, k=3, lambda_=0.5))
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
