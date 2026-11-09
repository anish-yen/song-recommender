"""Regression guards for cross-query result diversity.

Context: on the real FMA catalog, different mood queries were surfacing the
same tracks. These tests pin the behaviors that prevent that on synthetic
catalogs (no model weights needed):

1. Well-separated clusters must not bleed into each other's top-k.
2. When one dense cluster dominates two nearby queries, plain similarity
   repeats the same tracks across queries; MMR must cut that overlap.
"""

import numpy as np

from recommendation_engine.diversity import mmr_rerank
from recommendation_engine.similarity import rank_by_score, search, cosine_scores
from tests.conftest import make_catalog


def _cluster(prefix, center, count, spread=0.02):
    """Tracks tightly clustered around a unit-ish center vector."""
    center = np.array(center, dtype=np.float64)
    center = center / np.linalg.norm(center)
    out = []
    for i in range(count):
        jitter = np.full(len(center), spread * ((i % 5) - 2) / 2.0)
        v = center + jitter
        v = v / np.linalg.norm(v)
        out.append((f"{prefix}{i}", f"artist-{prefix}{i}", v.tolist()))
    return out


def _top_ids(catalog, query, k=10, mmr_lambda=None):
    q = np.array(query, dtype=np.float32)
    if mmr_lambda is None:
        return [r.track_id for r in search(catalog, q, k=k)]
    scores = cosine_scores(q, catalog.vectors)
    pool = rank_by_score(catalog, scores, k=50)
    return [r.track_id for r in mmr_rerank(catalog, pool, k=k, lambda_=mmr_lambda)]


def test_orthogonal_queries_do_not_share_results():
    catalog = make_catalog(
        _cluster("calm", [1, 0, 0, 0], 8) + _cluster("energy", [0, 1, 0, 0], 8)
    )
    calm = set(_top_ids(catalog, [1, 0, 0, 0], k=5))
    energy = set(_top_ids(catalog, [0, 1, 0, 0], k=5))
    assert calm.isdisjoint(energy)
    assert all(t.startswith("calm") for t in calm)
    assert all(t.startswith("energy") for t in energy)


def test_mmr_reduces_cross_query_overlap_for_nearby_queries():
    # One dense "pop" cluster sits between two nearby queries, with distinct
    # peripheral tracks closer to each query individually. Plain similarity
    # fills both top-10s from the shared cluster; MMR should mix in the
    # query-specific tracks and lower the overlap.
    shared = _cluster("pop", [1, 1, 0, 0], 12, spread=0.01)
    left = _cluster("left", [1, 0.8, 0.4, 0], 6, spread=0.01)
    right = _cluster("right", [0.8, 1, 0, 0.4], 6, spread=0.01)
    catalog = make_catalog(shared + left + right)

    q1, q2 = [1, 0.9, 0.2, 0], [0.9, 1, 0, 0.2]
    plain = set(_top_ids(catalog, q1)) & set(_top_ids(catalog, q2))
    diverse = set(_top_ids(catalog, q1, mmr_lambda=0.5)) & set(
        _top_ids(catalog, q2, mmr_lambda=0.5)
    )
    assert len(diverse) < len(plain)


def test_overlap_metric_flags_the_reported_regression():
    # Directly pin the failure mode: if two different mood queries return
    # identical top-10 lists on a catalog with enough distinct material,
    # that is the bug reported from the FMA demo. Guard the metric itself.
    catalog = make_catalog(
        _cluster("a", [1, 0, 0, 0], 10) + _cluster("b", [0, 0, 1, 0], 10)
    )
    q1 = set(_top_ids(catalog, [1, 0.05, 0, 0]))
    q2 = set(_top_ids(catalog, [0, 0.05, 1, 0]))
    overlap = len(q1 & q2) / 10
    assert overlap <= 0.2
