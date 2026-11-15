"""Hub correction (``hubness.py``): penalties, ordering, persistence, query bank.

The scenario tests build the failure seen on the FMA sample — one track in most
mood top-ks — on a synthetic catalog, so no model weights are needed.

Not pinned here, on purpose: how hub correction and MMR combine. In this synthetic
world, weight 0.5 plus MMR can let the hub back in (MMR rewards it for sounding
unlike the list) while 0.75 keeps it out. That depends on the real catalog's
geometry, so it is measured by ``hubness_eval.py``, not asserted.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from recommendation_engine import hubness
from recommendation_engine.config import EmbeddingSpace
from recommendation_engine.diversity import mmr_rerank
from recommendation_engine.errors import IndexCorruptError, RecommenderError
from recommendation_engine.hubness import (
    HubPenalties,
    adjustment,
    catalog_digest,
    compute_penalties,
    load_bank,
    load_hubness,
    neighbourhood_similarity,
    save_hubness,
)
from recommendation_engine.similarity import (
    Recommendation,
    cosine_scores,
    rank_by_score,
    search,
)
from tests.conftest import make_catalog

R = 2**-0.5


# ---------------------------------------------------------------- penalties


def test_neighbourhood_similarity_by_hand(catalog):
    # a=[1,0,0,0] b=[0,1,0,0] c=[r,r,0,0] d=[-1,0,0,0]; each track excluded from
    # its own neighbourhood.
    k1 = neighbourhood_similarity(catalog.vectors, catalog.vectors, k=1, exclude_self=True)
    assert k1 == pytest.approx([R, R, R, 0.0])
    k2 = neighbourhood_similarity(catalog.vectors, catalog.vectors, k=2, exclude_self=True)
    assert k2 == pytest.approx([R / 2, R / 2, R, -R / 2])


def test_neighbourhood_similarity_against_a_separate_reference_set(catalog):
    references = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    got = neighbourhood_similarity(catalog.vectors, references, k=1)
    assert got == pytest.approx([1.0, 0.0, R, 0.0])


def test_neighbourhood_similarity_is_the_same_in_blocks(monkeypatch):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(23, 8))
    whole = neighbourhood_similarity(matrix, matrix, k=4, exclude_self=True)
    monkeypatch.setattr(hubness, "_CHUNK", 5)
    blocked = neighbourhood_similarity(matrix, matrix, k=4, exclude_self=True)
    assert blocked == pytest.approx(whole)


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_neighbourhood_rejects_bad_k(catalog, k):
    # 4 tracks minus self leaves 3 neighbours, so k=4 is too many.
    with pytest.raises(ValueError):
        neighbourhood_similarity(catalog.vectors, catalog.vectors, k=k, exclude_self=True)


def test_compute_penalties_without_a_bank_has_no_text_side(catalog):
    penalties = compute_penalties(catalog, neighbours=2)
    assert penalties.audio == pytest.approx([R / 2, R / 2, R, -R / 2])
    assert penalties.text is None
    assert adjustment(penalties, "text", 0.5) is None
    assert adjustment(penalties, "audio", 0.5) is not None


@pytest.mark.parametrize("weight", [-0.1, 1.5, True, "0.5"])
def test_weight_outside_zero_to_one_is_refused(catalog, weight):
    with pytest.raises(ValueError):
        adjustment(compute_penalties(catalog, neighbours=1), "audio", weight)


def test_weight_zero_means_no_adjustment(catalog):
    assert adjustment(compute_penalties(catalog, neighbours=1), "audio", 0.0) is None


# ------------------------------------------------------------------ ordering

TOWARD_A = np.array([1, 0, 0, 0], dtype=np.float32)


def test_penalty_reorders_but_similarity_stays_the_raw_cosine(catalog):
    raw = cosine_scores(TOWARD_A, catalog.vectors)  # a 1.0, b 0, c r, d -1
    penalty = np.array([0.5, 0.0, 0.0, 0.0])
    results = rank_by_score(catalog, raw, k=4, penalty=penalty)
    assert [r.track_id for r in results] == ["c", "a", "b", "d"]
    by_id = {r.track_id: r for r in results}
    assert by_id["a"].similarity == pytest.approx(1.0)
    assert by_id["a"].rank_score == pytest.approx(0.5)
    assert by_id["c"].rank_score == pytest.approx(R)
    assert by_id["a"].to_dict()["rank_score"] == pytest.approx(0.5)


def test_without_a_penalty_rank_score_is_unset(catalog):
    results = search(catalog, TOWARD_A, k=2)
    assert all(r.rank_score is None for r in results)
    assert "rank_score" not in results[0].to_dict()


def test_penalty_ties_still_break_by_track_id(catalog):
    raw = cosine_scores(TOWARD_A, catalog.vectors)
    # a: 1 - 0.293 = r, equal to c's r - 0
    penalty = np.array([1 - R, 0.0, 0.0, 0.0])
    results = rank_by_score(catalog, raw, k=2, penalty=penalty)
    assert [r.track_id for r in results] == ["a", "c"]


@pytest.mark.parametrize("penalty", [np.zeros(3), np.array([0, 0, np.nan, 0])])
def test_bad_penalty_is_refused(catalog, penalty):
    with pytest.raises(ValueError):
        rank_by_score(catalog, cosine_scores(TOWARD_A, catalog.vectors), k=2, penalty=penalty)


def test_mmr_relevance_follows_the_ranking_score_not_raw_similarity():
    catalog = make_catalog([
        ("p0", "x", [1, 0, 0, 0]),
        ("p1", "y", [0, 1, 0, 0]),
        ("p2", "z", [0, 0, 1, 0]),
    ])
    # Orthogonal vectors: no redundancy, so MMR orders purely by relevance.
    pool = [
        Recommendation("p0", "", "x", similarity=1.0, rank_score=1.0),
        Recommendation("p2", "", "z", similarity=0.1, rank_score=0.9),
        Recommendation("p1", "", "y", similarity=0.9, rank_score=0.1),
    ]
    picked = mmr_rerank(catalog, pool, k=3, lambda_=0.7)
    assert [r.track_id for r in picked] == ["p0", "p2", "p1"]


# ------------------------------------------------------------------ scenario

SPACE_8D = EmbeddingSpace(dim=8, model_revision="test-revision")
CLUSTERS = (0, 1, 2)  # three sounds; dims 3-5 let tracks stray from them
AUDIO, TEXT = 6, 7    # every track shares AUDIO, every search shares TEXT


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _axis(i, scale=1.0):
    v = np.zeros(8)
    v[i] = scale
    return v


def _query(i, rng=None, noise=0.0):
    v = _axis(i) + _axis(TEXT)
    if rng is not None:
        v[3:6] += noise * rng.normal(size=3)
    return _unit(v)


@pytest.fixture
def hub_world():
    """Three sound clusters of 8 tracks, one hub, and a text query bank.

    A small model of CLAP's modality gap: audio vectors share one direction and text
    vectors another. The hub is the track leaning toward the text side, so it scores
    near the top for every search (3rd of 25 here) without matching any of them.
    """
    rng = np.random.default_rng(7)
    entries = []
    for i in CLUSTERS:
        for j, m in enumerate(np.linspace(0.2, 1.4, 8)):
            stray = rng.normal(size=3)
            v = _axis(i) + _axis(AUDIO)
            v[3:6] = m * stray / np.linalg.norm(stray)
            entries.append((f"c{i}-{j}", f"artist-{i}-{j}", _unit(v).tolist()))
    entries.append(("hub", "artist-hub", _unit(_axis(AUDIO) + _axis(TEXT, 0.9)).tolist()))
    catalog = make_catalog(entries, space=SPACE_8D)

    # Bank: searches near each cluster's query, plus generic searches ("some
    # music") along TEXT, as a real bank of everyday searches has.
    bank = [_query(i, rng, 0.05) for i in CLUSTERS for _ in range(12)]
    bank += [_unit(_axis(TEXT) + np.concatenate([[0, 0, 0], 0.05 * rng.normal(size=3), [0, 0]]))
             for _ in range(12)]
    penalties = compute_penalties(catalog, bank_vectors=np.array(bank), neighbours=5)
    return catalog, penalties


def _top(catalog, query, k, hub=None, mmr=None):
    pool = search(catalog, query.astype(np.float32), k=50 if mmr else k,
                  penalty=hub.offsets() if hub else None)
    if mmr is not None:
        pool = mmr_rerank(catalog, pool, k=k, lambda_=mmr)
    return [r.track_id for r in pool]


def test_plain_cosine_puts_the_hub_in_every_list(hub_world):
    catalog, _ = hub_world
    for i in CLUSTERS:
        assert "hub" in _top(catalog, _query(i), k=5)


def test_mmr_alone_does_not_remove_the_hub(hub_world):
    # Why within-list diversity is not enough: the hub sounds unlike the cluster
    # tracks, so MMR treats it as welcome variety rather than a repeat.
    catalog, _ = hub_world
    for i in CLUSTERS:
        assert "hub" in _top(catalog, _query(i), k=5, mmr=0.7)


def test_hub_correction_removes_the_hub_from_unrelated_lists(hub_world):
    catalog, penalties = hub_world
    hub = adjustment(penalties, "text", 0.5)
    for i in CLUSTERS:
        top = _top(catalog, _query(i), k=5, hub=hub)
        assert "hub" not in top
        assert all(t.startswith(f"c{i}-") for t in top)


def test_hub_is_still_first_when_the_query_is_about_it(hub_world):
    catalog, penalties = hub_world
    hub = adjustment(penalties, "text", 0.5)
    assert _top(catalog, catalog.vector_of("hub"), k=1, hub=hub) == ["hub"]


def test_the_hub_has_the_highest_text_penalty(hub_world):
    catalog, penalties = hub_world
    assert catalog.track_ids[int(np.argmax(penalties.text))] == "hub"


# --------------------------------------------------------------- persistence


def test_sidecar_round_trip(tmp_path, catalog):
    penalties = HubPenalties(neighbours=2, audio=np.array([0.1, 0.2, 0.3, 0.4]),
                             text=np.array([0.5, 0.6, 0.7, 0.8]),
                             bank_sha256="abc", bank_size=9)
    save_hubness(tmp_path, catalog, penalties)
    loaded = load_hubness(tmp_path, catalog)
    assert loaded.neighbours == 2
    assert loaded.audio == pytest.approx(penalties.audio)
    assert loaded.text == pytest.approx(penalties.text)
    assert (loaded.bank_sha256, loaded.bank_size) == ("abc", 9)


def test_sidecar_without_text_side_round_trips(tmp_path, catalog):
    save_hubness(tmp_path, catalog, compute_penalties(catalog, neighbours=1))
    assert load_hubness(tmp_path, catalog).text is None


def test_missing_sidecar_is_a_domain_error(tmp_path, catalog):
    with pytest.raises(IndexCorruptError, match="hubness"):
        load_hubness(tmp_path, catalog)


def test_sidecar_is_refused_after_any_vector_changes(tmp_path, catalog):
    save_hubness(tmp_path, catalog, compute_penalties(catalog, neighbours=1))
    vectors = catalog.vectors.copy()
    vectors[3] = [0.0, 0.0, 1.0, 0.0]  # only track d changed
    rebuilt = make_catalog([(t, catalog.metadata[t]["artist"], v.tolist())
                            for t, v in zip(catalog.track_ids, vectors)])
    assert catalog_digest(rebuilt) != catalog_digest(catalog)
    with pytest.raises(IndexCorruptError, match="different index build"):
        load_hubness(tmp_path, rebuilt)


def test_sidecar_is_refused_for_another_embedding_space(tmp_path, catalog):
    save_hubness(tmp_path, catalog, compute_penalties(catalog, neighbours=1))
    other = make_catalog(
        [(t, "x", v.tolist()) for t, v in zip(catalog.track_ids, catalog.vectors)],
        space=EmbeddingSpace(dim=4, model_revision="another-revision"))
    with pytest.raises(IndexCorruptError, match="embedding space"):
        load_hubness(tmp_path, other)


def test_sidecar_missing_a_track_is_refused(tmp_path, catalog):
    path = save_hubness(tmp_path, catalog, compute_penalties(catalog, neighbours=1))
    payload = json.loads(path.read_text())
    del payload["audio"]["penalties"]["c"]
    path.write_text(json.dumps(payload))
    with pytest.raises(IndexCorruptError, match="incomplete"):
        load_hubness(tmp_path, catalog)


def test_save_refuses_penalties_of_the_wrong_length(tmp_path, catalog):
    with pytest.raises(ValueError):
        save_hubness(tmp_path, catalog, HubPenalties(neighbours=1, audio=np.zeros(3)))


# ----------------------------------------------------------------- query bank


def test_bank_drops_case_and_space_duplicates_and_hashes_the_result(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"queries": ["Calm Piano", "calm  piano", "loud drums"]}))
    queries, digest = load_bank(path)
    assert queries == ["Calm Piano", "loud drums"]
    path.write_text(json.dumps(["Calm Piano", "loud drums"]))
    assert load_bank(path)[1] == digest


@pytest.mark.parametrize("payload", [[], {"queries": ["ok", " "]}, {"queries": [3]}, {"x": 1}])
def test_bank_rejects_bad_entries(tmp_path, payload):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RecommenderError):
        load_bank(path)


def test_shipped_bank_shares_nothing_with_the_evaluation_queries():
    # The eval measures repetition on these queries; if the bank contained them,
    # the penalty would be estimated from the very queries it is graded on.
    norm = lambda s: " ".join(re.findall(r"[a-z0-9']+", s.casefold()))  # noqa: E731
    bank, _ = load_bank()
    evals = json.loads(hubness.DEFAULT_BANK.with_name(
        "hubness_eval_queries.json").read_text(encoding="utf-8"))["queries"]
    assert len(bank) > 1000 and len(evals) >= 10
    eval_norms = [norm(q) for q in evals]
    for entry in bank:
        n = norm(entry)
        for e in eval_norms:
            assert n != e and f" {e} " not in f" {n} " and set(n.split()) != set(e.split()), (
                entry, e)
