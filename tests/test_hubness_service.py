"""Hub correction at the service layer, over the 4-track fixture.

Against the query [1,0,0,0] the catalog ranks a (1.0) > c (0.707) > b (0.0) > d (-1.0);
against seed a, the others rank c (0.707) > b (0.0) > d (-1.0). Penalties are chosen
by hand so the expected reorderings can be read off directly. MMR is switched off
(``diversity=1.0``) so only the hub correction moves anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.service import BLEND_WEIGHTS, Recommender, _zscore
from recommendation_engine.hubness import HubPenalties
from recommendation_engine.similarity import cosine_scores

TOWARD_A = np.array([1, 0, 0, 0], dtype=np.float32)
R = 2**-0.5


class Encoder:
    """Every query encodes to [1,0,0,0]; enough to drive the mood path."""

    def __init__(self, space):
        self.space = space

    def embed_text(self, texts):
        return np.stack([TOWARD_A for _ in texts])


#: Text penalties push a below c (1 - 0.5*0.8 = 0.6 < 0.707 - 0).
#: Audio penalties push b above c for seed a (0 - 0.5*-1 = 0.5 > 0.707 - 0.5*0.8 = 0.307).
PENALTIES = HubPenalties(
    neighbours=1,
    audio=np.array([0.0, -1.0, 0.8, 0.0]),
    text=np.array([0.8, 0.0, 0.0, 0.0]),
    bank_sha256="test", bank_size=1,
)


def make(catalog, *, hubness=PENALTIES, weight=0.5):
    return Recommender(catalog, None, Encoder(catalog.space), diversity=1.0,
                       hubness=hubness, hub_weight=weight)


def test_mood_search_uses_text_penalties_and_reports_raw_cosines(catalog):
    answer = make(catalog).recommend(query="anything", k=4)
    assert [r.track_id for r in answer.results] == ["c", "a", "b", "d"]
    a = next(r for r in answer.results if r.track_id == "a")
    assert a.score == pytest.approx(1.0)
    assert a.similarities == {"query": pytest.approx(1.0)}
    assert a.evidence["hubness"] == {
        "method": "csls", "reference": "query bank", "neighbours": 1,
        "weight": 0.5, "penalty": 0.8}
    assert a.evidence["match"]["similarity"] == pytest.approx(1.0)


def test_similar_songs_use_audio_penalties(catalog):
    answer = make(catalog).recommend(seed_id="a", k=3)
    assert [r.track_id for r in answer.results] == ["b", "c", "d"]
    assert answer.results[0].evidence["hubness"]["reference"] == "catalog tracks"
    assert answer.results[0].similarities == {"seed": pytest.approx(0.0)}


@pytest.mark.parametrize("make_kwargs", [{"weight": 0.0}, {"hubness": None}])
def test_off_means_plain_cosine_order_and_no_hub_evidence(catalog, make_kwargs):
    rec = make(catalog, **make_kwargs)
    search = rec.recommend(query="anything", k=4)
    similar = rec.recommend(seed_id="a", k=3)
    assert [r.track_id for r in search.results] == ["a", "c", "b", "d"]
    assert [r.track_id for r in similar.results] == ["c", "b", "d"]
    assert all("hubness" not in r.evidence for r in search.results + similar.results)


def test_text_side_missing_leaves_mood_search_uncorrected(catalog):
    audio_only = HubPenalties(neighbours=1, audio=PENALTIES.audio)
    rec = make(catalog, hubness=audio_only)
    assert [r.track_id for r in rec.recommend(query="x", k=4).results] == ["a", "c", "b", "d"]
    assert [r.track_id for r in rec.recommend(seed_id="a", k=3).results] == ["b", "c", "d"]


def test_blend_corrects_each_path_before_standardising(catalog):
    answer = make(catalog).recommend(query="anything", seed_id="b", k=3)
    q = cosine_scores(TOWARD_A, catalog.vectors) - 0.5 * PENALTIES.text
    s = cosine_scores(catalog.vector_of("b"), catalog.vectors) - 0.5 * PENALTIES.audio
    blended = BLEND_WEIGHTS["query"] * _zscore(q) + BLEND_WEIGHTS["seed"] * _zscore(s)
    expected = sorted((t for t in catalog.track_ids if t != "b"),
                      key=lambda t: (-blended[catalog.track_ids.index(t)], t))
    assert [r.track_id for r in answer.results] == expected
    raw_q = dict(zip(catalog.track_ids, cosine_scores(TOWARD_A, catalog.vectors)))
    for r in answer.results:
        assert r.similarities["query"] == pytest.approx(raw_q[r.track_id])
        assert set(r.evidence["hubness"]) == {"query", "seed"}


def test_invalid_weight_is_refused(catalog):
    with pytest.raises(ValueError):
        make(catalog, weight=2.0)
