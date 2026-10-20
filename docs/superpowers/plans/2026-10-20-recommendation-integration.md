# Recommendation Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the MMR diversity rerank into every recommendation path with a real candidate pool, pin the mood/seed/blend retrieval behaviour, make each "why this song" line specific and honest, and prove the filters constrain results.

**Architecture:** Retrieval stays exact NumPy cosine over the in-memory index (`recommendation_engine/similarity.py`); there is no pgvector or Python-side database, so no migration is needed. `api/service.py::Recommender.recommend` retrieves a pool of `CANDIDATE_POOL` (50) candidates per mode, `diversity.mmr_rerank` picks the final `k`, and `evidence.short_reason` turns structured evidence into one line per result.

**Tech Stack:** Python 3.12, NumPy, FastAPI, pytest (run with `uv run --frozen --offline pytest`); Next.js 15 frontend (`npm test`, `npm run lint`, `npm run build`).

**Spec:** the task request of 2026-10-20 (conversation), plus `AGENTS.md` and `recommendation_engine/README.md` evidence rules 1-4.

## Global Constraints

- Work on `feature/recommendation-integration` in the existing checkout (`~/Desktop/song-recommender`); never commit to `main`; no push, merge, PR or deploy without approval.
- Preserve `data/audio/fma/` (8,000 MP3s), `.index-fma-small/` (7,993 tracks, fingerprint `71ffea9c…`), `.index/`, `data/manifest*.json`, `~/.cache/huggingface/hub`. No `git clean`/`reset`, no song or model download, no full reindex without asking.
- Run Python with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Do not change the embedding space (`config.py`, `audio.py`, `embeddings.py`, `vectors.py`, `index.py`): any change there invalidates the 7,993-track index.
- Do not touch activity-word filtering (PLAN.md "P1: activity auto-filter") or auth/playlists/Prisma (`src/app/api/auth`, `src/app/api/playlists`, `src/lib/auth.js`, `src/lib/playlist-access.js`, `prisma/`).
- No team-member names in code or documentation.
- Explanations use only measured evidence: never state a genre, mood, vocals, tempo or energy value the attributes do not contain; estimates are worded as estimates (`ESTIMATE_KIND = "ast-audioset"`), measurements as measured (`MEASURED_KIND = "measured"`).
- Keep `tests/test_api.py::TestDiscoverPagePayload` passing unchanged.

## Measured baseline (real index, 2026-10-20)

- The Next.js route asks for `k=5`; `Recommender.recommend` retrieves exactly `k`, so a rerank would have nothing to choose from.
- `mmr_rerank` compares raw relevance with raw cosine redundancy. Query-mode relevance (text-to-audio cosine) spans only 0.085 across the top 50 while candidate-to-candidate cosine spans ~0.3, so at λ=0.7 MMR promoted similarity rank 41 to position 2. Blend relevance is a sum of z-scores (~6), so redundancy is negligible and MMR is a no-op.
- With relevance min-max scaled within the pool, λ=0.7 kept the top match first and never picked below similarity rank 12 across 5 queries, 3 seeds and 1 blend. The artist cap already removes near-duplicates in these lists, so expect small reorders, not big changes. MMR does not address cross-query hub tracks.
- Attribute coverage on 7,993 tracks: vocals 7,565, genre 6,845, mood 4,777, tempo 6,254, energy 7,993.
- Current `why` lines repeat: 4 of 5 results for "calm piano at night" read identically.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `recommendation_engine/diversity.py` | modify | MMR with pool-scaled relevance and non-negative redundancy |
| `api/service.py` | modify | candidate pool, rerank call, blend weights constant, blend evidence shape, diversity evidence |
| `recommendation_engine/evidence.py` | modify | `short_reason` wording: lead + rank anchor + variety note + own facts |
| `tests/test_diversity.py` | modify | scale invariance, clipping, graded-duplicate case |
| `tests/test_api.py` | modify | rerank is called, pool size, duplicates, too-few, blend, filters invariants |
| `tests/test_evidence.py` | modify | new wording, no invented attributes, distinct lines, filter boundaries |
| `tests/test_service.py` | create | mood/seed/blend score meanings at the service layer |
| `recommendation_engine/PLAN.md`, `api/README.md` | modify | status row and `why` example |

---

### Task 1: Scale-safe MMR wired into every path

**Files:**
- Modify: `recommendation_engine/diversity.py`
- Modify: `api/service.py` (`Recommender.__init__`, `recommend`, `_blend`)
- Test: `tests/test_diversity.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `mmr_rerank(catalog, ranked, *, k, lambda_=DEFAULT_LAMBDA) -> list[Recommendation]` (unchanged signature; relevance is now pool-scaled). `api.service.CANDIDATE_POOL = DEFAULT_POOL` (50). `Recommender(catalog, attributes, encoder, *, diversity=DEFAULT_LAMBDA)`. Each result's `evidence["diversity"] = {"method": "mmr", "lambda": float, "pool": int, "similarity_rank": int, "position": int}`; `evidence["match"]["rank"]` stays the similarity (or blended) rank.

- [ ] **Step 1: Write failing diversity tests**

Append to `tests/test_diversity.py`:

```python
def test_relevance_scale_does_not_change_the_picks(near_duplicates):
    """Blend scores are z-score sums, not cosines; an affine rescale must not matter."""
    pool = ranked(near_duplicates, QUERY)
    stretched = [dataclasses.replace(r, similarity=r.similarity * 9.0 - 3.0) for r in pool]
    assert ids(mmr_rerank(near_duplicates, stretched, k=4)) == ids(mmr_rerank(near_duplicates, pool, k=4))


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
```

Add `import dataclasses` at the top. Change `test_breaks_up_near_duplicates` to `lambda_=0.2` with the comment `# relevance is pool-scaled: b and c sit at 0, so variety must be weighted heavily to beat a near copy at 1.0`; its fixture has no candidates between 0.707 and 1.0.

- [ ] **Step 2: Run them, expect failures**

Run: `HF_HUB_OFFLINE=1 uv run --frozen --offline pytest tests/test_diversity.py -q`
Expected: `test_relevance_scale_does_not_change_the_picks`, `test_opposite_sounding_tracks_earn_no_variety_bonus` and `test_breaks_up_near_duplicates` FAIL.

- [ ] **Step 3: Implement in `diversity.py`**

Replace the vector/relevance block in `mmr_rerank` with:

```python
    vectors = np.stack([catalog.vector_of(r.track_id) for r in ranked]).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Opposite-sounding tracks are not "more different" than unrelated ones, so a
    # negative cosine earns no bonus.
    pairwise = np.clip(vectors @ vectors.T, 0.0, None)
    relevance = _pool_scaled(np.array([r.similarity for r in ranked], dtype=np.float64))
```

and add:

```python
def _pool_scaled(scores: np.ndarray) -> np.ndarray:
    """Map the pool's scores onto [0, 1] so lambda means the same in every mode.

    Mood search scores (text-to-audio cosine) span ~0.1 over a pool, seed scores
    (audio-to-audio cosine) ~0.15, and blend scores are z-score sums. Redundancy is
    always an audio-to-audio cosine, so only relevance needs rescaling.
    """
    spread = float(scores.max() - scores.min())
    if spread < 1e-12:
        return np.zeros_like(scores)
    return (scores - float(scores.min())) / spread
```

Update the module docstring bullet list with the two rules above.

- [ ] **Step 4: Run diversity tests, expect PASS**

Run: `HF_HUB_OFFLINE=1 uv run --frozen --offline pytest tests/test_diversity.py -q`

- [ ] **Step 5: Write failing API tests proving the rerank is used**

Add to `tests/test_api.py` (imports: `from recommendation_engine import diversity`, `import api.service as service_module`, `from .conftest import make_catalog`):

```python
class SpyRerank:
    def __init__(self):
        self.calls = []

    def __call__(self, catalog, ranked, *, k, lambda_):
        self.calls.append({"pool": [r.track_id for r in ranked], "k": k, "lambda": lambda_})
        return diversity.mmr_rerank(catalog, ranked, k=k, lambda_=lambda_)


def wide_catalog(n=12):
    """n tracks fanning away from [1,0,0,0], each by its own artist."""
    rows = []
    for i in range(n):
        angle = i * 0.12
        rows.append((f"t{i:02d}", f"artist-{i}", [np.cos(angle), np.sin(angle), 0.0, 0.0]))
    return make_catalog(rows)


class TestRerankIsWired:
    @pytest.mark.parametrize("payload", [
        {"query": "toward a", "k": 3},
        {"seed": {"providerId": "t00"}, "k": 3},
        {"query": "toward a", "seed": {"providerId": "t05"}, "k": 3},
    ])
    def test_every_mode_reranks_a_pool_larger_than_k(self, monkeypatch, payload):
        spy = SpyRerank()
        monkeypatch.setattr(service_module, "mmr_rerank", spy)
        with TestClient(create_app(Recommender(wide_catalog(), None, StubEncoder()))) as c:
            response = c.post("/recommendations", json=payload)
        assert response.status_code == 200
        assert len(spy.calls) == 1
        assert spy.calls[0]["k"] == 3
        expected_pool = 11 if "seed" in payload else 12  # the seed is never a candidate
        assert len(spy.calls[0]["pool"]) == expected_pool
        assert ids(response)[0] == spy.calls[0]["pool"][0]  # best match stays first

    def test_pool_is_capped_at_the_candidate_pool(self, monkeypatch):
        spy = SpyRerank()
        monkeypatch.setattr(service_module, "mmr_rerank", spy)
        with TestClient(create_app(Recommender(wide_catalog(60), None, StubEncoder()))) as c:
            c.post("/recommendations", json={"query": "toward a", "k": 5, "max_per_artist": None})
        assert len(spy.calls[0]["pool"]) == service_module.CANDIDATE_POOL

    def test_near_copy_gives_way_to_a_distinct_match(self):
        catalog = make_catalog([
            ("a1", "artist-1", [1.0, 0.20, 0.0, 0.0]),
            ("a2", "artist-2", [1.0, 0.19, 0.0, 0.0]),
            ("b1", "artist-3", [0.15, 1.0, 0.0, 0.0]),
            ("f", "artist-4", [0.0, 0.0, 1.0, 0.0]),
        ])

        class Between:
            space = SPACE_4D

            def embed_text(self, texts):
                return np.array([[0.7071, 0.7071, 0.0, 0.0]] * len(texts), dtype=np.float32)

        with TestClient(create_app(Recommender(catalog, None, Between()))) as c:
            body = c.post("/recommendations", json={"query": "x", "k": 3}).json()
        assert [r["providerId"] for r in body["results"]] == ["a1", "b1", "a2"]
        b1 = body["results"][1]
        assert b1["evidence"]["match"]["rank"] == 3          # similarity rank kept
        assert b1["evidence"]["diversity"]["position"] == 2
        assert b1["score"] == pytest.approx(b1["similarities"]["query"], abs=1e-4)

    def test_too_few_candidates_returns_what_exists(self, client):
        response = client.post("/recommendations", json={"query": "toward a", "k": 10})
        assert sorted(ids(response)) == ["a", "b", "c", "d"]
        only_one = client.post("/recommendations", json={
            "query": "toward a", "filters": {"genre": "rock"}, "k": 5})
        assert ids(only_one) == ["d"]

    def test_lambda_one_restores_plain_similarity_order(self):
        rec = Recommender(wide_catalog(), None, StubEncoder(), diversity=1.0)
        with TestClient(create_app(rec)) as c:
            assert ids(c.post("/recommendations", json={"query": "toward a", "k": 5,
                                                        "max_per_artist": None})) == [
                "t00", "t01", "t02", "t03", "t04"]
```

- [ ] **Step 6: Run, expect failures**

Run: `HF_HUB_OFFLINE=1 uv run --frozen --offline pytest tests/test_api.py -q`
Expected: `TestRerankIsWired` fails (`mmr_rerank` attribute missing on `api.service`, unexpected `diversity` kwarg).

- [ ] **Step 7: Implement in `api/service.py`**

```python
from recommendation_engine.diversity import DEFAULT_LAMBDA, DEFAULT_POOL, mmr_rerank

#: Candidates retrieved per request before the diversity rerank picks ``k``.
CANDIDATE_POOL = DEFAULT_POOL
```

`__init__(self, catalog, attributes, encoder, *, diversity: float = DEFAULT_LAMBDA)` stores `self.diversity`.

In `recommend`, compute `pool_k = max(k, CANDIDATE_POOL)`, pass `k=pool_k` to `explained_similar` / `explained_search`, then `chosen = self._rerank(out.results, k)` and build `Ranked` objects from `chosen`. Pass `pool_k` into `_blend` and replace its `rank_by_score(..., k=k, ...)` with `k=pool_k` followed by `ranked = self._rerank(pool, k)`.

```python
    def _rerank(self, pool: list, k: int) -> list:
        """Pick ``k`` of ``pool`` with MMR; similarity ranks and scores are untouched."""
        picked = mmr_rerank(self.catalog, pool, k=k, lambda_=self.diversity)
        rank_of = {r.track_id: i for i, r in enumerate(pool, start=1)}
        return [
            dataclasses.replace(r, evidence={**r.evidence, "diversity": {
                "method": "mmr", "lambda": self.diversity, "pool": len(pool),
                "similarity_rank": rank_of[r.track_id], "position": position}})
            for position, r in enumerate(picked, start=1)
        ]
```

- [ ] **Step 8: Run full Python suite, expect PASS**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --offline pytest -q`
Expected: all pass (219 before + the new tests), 11 deselected.

- [ ] **Step 9: Real-index check (no model download; cached weights)**

Run the probe in `scratchpad/probe_real.py` and confirm: top match unchanged in every mode; no result below similarity rank 15; blend latency still under 2 s.

- [ ] **Step 10: Commit**

```bash
git add recommendation_engine/diversity.py api/service.py tests/test_diversity.py tests/test_api.py
git commit -m "feat(rerank): wire MMR over a 50-candidate pool with pool-scaled relevance"
```

---

### Task 2: Mood, seed and combined retrieval with explicit blending

**Files:**
- Modify: `api/service.py` (`_blend`)
- Create: `tests/test_service.py`

**Interfaces:**
- Consumes: `Recommender`, `CANDIDATE_POOL` from Task 1.
- Produces: `api.service.BLEND_WEIGHTS = {"query": 1.0, "seed": 1.0}` used by both the score and `evidence["blend"]["weights"]`. Blend evidence gains a top-level `"match": {"kind": "blend", "rank": int, "of": int, "query": str, "query_rank": int, "seed_track_id": str, "seed_title": str, "seed_rank": int}` plus `filters_passed` / `query_agreement` / `estimates` / `measured` copied from the query-path evidence and `shared_with_seed` from the seed-path evidence. Nested `query`, `seed`, `blend` keys stay.

- [ ] **Step 1: Write failing service tests** in `tests/test_service.py`:

```python
"""Score meanings per mode, over the 4-track fixture (no model weights)."""

from __future__ import annotations

import numpy as np
import pytest

from api.service import BLEND_WEIGHTS, Recommender, _zscore
from recommendation_engine.similarity import cosine_scores

from .test_api import StubEncoder
from .test_evidence import ATTRS

A = np.array([1, 0, 0, 0], dtype=np.float32)


@pytest.fixture
def rec(catalog):
    return Recommender(catalog, ATTRS, StubEncoder())


def test_mood_only_scores_are_query_cosines(rec, catalog):
    answer = rec.recommend(query="toward a", k=4)
    cos = dict(zip(catalog.track_ids, cosine_scores(A, catalog.vectors)))
    assert answer.mode == "search"
    for r in answer.results:
        assert r.score == pytest.approx(cos[r.track_id])
        assert r.similarities == {"query": pytest.approx(cos[r.track_id])}


def test_seed_only_scores_are_seed_cosines_and_exclude_the_seed(rec, catalog):
    answer = rec.recommend(seed_id="a", k=4)
    cos = dict(zip(catalog.track_ids, cosine_scores(catalog.vector_of("a"), catalog.vectors)))
    assert answer.mode == "similar"
    assert "a" not in [r.track_id for r in answer.results]
    for r in answer.results:
        assert r.score == pytest.approx(cos[r.track_id])


def test_both_together_score_is_the_weighted_sum_of_z_scores(rec, catalog):
    answer = rec.recommend(query="toward a", seed_id="b", k=3)
    q = cosine_scores(A, catalog.vectors)
    s = cosine_scores(catalog.vector_of("b"), catalog.vectors)
    blended = dict(zip(catalog.track_ids,
                       BLEND_WEIGHTS["query"] * _zscore(q) + BLEND_WEIGHTS["seed"] * _zscore(s)))
    assert answer.mode == "blend"
    assert [r.track_id for r in answer.results] == ["c", "a", "d"]
    for r in answer.results:
        assert r.score == pytest.approx(blended[r.track_id])
        assert r.evidence["blend"]["weights"] == BLEND_WEIGHTS
        match = r.evidence["match"]
        assert match["kind"] == "blend" and match["seed_track_id"] == "b"
        assert match["query_rank"] >= 1 and match["seed_rank"] >= 1


def test_blend_hides_filtered_tracks_and_never_returns_the_seed(rec):
    from recommendation_engine.filters import AttributeFilters
    answer = rec.recommend(query="toward a", seed_id="b", k=4,
                           filters=AttributeFilters(vocals="instrumental"))
    assert [r.track_id for r in answer.results] == ["d"]
    assert (answer.hidden_mismatch, answer.hidden_unknown) == (1, 1)
```

- [ ] **Step 2: Run, expect failures** — `BLEND_WEIGHTS` import error.

Run: `HF_HUB_OFFLINE=1 uv run --frozen --offline pytest tests/test_service.py -q`

- [ ] **Step 3: Implement**

In `api/service.py`: add `BLEND_WEIGHTS = {"query": 1.0, "seed": 1.0}`; compute `blended = BLEND_WEIGHTS["query"] * _zscore(q_scores) + BLEND_WEIGHTS["seed"] * _zscore(s_scores)`; build each result's evidence as:

```python
            q, s = q_ev.get(rec.track_id, {}), s_ev.get(rec.track_id, {})
            evidence = {
                "match": {"kind": "blend", "rank": rec.evidence.get("diversity", {}).get(
                              "similarity_rank", position),
                          "of": len(catalog) - len(excluded | {seed_id}),
                          "query": query, "query_rank": q.get("match", {}).get("rank"),
                          "seed_track_id": seed_id, "seed_title": seed_title,
                          "seed_rank": s.get("match", {}).get("rank")},
                **{key: q[key] for key in ("filters_passed", "query_agreement", "estimates", "measured")
                   if key in q},
                **({"shared_with_seed": s["shared_with_seed"]} if "shared_with_seed" in s else {}),
                "diversity": rec.evidence["diversity"],
                "query": q, "seed": s,
                "blend": {"method": "sum of z-scored cosines", "weights": BLEND_WEIGHTS},
            }
```

(`rank_by_score` results carry empty evidence, so attach the diversity block in `_rerank` before this loop; `seed_title = str(catalog.metadata[seed_id].get("title", seed_id))`.) Keep `why` built by Task 3; until then use `short_reason(evidence)` only after Task 3 lands — in this task keep the existing join of the two path reasons.

- [ ] **Step 4: Run full suite, expect PASS**

- [ ] **Step 5: Commit** `feat(blend): one weights constant and a top-level blend match in evidence`

---

### Task 3: Specific, honest "why this song"

**Files:**
- Modify: `recommendation_engine/evidence.py` (`short_reason` and helpers)
- Modify: `api/service.py` (blend `why = short_reason(evidence)`)
- Test: `tests/test_evidence.py` (`TestShortReason` rewritten, new `TestWhyIsHonest`), `tests/test_api.py`

**Interfaces:**
- Consumes: evidence shapes from Tasks 1-2 (`match.kind in {"query","seed","blend"}`, optional `diversity`).
- Produces: `short_reason(evidence) -> str` formatted `lead · anchor[ · chosen for variety][ · estimated …][ · … (measured)]`.

Wording rules:
- **lead** (first true one): filters that kept the track; else search words the tagger agrees with; else attributes shared with the seed. Same phrases as today.
- **anchor** (always; the actual ranking contribution): query `#3 of 7,993 for "calm piano"`; seed `#2 of 7,992 closest in sound to Night`; blend `#12 for "calm piano" and #3 closest to Night (equal weight)`.
- **variety**: `chosen for variety` only when `diversity.position < diversity.similarity_rank`.
- **own facts**: the track's known estimates (`estimated instrumental, ambient, calm`) and measurements (`72 BPM, low energy for this catalog (measured)`), skipping any attribute the lead already stated. Missing values are omitted.

- [ ] **Step 1: Rewrite `TestShortReason` to the new wording and add honesty tests**

```python
class TestShortReason:
    def test_filters_come_first(self, catalog):
        out = explained_search(catalog, QUERY, query_text="calm", attributes=ATTRS,
                               filters=AttributeFilters(vocals="instrumental"), k=1)
        assert short_reason(out.results[0].evidence) == (
            'Instrumental (model estimate) · #1 of 2 for "calm" · '
            'estimated calm · 124 BPM, medium energy for this catalog (measured)'
        )

    def test_query_agreement_next(self, catalog):
        out = explained_search(catalog, QUERY, query_text="relaxing", attributes=ATTRS, k=1)
        assert short_reason(out.results[0].evidence) == (
            'Matches "relaxing" (calm) in your search (model estimate) · #1 of 4 for "relaxing" · '
            'estimated vocal, jazz · 120 BPM, low energy for this catalog (measured)'
        )

    def test_similar_songs_name_what_they_share(self, catalog):
        out = explained_similar(catalog, "a", attributes=ATTRS, k=2)
        assert short_reason(out.results[1].evidence) == (
            "Both calm (model estimates), similar tempo · #2 of 3 closest in sound to Title a · "
            "estimated instrumental · medium energy for this catalog (measured)"
        )

    def test_otherwise_only_the_match(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=None, k=1)
        assert short_reason(out.results[0].evidence) == '#1 of 4 for "q"'


class TestWhyIsHonest:
    def test_missing_attributes_are_never_mentioned(self, catalog):
        # c: vocals unknown, no mood, no tempo; genre jazz and energy known
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS, k=2)
        why = short_reason(out.results[1].evidence)
        assert out.results[1].track_id == "c"
        assert "jazz" in why and "high energy" in why
        for absent in ("BPM", "tempo", "vocal", "instrumental", "calm", "sad"):
            assert absent not in why

    def test_estimates_and_measurements_are_labelled(self, catalog):
        why = short_reason(explained_search(catalog, QUERY, query_text="q",
                                            attributes=ATTRS, k=1).results[0].evidence)
        assert "estimated vocal, jazz, calm, sad" in why
        assert "120 BPM, low energy for this catalog (measured)" in why

    def test_each_result_reads_differently(self, catalog):
        out = explained_search(catalog, QUERY, query_text="q", attributes=ATTRS, k=4)
        whys = [short_reason(r.evidence) for r in out.results]
        assert len(set(whys)) == 4

    def test_variety_note_only_when_moved_up(self, catalog):
        ev = explained_search(catalog, QUERY, query_text="q", attributes=None, k=1).results[0].evidence
        moved = {**ev, "diversity": {"position": 2, "similarity_rank": 3}}
        kept = {**ev, "diversity": {"position": 3, "similarity_rank": 2}}
        assert "chosen for variety" in short_reason(moved)
        assert "chosen for variety" not in short_reason(kept)
```

Add to `tests/test_api.py::TestRecommendationsContract`:

```python
    def test_blend_why_names_both_contributions(self, client):
        body = client.post("/recommendations",
                           json={"query": "toward a", "seed": {"providerId": "b"}}).json()
        why = body["results"][0]["why"]
        assert 'for "toward a" and #' in why and "closest to Title b (equal weight)" in why
```

- [ ] **Step 2: Run, expect failures** in `TestShortReason`, `TestWhyIsHonest`, the blend test.

- [ ] **Step 3: Implement in `evidence.py`**

```python
def _anchor(match: dict[str, object]) -> str:
    if match["kind"] == "query":
        return f"#{match['rank']} of {match['of']:,} for \"{match['query']}\""
    if match["kind"] == "seed":
        return f"#{match['rank']} of {match['of']:,} closest in sound to {match['seed_title']}"
    return (f"#{match['query_rank']} for \"{match['query']}\" and "
            f"#{match['seed_rank']} closest to {match['seed_title']} (equal weight)")


def _lead(evidence: dict[str, object]) -> tuple[str | None, set[str]]:
    """The strongest true statement, and which attributes it already names."""
    if "filters_passed" in evidence:
        entries = evidence["filters_passed"]
        names = {"tempo_bpm" if e["attribute"] == "tempo" else e["attribute"] for e in entries}
        return ", ".join(_describe_filter(e) for e in entries), names
    if "query_agreement" in evidence:
        terms = evidence["query_agreement"]["terms"]
        return (f"matches {', '.join(_describe_term(t) for t in terms)} in your search "
                "(model estimate)", {t["attribute"] for t in terms})
    shared = evidence.get("shared_with_seed")
    if shared:
        parts, names = [], set()
        if "estimates" in shared:
            parts.append("both " + ", ".join(_describe_estimates(shared["estimates"]))
                         + " (model estimates)")
            names |= set(shared["estimates"]) - {"kind"}
        measured = shared.get("measured", {})
        if "tempo_bpm" in measured:
            parts.append("similar tempo")
            names.add("tempo_bpm")
        if "energy" in measured:
            parts.append(f"both {measured['energy']} energy")
            names.add("energy")
        return ", ".join(parts), names
    return None, set()


def _own_facts(evidence: dict[str, object], skip: set[str]) -> list[str]:
    est = {k: v for k, v in evidence.get("estimates", {}).items() if k != "kind" and k not in skip}
    meas = {k: v for k, v in evidence.get("measured", {}).items() if k != "kind" and k not in skip}
    facts = []
    if _describe_estimates(est):
        facts.append("estimated " + ", ".join(_describe_estimates(est)))
    if _describe_measured(meas):
        facts.append(", ".join(_describe_measured(meas)) + " (measured)")
    return facts


def short_reason(evidence: dict[str, object]) -> str:
    """One specific line per result. Every clause is a measured or estimated fact:
    the lead (filters / search agreement / shared with seed), the actual rank that
    put the track here, a variety note when the rerank moved it up, then the
    track's own known attributes. Unknown attributes are never mentioned."""
    lead, named = _lead(evidence)
    parts = [p for p in (lead, _anchor(evidence["match"])) if p]
    moved = evidence.get("diversity", {})
    if moved and moved["position"] < moved["similarity_rank"]:
        parts.append("chosen for variety")
    parts += _own_facts(evidence, named)
    return _capitalize(" · ".join(parts))
```

Keep `summarize` unchanged (it is the long form and its tests stay). In `api/service.py::_blend`, set `why = short_reason(evidence)`.

- [ ] **Step 4: Run full suite, expect PASS.** Run `probe_real.py`; confirm the 5 mood results' `why` lines are pairwise distinct and no line names an attribute absent from that track's `estimates`/`measured`.

- [ ] **Step 5: Update `api/README.md` `why` example and commit** `feat(evidence): rank-specific why lines with labelled own attributes`

---

### Task 4: Filters constrain every mode

**Files:**
- Test: `tests/test_evidence.py` (`TestFilterBoundaries`), `tests/test_api.py` (`TestFiltersThroughTheApi`)
- Modify: only if a test exposes a defect (`recommendation_engine/filters.py` or `api/app.py`).

- [ ] **Step 1: Write the tests**

```python
class TestFilterBoundaries:
    @pytest.mark.parametrize("filters, passes", [
        (AttributeFilters(tempo_min=120.0), True),
        (AttributeFilters(tempo_max=120.0), True),
        (AttributeFilters(tempo_min=120.0, tempo_max=120.0), True),
        (AttributeFilters(tempo_min=120.01), False),
        (AttributeFilters(tempo_max=119.99), False),
    ])
    def test_tempo_bounds_are_inclusive(self, filters, passes):
        outcome = apply_filters(["a"], {"a": ATTRS["a"]}, filters)
        assert ("a" in outcome.passed) is passes

    @pytest.mark.parametrize("bpm", [30.0, 300.0])
    def test_tempo_limits_themselves_are_accepted(self, bpm):
        AttributeFilters(tempo_min=bpm).validate()

    @pytest.mark.parametrize("bpm", [29.99, 300.01])
    def test_tempo_just_outside_the_limits_is_rejected(self, bpm):
        with pytest.raises(ValueError):
            AttributeFilters(tempo_max=bpm).validate()

    @pytest.mark.parametrize("percentile, low, medium", [
        (1 / 3 - 1e-9, True, False), (1 / 3, False, True),
    ])
    def test_energy_band_edges(self, percentile, low, medium):
        attrs = {"x": track("vocal", "jazz", percentile, 100.0)}
        assert ("x" in apply_filters(["x"], attrs, AttributeFilters(energy="low")).passed) is low
        assert ("x" in apply_filters(["x"], attrs, AttributeFilters(energy="medium")).passed) is medium

    def test_mood_filter_passes_any_detected_mood_and_hides_undetected_as_unknown(self):
        outcome = apply_filters(list(ATTRS), ATTRS, AttributeFilters(mood="sad"))
        assert set(outcome.passed) == {"a"}
        assert outcome.unknown == {"c"}
        assert outcome.mismatched == {"b", "d"}
```

```python
def satisfies(result_id, filters, attrs):
    from recommendation_engine.filters import AttributeFilters, apply_filters
    return result_id in apply_filters([result_id], attrs, AttributeFilters(**filters)).passed


class TestFiltersThroughTheApi:
    EMPTY = TestDiscoverPagePayload.EMPTY

    @pytest.mark.parametrize("filters", [
        {"mood": "calm"}, {"energy": "low"}, {"tempo_min": "100", "tempo_max": "122"},
        {"mood": "calm", "energy": "medium"},
    ])
    @pytest.mark.parametrize("extra", [
        {"query": "toward a"}, {"seed": {"providerId": "d", "title": "Title d", "artist": "Artist Three"}},
        {"query": "toward a", "seed": {"providerId": "d", "title": "Title d", "artist": "Artist Three"}},
    ])
    def test_every_result_satisfies_every_active_filter(self, client, filters, extra):
        payload = {**extra, "filters": {**self.EMPTY, **filters}, "k": 5}
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        numeric = {k: (float(v) if k.startswith("tempo") else v) for k, v in filters.items()}
        for result in response.json()["results"]:
            assert satisfies(result["providerId"], numeric, ATTRS)

    @pytest.mark.parametrize("tempo, status", [("30", 200), ("300", 200), ("29", 422), ("301", 422)])
    def test_tempo_limits_from_the_page(self, client, tempo, status):
        payload = {"query": "toward a", "filters": {**self.EMPTY, "tempo_min": tempo}, "k": 5}
        assert client.post("/recommendations", json=payload).status_code == status

    def test_filters_without_attributes_are_refused(self, catalog):
        with TestClient(create_app(Recommender(catalog, None, StubEncoder()))) as c:
            response = c.post("/recommendations",
                              json={"query": "toward a", "filters": {**self.EMPTY, "mood": "calm"}})
        assert response.status_code == 422
        assert "attributes" in response.json()["detail"]
```

- [ ] **Step 2: Run.** Expected: PASS if the current filters are correct. If any fail, use superpowers:systematic-debugging, fix the defect in `filters.py`/`app.py`, rerun.

- [ ] **Step 3: Commit** `test(filters): boundaries, page payloads and every-mode constraint`

---

### Task 5: Verification and docs

- [ ] **Step 1:** Update `recommendation_engine/PLAN.md` row "Artist cap / diversity" to `done` with the measured effect (small reorders; cross-query hubs remain), and keep the central README links intact.
- [ ] **Step 2:** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --offline pytest -q -rs`; `npm test`; `npm run lint`; back up `.next` to scratchpad, `npm run build`; `git diff --check main...HEAD`.
- [ ] **Step 3:** Start the API on 8002 (`INDEX_DIR=.index-fma-small`, offline) and `next start -p 3002` with `RECOMMENDATION_API_URL=http://127.0.0.1:8002`; drive `/discover` in a browser: mood search, seed search, combined, each filter, read the why lines, press play. Screenshot each.
- [ ] **Step 4:** Recompute the asset fingerprints and compare with `scratchpad/assets-step0.txt`; count MP3s (8,000) and confirm `.index-fma-small/CURRENT` unchanged.
- [ ] **Step 5:** Report changed files, exact counts, browser evidence, preservation and anything unverified. Stop for review.
