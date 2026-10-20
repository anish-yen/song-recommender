"""Diversity rerank: maximal marginal relevance (MMR) over stored embeddings.

Pure similarity ranking can return near-duplicates: several tracks that sit close to
the query *and* to each other. MMR builds the list one pick at a time and scores
each remaining candidate as

    lambda * relevance(candidate)  -  (1 - lambda) * max cosine(candidate, already picked)

so a track that sounds like something already in the list has to beat it by more.

* ``lambda_ = 1.0`` reproduces plain similarity order exactly (tested), so this can
  ship switched off and be tuned later.
* The weight is a testing guess, not a value derived from data (design doc:
  "initial numbers are testing guesses"). Tune it against the graded evaluation.
* Relevance is the result's own score rescaled to [0, 1] within the pool, because
  the three modes score on different scales (see ``_pool_scaled``). The result's
  ``similarity`` field is never replaced, so explanations remain measured evidence.
* Redundancy is a cosine clipped at 0: sounding opposite earns no variety bonus.
"""

from __future__ import annotations

import numpy as np

from .similarity import Catalog, Recommendation

DEFAULT_LAMBDA = 0.7
#: How many top candidates MMR considers. Larger = more diverse but slower and
#: more likely to reach into weak matches.
DEFAULT_POOL = 50


def mmr_rerank(
    catalog: Catalog,
    ranked: list[Recommendation],
    *,
    k: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[Recommendation]:
    """Reorder an already-ranked candidate pool and keep ``k`` results.

    ``ranked`` must be in descending similarity order (the output of
    ``rank_by_score`` with a ``k`` of the pool size). Ties are broken by the
    candidate's original position, so the output is deterministic.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(ranked) <= 1 or lambda_ == 1.0:
        return list(ranked[:k])

    vectors = np.stack([catalog.vector_of(r.track_id) for r in ranked]).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Cosine between every pair of candidates. An opposite-sounding track is not
    # "more different" than an unrelated one, so a negative cosine earns no bonus.
    pairwise = np.clip(vectors @ vectors.T, 0.0, None)
    relevance = _pool_scaled(np.array([r.similarity for r in ranked], dtype=np.float64))

    picked: list[int] = [0]  # the best match always stays first
    # max similarity of each candidate to anything picked so far
    redundancy = pairwise[0].copy()
    remaining = set(range(1, len(ranked)))

    while remaining and len(picked) < k:
        best_i, best_score = -1, -np.inf
        for i in sorted(remaining):  # sorted -> original-order tie-break
            score = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy[i]
            if score > best_score:
                best_i, best_score = i, score
        picked.append(best_i)
        remaining.remove(best_i)
        redundancy = np.maximum(redundancy, pairwise[best_i])

    return [ranked[i] for i in picked]


def _pool_scaled(scores: np.ndarray) -> np.ndarray:
    """Map the pool's scores onto [0, 1] so ``lambda_`` means the same in every mode.

    Mood-search scores (text-to-audio cosine) span ~0.1 over a 50-track pool,
    similar-song scores (audio-to-audio cosine) ~0.15, and blended scores are sums
    of z-scores. Redundancy is always an audio-to-audio cosine, so only relevance
    needs rescaling. Order is preserved, so ``lambda_ = 1.0`` is still plain order.
    """
    spread = float(scores.max() - scores.min())
    if spread < 1e-12:
        return np.zeros_like(scores)
    return (scores - float(scores.min())) / spread
