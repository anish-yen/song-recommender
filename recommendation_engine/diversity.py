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
* Relevance stays the original cosine score; the result's ``similarity`` field is
  never replaced by the MMR score, so explanations remain measured evidence.
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
    pairwise = vectors @ vectors.T  # cosine between every pair of candidates
    relevance = np.array([r.similarity for r in ranked], dtype=np.float64)

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
