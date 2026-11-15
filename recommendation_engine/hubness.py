"""Hub correction: stop a few tracks from showing up for every query.

In high-dimensional embedding spaces some vectors sit close to *everything*
("hubs"). Plain cosine ranking then returns the same handful of tracks for
unrelated queries — on the 40-track FMA sample one track appeared in 7 of 12 mood
top-5s. The MMR rerank in ``diversity.py`` cannot fix that: it only spreads out the
tracks *within* one list, and a hub is usually the most relevant-looking candidate
in every list.

The correction is CSLS (cross-domain similarity local scaling, Conneau et al. 2018).
Each track gets a penalty ``r(t)``: the mean cosine between the track and its ``k``
nearest *reference queries*. A track that many unrelated queries rate highly has a
high ``r(t)``. Ranking then uses

    adjusted(q, t) = cos(q, t) - weight * r(t)

CSLS proper is ``2 cos - r(q) - r(t)``; for one query ``r(q)`` is a constant, so
``weight = 0.5`` ranks exactly like CSLS and ``weight = 0`` is plain cosine.

Two reference sets, because the two paths see different queries:

* ``audio`` — for similar-song search the queries are other tracks, so ``r(t)`` is
  the mean cosine to the track's ``k`` nearest *other* catalog tracks. Needs only
  the stored vectors.
* ``text`` — for mood search the queries are text, so ``r(t)`` is the mean cosine to
  the ``k`` nearest searches in ``query_bank.json``, encoded with the catalog's own
  CLAP text tower. Needs the encoder once, at build time.

Rules that keep this honest:

* Only the *order* changes. Every result's reported ``similarity`` is still the raw
  cosine, so explanations remain measured evidence; the penalty is recorded in the
  result's evidence.
* Penalties are precomputed per track and stored next to the index
  (``hubness.json``), bound to a digest of the exact vectors they were computed
  from. A rebuilt index makes them stale and they are refused, not silently reused.
* The weight is a testing guess (design doc: "initial numbers are testing
  guesses"). ``hubness_eval.py`` measures repetition and relevance cost on the real
  catalog so it can be tuned or switched off.
* Tune it with MMR switched on, as the service runs. MMR rewards a candidate for
  sounding unlike the list so far, and a demoted hub often does, so MMR can pull
  it back in. On synthetic catalogs 0.5 sometimes let that happen where 0.75 did
  not; 1.0 visibly scrambled relevance. ``hubness_eval.py`` runs the full service
  path for exactly this reason.

For the pgvector port: the penalty is one stored float per track, so the same
ordering is ``ORDER BY (embedding <=> :q) + :weight * penalty`` (``<=>`` is cosine
*distance*, ``1 - cos``). An HNSW index cannot serve that ordering directly; an
exact scan or an over-fetched candidate pool reranked in SQL can.

Build:  ``python -m recommendation_engine.hubness --index-dir .index``
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .errors import IndexCorruptError, RecommenderError
from .similarity import Catalog

#: Bumped when the meaning or layout of ``hubness.json`` changes.
HUBNESS_VERSION = 1
HUBNESS_FILE = "hubness.json"
#: Neighbourhood size for ``r(t)``; CSLS's published default.
DEFAULT_NEIGHBOURS = 10
#: 0.5 ranks exactly like CSLS; 0 switches the correction off.
DEFAULT_HUB_WEIGHT = 0.5
DEFAULT_BANK = Path(__file__).with_name("query_bank.json")
#: Rows per matrix block when finding neighbours; bounds memory on large catalogs.
_CHUNK = 1024

PATHS = ("audio", "text")


# ----------------------------------------------------------------------- maths


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("matrix contains a zero-norm row")
    return matrix / norms


def neighbourhood_similarity(
    targets: np.ndarray,
    references: np.ndarray,
    *,
    k: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Mean cosine between each target row and its ``k`` most similar reference rows.

    ``exclude_self`` is for when targets and references are the same matrix (the
    audio path): row ``i`` is then never its own neighbour. ``k`` larger than the
    references available is an error rather than a silent shrink, because a
    penalty averaged over fewer neighbours is not comparable with one over ``k``.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    targets, references = _unit_rows(targets), _unit_rows(references)
    if targets.shape[1] != references.shape[1]:
        raise ValueError(
            f"targets have {targets.shape[1]} dimensions, references {references.shape[1]}"
        )
    if exclude_self and targets.shape != references.shape:
        raise ValueError("exclude_self needs targets and references to be the same matrix")
    available = references.shape[0] - (1 if exclude_self else 0)
    if k > available:
        raise ValueError(f"k={k} but only {available} reference rows are available")

    out = np.empty(targets.shape[0], dtype=np.float64)
    for start in range(0, targets.shape[0], _CHUNK):
        block = targets[start:start + _CHUNK] @ references.T
        if exclude_self:
            rows = np.arange(block.shape[0])
            block[rows, start + rows] = -np.inf
        # partition picks the k largest without a full sort; their mean does
        # not depend on order, so no tie-break is needed.
        top = np.partition(block, block.shape[1] - k, axis=1)[:, -k:]
        out[start:start + block.shape[0]] = top.mean(axis=1)
    return out


def catalog_digest(catalog: Catalog) -> str:
    """sha256 over track order and the exact stored vectors.

    Penalties are a function of every vector in the catalog, not just one track's,
    so adding, removing or re-embedding any track invalidates all of them.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(list(catalog.track_ids)).encode("utf-8"))
    digest.update(np.ascontiguousarray(catalog.vectors, dtype=np.float32).tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ penalties


@dataclasses.dataclass(frozen=True)
class HubPenalties:
    """Per-track ``r(t)`` for each path, aligned with ``catalog.track_ids``."""

    neighbours: int
    audio: np.ndarray
    text: np.ndarray | None = None
    bank_sha256: str | None = None
    bank_size: int = 0

    def for_path(self, path: str) -> np.ndarray | None:
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, got {path!r}")
        return self.audio if path == "audio" else self.text


@dataclasses.dataclass(frozen=True)
class HubAdjustment:
    """One path's penalties at one weight: what ranking subtracts, and its evidence."""

    path: str
    penalties: np.ndarray
    weight: float
    neighbours: int

    def offsets(self) -> np.ndarray:
        """Subtracted from the path's cosine scores for ordering only."""
        return self.weight * self.penalties

    def evidence(self, position: int) -> dict[str, object]:
        return {
            "method": "csls",
            "reference": "catalog tracks" if self.path == "audio" else "query bank",
            "neighbours": self.neighbours,
            "weight": self.weight,
            "penalty": round(float(self.penalties[position]), 4),
        }


def adjustment(
    penalties: HubPenalties | None, path: str, weight: float
) -> HubAdjustment | None:
    """The adjustment to apply, or ``None`` when it would change nothing."""
    validate_weight(weight)
    if penalties is None or weight == 0.0:
        return None
    values = penalties.for_path(path)
    if values is None:
        return None
    return HubAdjustment(path, values, float(weight), penalties.neighbours)


def validate_weight(weight: float) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"hub weight must be a number, got {type(weight).__name__}")
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError(f"hub weight must be in [0, 1], got {weight}")
    return float(weight)


def compute_penalties(
    catalog: Catalog,
    *,
    bank_vectors: np.ndarray | None = None,
    bank_sha256: str | None = None,
    neighbours: int = DEFAULT_NEIGHBOURS,
) -> HubPenalties:
    """Audio penalties from the catalog itself; text penalties when a bank is given."""
    if len(catalog) == 0:
        raise ValueError("cannot compute hub penalties for an empty catalog")
    audio = neighbourhood_similarity(
        catalog.vectors, catalog.vectors, k=neighbours, exclude_self=True)
    text = None
    if bank_vectors is not None:
        text = neighbourhood_similarity(catalog.vectors, bank_vectors, k=neighbours)
    return HubPenalties(
        neighbours=neighbours,
        audio=audio,
        text=text,
        bank_sha256=bank_sha256 if text is not None else None,
        bank_size=int(bank_vectors.shape[0]) if bank_vectors is not None else 0,
    )


# ------------------------------------------------------------------ query bank


def load_bank(path=DEFAULT_BANK) -> tuple[list[str], str]:
    """Queries from a bank file, plus a sha256 identifying exactly that list.

    Accepts ``{"queries": [...]}`` or a bare list. Blank entries are an error;
    case/whitespace duplicates are dropped so one phrase cannot count twice.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecommenderError(f"query bank {path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise RecommenderError(f"query bank {path} is not valid JSON: {exc}") from exc
    items = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise RecommenderError(f"query bank {path}: expected a non-empty list of queries")
    queries, seen = [], set()
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise RecommenderError(f"query bank {path}: entry {i} is empty or not a string")
        key = " ".join(item.casefold().split())
        if key not in seen:
            seen.add(key)
            queries.append(item.strip())
    canonical = json.dumps(queries, ensure_ascii=False, separators=(",", ":"))
    return queries, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- persistence


def save_hubness(index_dir, catalog: Catalog, penalties: HubPenalties) -> Path:
    """Write ``hubness.json`` atomically next to the index."""
    for path in PATHS:
        values = penalties.for_path(path)
        if values is not None and (values.shape != (len(catalog),)
                                   or not np.all(np.isfinite(values))):
            raise ValueError(f"{path} penalties do not match the catalog")
    def by_track(values):
        return {t: round(float(v), 6) for t, v in zip(catalog.track_ids, values)}

    payload = {
        "version": HUBNESS_VERSION,
        "method": "csls: mean cosine to the k nearest reference queries",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "neighbours": penalties.neighbours,
        "space_fingerprint": catalog.space.fingerprint(),
        "catalog_sha256": catalog_digest(catalog),
        "audio": {"reference": "catalog", "penalties": by_track(penalties.audio)},
        "text": None if penalties.text is None else {
            "reference": "query bank",
            "bank_sha256": penalties.bank_sha256,
            "bank_size": penalties.bank_size,
            "penalties": by_track(penalties.text),
        },
    }
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / HUBNESS_FILE
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".hubness-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, allow_nan=False)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_hubness(index_dir, catalog: Catalog) -> HubPenalties:
    """Penalties for exactly this catalog, or ``IndexCorruptError`` saying why not."""
    path = Path(index_dir) / HUBNESS_FILE
    build = "run 'python -m recommendation_engine.hubness' to build them"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IndexCorruptError(f"no hub penalties at {path}; {build}") from exc
    except json.JSONDecodeError as exc:
        raise IndexCorruptError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != HUBNESS_VERSION:
        raise IndexCorruptError(f"{path}: unsupported hub penalty format; {build}")
    if payload.get("space_fingerprint") != catalog.space.fingerprint():
        raise IndexCorruptError(f"{path} was built for another embedding space; {build}")
    if payload.get("catalog_sha256") != catalog_digest(catalog):
        raise IndexCorruptError(f"{path} was built for a different index build; {build}")
    neighbours = payload.get("neighbours")
    if isinstance(neighbours, bool) or not isinstance(neighbours, int) or neighbours < 1:
        raise IndexCorruptError(f"{path}: invalid neighbours value {neighbours!r}")

    def aligned(section, name) -> np.ndarray:
        values = section.get("penalties") if isinstance(section, dict) else None
        if not isinstance(values, dict):
            raise IndexCorruptError(f"{path}: {name} section has no penalties mapping")
        try:
            out = np.array([float(values[t]) for t in catalog.track_ids], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexCorruptError(f"{path}: {name} penalties are incomplete: {exc}") from exc
        if not np.all(np.isfinite(out)):
            raise IndexCorruptError(f"{path}: {name} penalties are not finite")
        return out

    text_section = payload.get("text")
    return HubPenalties(
        neighbours=neighbours,
        audio=aligned(payload.get("audio"), "audio"),
        text=None if text_section is None else aligned(text_section, "text"),
        bank_sha256=None if text_section is None else text_section.get("bank_sha256"),
        bank_size=0 if text_section is None else int(text_section.get("bank_size", 0)),
    )


# ------------------------------------------------------------------------ CLI


def _encode_bank(encoder, queries: list[str], batch_size: int) -> np.ndarray:
    blocks = [encoder.embed_text(queries[i:i + batch_size])
              for i in range(0, len(queries), batch_size)]
    return np.concatenate(blocks, axis=0)


def _top(catalog: Catalog, values: np.ndarray, n: int = 8) -> list[str]:
    order = np.argsort(-values, kind="stable")[:n]
    lines = []
    for i in order:
        meta = catalog.metadata.get(catalog.track_ids[i], {})
        lines.append(f"  {values[i]:.4f}  {meta.get('title', '')} — {meta.get('artist', '')}"
                     f"  [{catalog.track_ids[i]}]")
    return lines


def main(argv: list[str] | None = None) -> int:
    from .index import load_index

    parser = argparse.ArgumentParser(
        prog="python -m recommendation_engine.hubness",
        description="Precompute per-track hub penalties next to an index.")
    parser.add_argument("--index-dir", default=".index")
    parser.add_argument("--bank", default=str(DEFAULT_BANK),
                        help="JSON list of reference text queries for the mood-search penalty")
    parser.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS)
    parser.add_argument("--audio-only", action="store_true",
                        help="skip the text penalty (no model load); mood search stays uncorrected")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    catalog = load_index(args.index_dir)
    bank_vectors = bank_sha = None
    if not args.audio_only:
        from .embeddings import ClapEncoder

        queries, bank_sha = load_bank(args.bank)
        print(f"encoding {len(queries)} reference queries from {args.bank} ...")
        bank_vectors = _encode_bank(ClapEncoder(space=catalog.space, device=args.device),
                                    queries, args.batch_size)
    penalties = compute_penalties(catalog, bank_vectors=bank_vectors, bank_sha256=bank_sha,
                                  neighbours=args.neighbours)
    target = save_hubness(args.index_dir, catalog, penalties)
    print(f"wrote {target} for {len(catalog)} tracks (k={args.neighbours})")
    for path in PATHS:
        values = penalties.for_path(path)
        if values is None:
            print(f"{path}: not computed")
            continue
        print(f"{path}: penalty mean {values.mean():.4f}, sd {values.std():.4f}; "
              f"highest (likeliest hubs):")
        print("\n".join(_top(catalog, values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
