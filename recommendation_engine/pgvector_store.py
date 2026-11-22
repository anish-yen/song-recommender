"""Similar-song retrieval over PostgreSQL + pgvector.

The design doc's P0 asks for seed-song similarity on pgvector. This module is that
port. ``similarity.similar_to`` (exact NumPy cosine) stays the reference: for any
seed, ``PgVectorStore.similar_to`` must return the same tracks in the same order —
descending cosine, ties by ``track_id``, seed excluded, ``exclude`` removed before
the ``k`` cut, optional per-artist cap. ``check`` below measures exactly that.

Search here is **exact** (a sequential scan with ``ORDER BY distance, track_id``).
An HNSW index would be approximate and could disagree with the reference, so it is
deliberately not created until ``check`` shows exact search is too slow for the
2-second target on the real catalog (REVIEW.md: compare exact results first).

Vectors live in their own Postgres schema, ``recommender``, so Prisma (which owns
``public``) never sees them as drift. Rows are partitioned by the embedding-space
fingerprint, so a rebuilt index with a different model can never be compared
against old vectors.

    export PGVECTOR_URL="postgresql://user:pass@host:5432/postgres"
    uv run python -m recommendation_engine.pgvector_store load  --index-dir .index
    uv run python -m recommendation_engine.pgvector_store check --index-dir .index
    uv run python -m recommendation_engine.pgvector_store similar <track_id> -k 5

The API uses it when ``VECTOR_BACKEND=pgvector`` (see ``api/README.md``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .config import EmbeddingSpace
from .errors import EmbeddingSpaceMismatch, IndexCorruptError, RecommenderError
from .similarity import Catalog, Recommendation, _validate_k
from .similarity import similar_to as numpy_similar_to

SCHEMA = "recommender"
TABLE = f"{SCHEMA}.track_embeddings"

#: Cosine scores closer than this are treated as equal when comparing against the
#: NumPy reference. pgvector accumulates in float32, NumPy in float64, so the two
#: legitimately differ around the 7th decimal; near-ties may swap order.
AGREEMENT_TOLERANCE = 1e-5


class VectorStoreError(IndexCorruptError):
    """The pgvector copy of the index is unreachable or disagrees with the index.

    A subclass of ``IndexCorruptError`` so the API answers 503 ("index
    unavailable"), the same as a missing local index.
    """


def schema_sql(dim: int) -> list[str]:
    """DDL for the vector table. Idempotent; safe to run on every load."""
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            track_id          text          NOT NULL,
            space_fingerprint text          NOT NULL,
            model_name        text          NOT NULL,
            model_version     text          NOT NULL,
            embedding         vector({int(dim)}) NOT NULL,
            created_at        timestamptz   NOT NULL DEFAULT now(),
            PRIMARY KEY (space_fingerprint, track_id)
        )
        """,
    ]


UPSERT_SQL = f"""
    INSERT INTO {TABLE} (track_id, space_fingerprint, model_name, model_version, embedding)
    VALUES (%s, %s, %s, %s, %s::vector)
    ON CONFLICT (space_fingerprint, track_id) DO UPDATE
       SET embedding     = EXCLUDED.embedding,
           model_name    = EXCLUDED.model_name,
           model_version = EXCLUDED.model_version,
           created_at    = now()
"""

DELETE_STALE_SQL = f"""
    DELETE FROM {TABLE}
     WHERE space_fingerprint = %(space)s
       AND track_id <> ALL(%(keep)s::text[])
"""

TRACK_IDS_SQL = f"SELECT track_id FROM {TABLE} WHERE space_fingerprint = %(space)s"

HAS_TRACK_SQL = f"""
    SELECT 1 FROM {TABLE} WHERE space_fingerprint = %(space)s AND track_id = %(seed)s
"""

#: Nearest neighbours of a stored seed. ``<=>`` is pgvector's cosine distance
#: (1 - cosine similarity). ``track_id`` as the second sort key gives the same tie
#: order as the NumPy reference, and also stops the planner from using an
#: approximate index, which is what keeps this search exact.
NEIGHBOURS_SQL = f"""
    SELECT e.track_id, e.embedding <=> s.embedding AS distance
      FROM {TABLE} AS e
      JOIN {TABLE} AS s
        ON s.space_fingerprint = e.space_fingerprint
       AND s.track_id = %(seed)s
     WHERE e.space_fingerprint = %(space)s
       AND e.track_id <> ALL(%(exclude)s::text[])
     ORDER BY distance, e.track_id
     LIMIT %(limit)s
"""


def vector_literal(vector: np.ndarray) -> str:
    """pgvector's text form. ``.9g`` round-trips every float32 exactly."""
    return "[" + ",".join(format(float(x), ".9g") for x in np.asarray(vector, np.float32)) + "]"


def _validate_cap(max_per_artist: int | None) -> None:
    if max_per_artist is None:
        return
    if isinstance(max_per_artist, bool) or not isinstance(max_per_artist, int):
        raise ValueError("max_per_artist must be an integer or None")
    if max_per_artist < 1:
        raise ValueError(f"max_per_artist must be at least 1, got {max_per_artist}")


class PgVectorStore:
    """One embedding space's vectors in Postgres, queried like ``similarity.similar_to``.

    ``connect`` returns a new DB-API connection (psycopg 3 in production). One
    connection is shared and guarded by a lock: FastAPI runs routes in worker
    threads, and a query takes milliseconds, so serialising costs little.
    """

    def __init__(self, connect: Callable[[], object], space: EmbeddingSpace) -> None:
        self._connect = connect
        self.space = space
        self._conn = None
        self._lock = threading.Lock()

    @classmethod
    def from_url(cls, url: str, space: EmbeddingSpace) -> "PgVectorStore":
        if not url:
            raise VectorStoreError("set PGVECTOR_URL to a Postgres connection string")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise VectorStoreError("pgvector backend needs psycopg; run 'uv sync'") from exc

        def connect():
            # prepare_threshold=None: Supabase's pooler (pgbouncer, transaction mode)
            # cannot hold server-side prepared statements across transactions.
            return psycopg.connect(url, autocommit=True, prepare_threshold=None)

        return cls(connect, space)

    # ------------------------------------------------------------------ plumbing

    def _run(self, work: Callable[[object], object]) -> object:
        """Run ``work(cursor)``, reconnecting once if the connection dropped."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None or getattr(self._conn, "closed", False):
                        self._conn = self._connect()
                    with self._conn.cursor() as cursor:
                        return work(cursor)
                except RecommenderError:
                    raise
                except Exception as exc:  # driver errors: psycopg.Error and friends
                    self._close_quietly()
                    if attempt == 2 or not _is_connection_error(exc):
                        raise VectorStoreError(f"pgvector query failed: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    def _close_quietly(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_quietly()

    # ---------------------------------------------------------------- writing

    def create_schema(self) -> None:
        def work(cursor):
            for statement in schema_sql(self.space.dim):
                cursor.execute(statement)
        self._run(work)

    def load_catalog(self, catalog: Catalog) -> dict[str, int]:
        """Make the table hold exactly this catalog's vectors for its space.

        One transaction: upsert every row, then delete rows for tracks no longer in
        the index. Readers see the old set or the new set, never a mix.
        """
        self._require_space(catalog)
        rows = [
            (track_id, catalog.space.fingerprint(), catalog.space.model_id,
             catalog.space.model_revision, vector_literal(vector))
            for track_id, vector in zip(catalog.track_ids, catalog.vectors)
        ]
        self.create_schema()
        before = len(self.track_ids())

        def work(cursor):
            cursor.execute("BEGIN")
            try:
                cursor.executemany(UPSERT_SQL, rows)
                cursor.execute(DELETE_STALE_SQL, {"space": catalog.space.fingerprint(),
                                                  "keep": list(catalog.track_ids)})
                removed = cursor.rowcount
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
            return removed

        removed = int(self._run(work) or 0)
        return {"written": len(rows), "removed": max(removed, 0),
                "new": max(len(rows) - (before - max(removed, 0)), 0)}

    # ---------------------------------------------------------------- reading

    def track_ids(self) -> set[str]:
        def work(cursor):
            try:
                cursor.execute(TRACK_IDS_SQL, {"space": self.space.fingerprint()})
            except Exception as exc:
                if _is_missing_table(exc):
                    return []
                raise
            return [row[0] for row in cursor.fetchall()]
        return set(self._run(work))

    def verify_matches(self, catalog: Catalog) -> None:
        """Refuse to serve when the database and the local index hold different tracks."""
        self._require_space(catalog)
        stored = self.track_ids()
        expected = set(catalog.track_ids)
        if stored != expected:
            missing, extra = len(expected - stored), len(stored - expected)
            raise VectorStoreError(
                f"pgvector holds {len(stored)} tracks for {catalog.space.describe()}, the index "
                f"holds {len(expected)} ({missing} missing, {extra} extra); run "
                "'python -m recommendation_engine.pgvector_store load' for this index")

    def _has_track(self, track_id: str) -> bool:
        def work(cursor):
            cursor.execute(HAS_TRACK_SQL, {"space": self.space.fingerprint(), "seed": track_id})
            return cursor.fetchone() is not None
        return bool(self._run(work))

    def _neighbours(self, seed: str, exclude: Sequence[str], limit: int) -> list[tuple[str, float]]:
        """``(track_id, cosine distance)`` rows, nearest first, ties by id."""
        def work(cursor):
            cursor.execute(NEIGHBOURS_SQL, {"space": self.space.fingerprint(), "seed": seed,
                                            "exclude": list(exclude), "limit": int(limit)})
            return [(str(track_id), float(distance)) for track_id, distance in cursor.fetchall()]
        return self._run(work)

    def similar_to(
        self,
        catalog: Catalog,
        track_id: str,
        *,
        k: int = 10,
        exclude: frozenset[str] = frozenset(),
        max_per_artist: int | None = None,
    ) -> list[Recommendation]:
        """Drop-in for ``similarity.similar_to``: same arguments, same results.

        ``catalog`` supplies titles and artists (the vectors come from Postgres).
        With an artist cap, capped tracks are skipped, so the query asks for more
        rows than ``k`` and asks again with a larger limit until ``k`` results
        survive or the table runs out.
        """
        k = _validate_k(k)
        _validate_cap(max_per_artist)
        self._require_space(catalog)
        catalog.position_of(track_id)  # UnknownTrackError, same as the reference
        if not self._has_track(track_id):
            raise VectorStoreError(
                f"track {track_id!r} is in the index but not in pgvector; reload the vectors")

        excluded = sorted(frozenset(exclude) | {track_id})
        limit = k if max_per_artist is None else max(4 * k, 32)
        while True:
            rows = self._neighbours(track_id, excluded, limit)
            results = self._take(catalog, rows, k=k, max_per_artist=max_per_artist)
            if len(results) == k or len(rows) < limit:
                return results
            limit *= 4

    @staticmethod
    def _take(catalog: Catalog, rows: Iterable[tuple[str, float]], *, k: int,
              max_per_artist: int | None) -> list[Recommendation]:
        """The artist-cap and metadata half of ``similarity.rank_by_score``."""
        results: list[Recommendation] = []
        per_artist: dict[str, int] = {}
        for track_id, distance in rows:
            if len(results) == k:
                break
            meta = catalog.metadata.get(track_id)
            if meta is None:
                raise VectorStoreError(
                    f"pgvector returned {track_id!r}, which the index does not hold; "
                    "reload the vectors")
            artist = str(meta.get("artist", ""))
            if max_per_artist is not None:
                key = artist.casefold()
                if per_artist.get(key, 0) >= max_per_artist:
                    continue
                per_artist[key] = per_artist.get(key, 0) + 1
            similarity = float(np.float32(np.clip(1.0 - distance, -1.0, 1.0)))
            results.append(Recommendation(
                track_id=track_id, title=str(meta.get("title", "")), artist=artist,
                similarity=similarity, provider_url=meta.get("provider_url") or None))
        return results

    def _require_space(self, catalog: Catalog) -> None:
        if catalog.space.fingerprint() != self.space.fingerprint():
            raise EmbeddingSpaceMismatch(
                f"pgvector store is for {self.space.describe()}, "
                f"catalog is {catalog.space.describe()}")


def _is_connection_error(exc: Exception) -> bool:
    return type(exc).__name__ in {"OperationalError", "InterfaceError"}


def _is_missing_table(exc: Exception) -> bool:
    return getattr(exc, "sqlstate", None) in {"42P01", "3F000"}  # undefined table / schema


# ---------------------------------------------------------------- agreement check


@dataclasses.dataclass(frozen=True)
class Agreement:
    seed: str
    verdict: str  # "same" | "near-tie" | "different"
    detail: str = ""


def compare_rankings(seed: str, reference: list[Recommendation],
                     candidate: list[Recommendation],
                     tolerance: float = AGREEMENT_TOLERANCE) -> Agreement:
    """``same``: identical ids and scores within tolerance. ``near-tie``: scores agree
    rank by rank but tracks with (nearly) equal scores swapped. Else ``different``."""
    if len(reference) != len(candidate):
        return Agreement(seed, "different",
                         f"{len(reference)} reference results, {len(candidate)} from pgvector")
    for rank, (ref, got) in enumerate(zip(reference, candidate), start=1):
        if abs(ref.similarity - got.similarity) > tolerance:
            return Agreement(seed, "different",
                             f"rank {rank}: reference {ref.track_id} {ref.similarity:.6f}, "
                             f"pgvector {got.track_id} {got.similarity:.6f}")
    if [r.track_id for r in reference] != [r.track_id for r in candidate]:
        return Agreement(seed, "near-tie", "equal-score tracks in a different order")
    return Agreement(seed, "same")


def pick_seeds(catalog: Catalog, count: int | None) -> list[str]:
    """``count`` seeds spread evenly over the index order; ``None`` = every track."""
    ids = list(catalog.track_ids)
    if count is None or count >= len(ids):
        return ids
    if count < 1:
        raise ValueError("--seeds must be at least 1")
    step = len(ids) / count
    return [ids[int(i * step)] for i in range(count)]


# ------------------------------------------------------------------------ CLI


def _store(args, catalog: Catalog) -> PgVectorStore:
    return PgVectorStore.from_url(args.dsn or os.environ.get("PGVECTOR_URL", ""), catalog.space)


def cmd_load(args) -> int:
    from .index import load_index

    catalog = load_index(args.index_dir)
    store = _store(args, catalog)
    started = time.perf_counter()
    report = store.load_catalog(catalog)
    store.verify_matches(catalog)
    print(f"loaded {report['written']} vectors ({report['new']} new, {report['removed']} "
          f"stale removed) for {catalog.space.describe()} "
          f"in {time.perf_counter() - started:.1f} s")
    return 0


def cmd_check(args) -> int:
    from .index import load_index

    catalog = load_index(args.index_dir)
    store = _store(args, catalog)
    store.verify_matches(catalog)
    seeds = pick_seeds(catalog, args.seeds)
    caps = [None] if args.max_per_artist is None else [None, args.max_per_artist]
    verdicts: dict[str, int] = {"same": 0, "near-tie": 0, "different": 0}
    timings: list[float] = []
    failures: list[Agreement] = []
    for cap in caps:
        for seed in seeds:
            reference = numpy_similar_to(catalog, seed, k=args.k, max_per_artist=cap)
            started = time.perf_counter()
            candidate = store.similar_to(catalog, seed, k=args.k, max_per_artist=cap)
            timings.append((time.perf_counter() - started) * 1000)
            outcome = compare_rankings(seed, reference, candidate)
            verdicts[outcome.verdict] += 1
            if outcome.verdict == "different":
                failures.append(dataclasses.replace(outcome, detail=f"cap={cap}: {outcome.detail}"))
    timings.sort()
    p95 = timings[min(len(timings) - 1, int(0.95 * len(timings)))]
    print(f"{len(seeds)} seeds x {len(caps)} cap setting(s), k={args.k}: "
          f"{verdicts['same']} same, {verdicts['near-tie']} near-tie, "
          f"{verdicts['different']} different")
    print(f"pgvector latency: median {statistics.median(timings):.1f} ms, p95 {p95:.1f} ms, "
          f"max {timings[-1]:.1f} ms")
    for failure in failures[:20]:
        print(f"  DIFFERENT {failure.seed}: {failure.detail}", file=sys.stderr)
    return 1 if failures else 0


def cmd_similar(args) -> int:
    from .index import load_index

    catalog = load_index(args.index_dir)
    store = _store(args, catalog)
    results = store.similar_to(catalog, args.track_id, k=args.k,
                               max_per_artist=args.max_per_artist)
    seed = catalog.metadata[args.track_id]
    print(f"like: {seed.get('title')} — {seed.get('artist')}")
    for rank, rec in enumerate(results, start=1):
        print(f"{rank:>2}. {rec.similarity:+.4f}  {rec.title} — {rec.artist}  [{rec.track_id}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recommendation_engine.pgvector_store",
        description="Load index vectors into pgvector and check them against NumPy.")
    parser.add_argument("--index-dir", default=os.environ.get("INDEX_DIR", ".index"))
    parser.add_argument("--dsn", help="Postgres URL (default: $PGVECTOR_URL)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("load", help="create the table and upsert this index's vectors"
                   ).set_defaults(func=cmd_load)

    check = sub.add_parser("check", help="compare pgvector rankings with the NumPy reference")
    check.add_argument("--seeds", type=int, default=None, help="seeds to test (default: all)")
    check.add_argument("-k", type=int, default=10)
    check.add_argument("--max-per-artist", type=int, default=2,
                       help="also check with this artist cap (default 2, as the product uses)")
    check.set_defaults(func=cmd_check)

    similar = sub.add_parser("similar", help="similar songs for one seed, from pgvector")
    similar.add_argument("track_id")
    similar.add_argument("-k", type=int, default=10)
    similar.add_argument("--max-per-artist", type=int, default=None)
    similar.set_defaults(func=cmd_similar)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RecommenderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
