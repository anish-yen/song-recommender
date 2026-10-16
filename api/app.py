"""HTTP routes over the recommendation engine.

    uv run uvicorn api.app:app --port 8000

Environment:
    INDEX_DIR        index built by ``recommendation_engine.demo build-index`` (default .index)
    DEVICE           cpu (default), mps, cuda, or auto — for query text encoding
    PUBLIC_API_URL   base URL for preview links (default: the URL the request arrived on)

The model is loaded once at startup, and every route is a plain ``def``: FastAPI
runs those in a worker thread, so CPU-bound encoding never blocks the event loop.
"""

from __future__ import annotations

import mimetypes
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from recommendation_engine.errors import (
    EmbeddingError,
    EmbeddingSpaceMismatch,
    IndexCorruptError,
    RecommenderError,
    UnknownTrackError,
)
from recommendation_engine.filters import AttributeFilters

from .service import MAX_K, MAX_QUERY_CHARS, Answer, Ranked, Recommender

#: Artist cap applied by the product endpoint when the caller does not choose one.
DEFAULT_MAX_PER_ARTIST = 2


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocals: str | None = None
    genre: str | None = None
    mood: str | None = None
    energy: str | None = None
    tempo_min: float | None = None
    tempo_max: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        # HTML selects send "" for "any"; that means no filter, not a value.
        return None if isinstance(value, str) and not value.strip() else value

    def engine(self) -> AttributeFilters:
        return AttributeFilters(**self.model_dump())


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    k: int = Field(default=10, ge=1, le=MAX_K)
    max_per_artist: int | None = Field(default=None, ge=1)
    filters: Filters = Filters()


class Seed(BaseModel):
    providerId: str | None = None
    title: str | None = None
    artist: str | None = None


class RecommendationRequest(BaseModel):
    """The frontend's contract (``src/lib/recommendation-client.js``)."""

    query: str = Field(default="", max_length=MAX_QUERY_CHARS)
    seed: Seed | None = None
    filters: Filters = Filters()
    k: int = Field(default=10, ge=1, le=MAX_K)
    max_per_artist: int | None = Field(default=DEFAULT_MAX_PER_ARTIST, ge=1)


def _status_for(exc: Exception) -> int:
    if isinstance(exc, UnknownTrackError):
        return 404
    if isinstance(exc, IndexCorruptError):
        return 503
    if isinstance(exc, EmbeddingSpaceMismatch):
        return 500
    if isinstance(exc, (ValueError, TypeError, EmbeddingError, RecommenderError)):
        return 422
    raise exc


def create_app(recommender: Recommender | None = None) -> FastAPI:
    """Build the app. Tests pass a ready ``Recommender``; production loads one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        if recommender is not None:
            app.state.recommender = recommender
        else:
            try:
                app.state.recommender = Recommender.from_index(
                    os.environ.get("INDEX_DIR", ".index"),
                    device=os.environ.get("DEVICE", "cpu"),
                )
            except RecommenderError as exc:
                # Stay up and answer 503 with the reason, rather than crash-looping.
                app.state.recommender = None
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="song-recommender", version="0.1.0", lifespan=lifespan)

    def service(request: Request) -> Recommender:
        rec = request.app.state.recommender
        if rec is None:
            raise HTTPException(503, request.app.state.startup_error or "index not loaded")
        return rec

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (RecommenderError, ValueError, TypeError) as exc:
            raise HTTPException(_status_for(exc), str(exc)) from exc

    def preview_url(request: Request, rec: Recommender, track_id: str) -> str | None:
        if rec.audio_file(track_id) is None:
            return None
        base = os.environ.get("PUBLIC_API_URL") or str(request.base_url)
        return f"{base.rstrip('/')}/audio/{track_id}"

    def result_json(request: Request, rec: Recommender, item: Ranked) -> dict[str, object]:
        meta = item.metadata
        return {
            "providerId": item.track_id,
            "title": meta.get("title"),
            "artist": meta.get("artist"),
            "album": None,
            "coverArtUrl": None,
            "providerUrl": meta.get("provider_url"),
            "previewUrl": preview_url(request, rec, item.track_id),
            "source": meta.get("source"),
            "license": meta.get("license"),
            "why": item.why,
            "score": round(item.score, 4),
            "similarities": {k: round(v, 4) for k, v in item.similarities.items()},
            "evidence": item.evidence,
        }

    def answer_json(request, rec, answer: Answer, started: float) -> dict[str, object]:
        return {
            "queryId": uuid.uuid4().hex,
            "mode": answer.mode,
            "results": [result_json(request, rec, r) for r in answer.results],
            "hidden": {"mismatch": answer.hidden_mismatch, "unknown": answer.hidden_unknown},
            "elapsedMs": round((time.perf_counter() - started) * 1000, 1),
        }

    @app.get("/health")
    def health(request: Request):
        rec = service(request)
        return {
            "tracks": len(rec.catalog),
            "embeddingSpace": rec.catalog.space.describe(),
            "attributes": rec.attributes is not None,
        }

    @app.get("/filters")
    def filters(request: Request):
        return service(request).filter_options()

    @app.get("/tracks")
    def find_tracks(request: Request, q: str = Query(min_length=1, max_length=200),
                    limit: int = Query(default=10, ge=1, le=50)):
        return {"tracks": service(request).find_tracks(q, limit=limit)}

    @app.get("/tracks/{track_id}")
    def track(request: Request, track_id: str):
        rec = service(request)
        info = call(rec.track, track_id)
        info.pop("audio_path", None)  # a server path, not something clients need
        info["previewUrl"] = preview_url(request, rec, track_id)
        return info

    @app.get("/audio/{track_id}")
    def audio(request: Request, track_id: str):
        path = call(service(request).audio_file, track_id)
        if path is None:
            raise HTTPException(404, "no licensed preview for this track")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.post("/search")
    def search(request: Request, body: SearchRequest):
        started = time.perf_counter()
        rec = service(request)
        answer = call(rec.recommend, query=body.query, filters=body.filters.engine(),
                      k=body.k, max_per_artist=body.max_per_artist)
        return answer_json(request, rec, answer, started)

    @app.get("/similar/{track_id}")
    def similar(request: Request, track_id: str,
                k: int = Query(default=10, ge=1, le=MAX_K),
                max_per_artist: int | None = Query(default=None, ge=1),
                vocals: str | None = None, genre: str | None = None,
                mood: str | None = None, energy: str | None = None,
                tempo_min: float | None = None, tempo_max: float | None = None):
        started = time.perf_counter()
        rec = service(request)
        chosen = call(Filters, vocals=vocals, genre=genre, mood=mood, energy=energy,
                      tempo_min=tempo_min, tempo_max=tempo_max)
        answer = call(rec.recommend, seed_id=track_id, filters=chosen.engine(),
                      k=k, max_per_artist=max_per_artist)
        return answer_json(request, rec, answer, started)

    @app.post("/recommendations")
    def recommendations(request: Request, body: RecommendationRequest):
        started = time.perf_counter()
        rec = service(request)
        seed_id = None
        if body.seed is not None:
            seed_id = call(resolve_seed, rec, body.seed)
        answer = call(rec.recommend, query=body.query, seed_id=seed_id,
                      filters=body.filters.engine(), k=body.k,
                      max_per_artist=body.max_per_artist)
        return answer_json(request, rec, answer, started)

    return app


def resolve_seed(rec: Recommender, seed: Seed) -> str:
    """A seed from the picker carries its id; a typed title must match exactly one track."""
    if seed.providerId:
        rec.catalog.position_of(seed.providerId)
        return seed.providerId
    title = (seed.title or "").strip().casefold()
    if not title:
        raise ValueError("seed needs a providerId or a title")
    matches = [t for t in rec.catalog.track_ids
               if str(rec.catalog.metadata[t].get("title", "")).casefold() == title]
    if len(matches) != 1:
        raise UnknownTrackError(
            f"{len(matches) or 'no'} catalog tracks are titled {seed.title!r}; "
            "pick the seed from the suggestions")
    return matches[0]


app = create_app()
