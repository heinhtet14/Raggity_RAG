"""Entry point: `uv run uvicorn raggity.main:app --reload`."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from raggity.api.auth import require_api_key
from raggity.api.routes import router
from raggity.config import get_settings
from raggity.observability import REQUEST_SECONDS
from raggity.providers.llm import LLMError
from raggity.service import RagService

log = logging.getLogger("raggity")


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Build providers once at startup: loading a local model takes seconds.
        app.state.service = service or RagService(get_settings())
        if app.state.service.settings.api_key is None:
            log.warning("RAGGITY_API_KEY is not set: every endpoint is open. Dev use only.")
        yield

    app = FastAPI(
        title="Raggity RAG",
        version="0.1.0",
        description="Collections of documents, searchable by an agent, with cited answers.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def record_latency(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        # Label by the route template (/v1/collections/{collection_id}/search), never the raw
        # path: one label per collection id would make the metric grow without bound.
        route = request.scope.get("route")
        REQUEST_SECONDS.labels(
            request.method, getattr(route, "path", "unmatched"), str(response.status_code)
        ).observe(time.perf_counter() - start)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        s: RagService = app.state.service
        return {"status": "ok", "embedder": s.embedder.name, "llm": s.llm.name}

    @app.get("/metrics", dependencies=[Depends(require_api_key)], include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    return app


app = create_app()
