"""Entry point: `uv run uvicorn raggity.main:app --reload`."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from raggity.api.routes import router
from raggity.config import get_settings
from raggity.providers.llm import LLMError
from raggity.service import RagService


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Build providers once at startup: loading a local model takes seconds.
        app.state.service = service or RagService(get_settings())
        yield

    app = FastAPI(
        title="Raggity RAG",
        version="0.1.0",
        description="Collections of documents, searchable by an agent, with cited answers.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        s: RagService = app.state.service
        return {"status": "ok", "embedder": s.embedder.name, "llm": s.llm.name}

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    return app


app = create_app()
