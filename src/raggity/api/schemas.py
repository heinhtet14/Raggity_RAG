"""Request and response bodies. FastAPI turns these into the OpenAPI spec at /docs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from raggity.retrieval.search import Mode, Transform


class CreateCollection(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""


class CollectionOut(BaseModel):
    id: str
    name: str
    description: str
    embedder: str
    dim: int
    document_count: int
    created_at: str


class TextDocument(BaseModel):
    """Ingest text the agent already has, e.g. a note or a web page it fetched."""

    filename: str = Field(min_length=1, description="a name to cite the text by")
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = {}


class DocumentOut(BaseModel):
    id: str
    filename: str
    metadata: dict[str, Any]
    chunk_count: int
    created_at: str
    timings_ms: dict[str, float] | None = Field(
        None, description="per-stage ingest time; only on the response to an upload"
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    mode: Mode = "hybrid"
    transform: Transform = "none"
    rerank: bool = False
    filter: dict[str, Any] = Field(
        default_factory=dict, description="only documents whose metadata has these values"
    )


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float
    scores: dict[str, float]
    metadata: dict[str, Any]


TIMINGS_DESCRIPTION = "milliseconds per pipeline stage, plus the total"


class SearchResponse(BaseModel):
    queries: list[str]
    results: list[ChunkOut]
    timings_ms: dict[str, float] = Field(description=TIMINGS_DESCRIPTION)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[ChunkOut] = Field(description="the sources the answer cites, in order")
    sources: list[ChunkOut] = Field(description="every source the model was shown")
    queries: list[str]
    model: str
    timings_ms: dict[str, float] = Field(description=TIMINGS_DESCRIPTION)
