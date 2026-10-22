"""HTTP routes. Kept thin: parse the request, call RagService, shape the response."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from raggity.api.schemas import (
    AnswerResponse,
    ChunkOut,
    CollectionOut,
    CreateCollection,
    DocumentOut,
    SearchRequest,
    SearchResponse,
    TextDocument,
)
from raggity.ingest.loaders import UnsupportedFileType
from raggity.retrieval.search import Hit
from raggity.service import RagService
from raggity.store.sqlite_store import Collection, Document, NotFound

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

router = APIRouter(prefix="/v1")


def get_service(request: Request) -> RagService:
    return request.app.state.service


Service = Annotated[RagService, Depends(get_service)]


def _collection(service: RagService, collection_id: str) -> Collection:
    try:
        return service.store.get_collection(collection_id)
    except NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _collection_out(service: RagService, c: Collection) -> CollectionOut:
    count = len(service.store.list_documents(c.id))
    return CollectionOut(
        id=c.id,
        name=c.name,
        description=c.description,
        embedder=c.embedder,
        dim=c.dim,
        document_count=count,
        created_at=c.created_at,
    )


def _document_out(d: Document) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        filename=d.filename,
        metadata=d.metadata,
        chunk_count=d.chunk_count,
        created_at=d.created_at,
    )


def _chunk_out(hit: Hit) -> ChunkOut:
    c = hit.chunk
    return ChunkOut(
        chunk_id=c.id,
        document_id=c.document_id,
        filename=c.filename,
        page=c.page,
        text=c.text,
        score=hit.score,
        scores=hit.scores,
        metadata=c.metadata,
    )


# -- collections ---------------------------------------------------------------------


@router.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(body: CreateCollection, service: Service) -> CollectionOut:
    try:
        collection = service.create_collection(body.name, body.description)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _collection_out(service, collection)


@router.get("/collections")
def list_collections(service: Service) -> list[CollectionOut]:
    return [_collection_out(service, c) for c in service.store.list_collections()]


@router.get("/collections/{collection_id}")
def get_collection(collection_id: str, service: Service) -> CollectionOut:
    return _collection_out(service, _collection(service, collection_id))


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, service: Service) -> None:
    service.store.delete_collection(_collection(service, collection_id).id)


# -- documents -----------------------------------------------------------------------


@router.post("/collections/{collection_id}/documents", status_code=status.HTTP_201_CREATED)
def upload_document(
    collection_id: str,
    service: Service,
    file: Annotated[UploadFile, File(description="PDF, Markdown or plain text")],
    metadata: Annotated[str, Form(description='JSON object, e.g. {"user_id": "u1"}')] = "{}",
    chunker: Annotated[Literal["fixed", "recursive"] | None, Form()] = None,
    chunk_size: Annotated[int | None, Form(ge=100, le=8000)] = None,
    chunk_overlap: Annotated[int | None, Form(ge=0, le=2000)] = None,
) -> DocumentOut:
    collection = _collection(service, collection_id)
    # A sync route runs in FastAPI's threadpool, so slow embedding doesn't block the server.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file is larger than 50 MB")
    meta = _parse_metadata(metadata)
    try:
        document = service.ingest_file(
            collection, file.filename or "upload", data, meta, chunker, chunk_size, chunk_overlap
        )
    except UnsupportedFileType as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _document_out(document)


@router.post("/collections/{collection_id}/documents/text", status_code=status.HTTP_201_CREATED)
def add_text(collection_id: str, body: TextDocument, service: Service) -> DocumentOut:
    collection = _collection(service, collection_id)
    try:
        document = service.ingest_text(collection, body.filename, body.text, body.metadata)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _document_out(document)


@router.get("/collections/{collection_id}/documents")
def list_documents(collection_id: str, service: Service) -> list[DocumentOut]:
    collection = _collection(service, collection_id)
    return [_document_out(d) for d in service.store.list_documents(collection.id)]


@router.delete(
    "/collections/{collection_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_document(collection_id: str, document_id: str, service: Service) -> None:
    collection = _collection(service, collection_id)
    try:
        service.store.delete_document(collection.id, document_id)
    except NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


def _parse_metadata(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "metadata must be JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "metadata must be an object")
    return value


# -- retrieval -----------------------------------------------------------------------


@router.post("/collections/{collection_id}/search")
def search(collection_id: str, body: SearchRequest, service: Service) -> SearchResponse:
    """Retrieval only. This is the endpoint to give an agent as a tool: it gets the passages
    and writes its own reply in its own voice."""
    collection = _collection(service, collection_id)
    result = _run_search(service, collection, body)
    return SearchResponse(queries=result.queries, results=[_chunk_out(h) for h in result.hits])


@router.post("/collections/{collection_id}/answer")
def answer(collection_id: str, body: SearchRequest, service: Service) -> AnswerResponse:
    """Retrieval plus a cited answer from the configured LLM."""
    collection = _collection(service, collection_id)
    result = _run_search(service, collection, body)
    reply = service.answer(body.query, result)
    sources = [_chunk_out(h) for h in result.hits]
    return AnswerResponse(
        answer=reply.text,
        citations=[sources[n - 1] for n in reply.cited],
        sources=sources,
        queries=result.queries,
        model=service.llm.name,
    )


def _run_search(service: RagService, collection: Collection, body: SearchRequest):
    try:
        return service.search(
            collection, body.query, body.top_k, body.mode, body.transform, body.rerank, body.filter
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
