"""Wires the pieces together. Routes call this; it has no HTTP in it, so tests and scripts
can use it directly."""

from __future__ import annotations

from typing import Any

from raggity.config import Settings
from raggity.generation import Answer, answer
from raggity.ingest.chunking import chunk_sections
from raggity.ingest.loaders import Section, load
from raggity.providers.embeddings import Embedder, make_embedder
from raggity.providers.llm import LLM, make_llm
from raggity.retrieval.rerank import Reranker, make_reranker
from raggity.retrieval.search import Mode, Retriever, SearchResult, Transform
from raggity.store.sqlite_store import Collection, Document, Store


class RagService:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or Store(settings.data_dir / "raggity.sqlite3")
        self.embedder = embedder or make_embedder(settings)
        self.llm = llm or make_llm(settings)
        self.reranker = reranker or make_reranker(settings, self.llm)
        self.retriever = Retriever(self.store, self.embedder, self.llm, self.reranker)

    def create_collection(self, name: str, description: str) -> Collection:
        return self.store.create_collection(
            name, description, self.embedder.name, self.embedder.dim
        )

    def ingest_file(
        self,
        collection: Collection,
        filename: str,
        data: bytes,
        metadata: dict[str, Any],
        chunker: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        return self._ingest(
            collection, filename, load(filename, data), metadata, chunker, chunk_size, chunk_overlap
        )

    def ingest_text(
        self, collection: Collection, filename: str, text: str, metadata: dict[str, Any]
    ) -> Document:
        return self._ingest(collection, filename, [Section(text)], metadata, None, None, None)

    def _ingest(
        self,
        collection: Collection,
        filename: str,
        sections: list[Section],
        metadata: dict[str, Any],
        chunker: str | None,
        chunk_size: int | None,
        chunk_overlap: int | None,
    ) -> Document:
        if collection.embedder != self.embedder.name:
            raise ValueError(
                f"collection uses {collection.embedder}; "
                f"the service is running {self.embedder.name}"
            )
        chunks = chunk_sections(
            sections,
            chunker or self.settings.chunker,
            chunk_size or self.settings.chunk_size,
            chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"no text could be extracted from {filename!r}")
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        return self.store.add_document(collection, filename, metadata, chunks, embeddings)

    def search(
        self,
        collection: Collection,
        query: str,
        top_k: int,
        mode: Mode,
        transform: Transform,
        rerank: bool,
        metadata_filter: dict[str, Any],
    ) -> SearchResult:
        return self.retriever.search(
            collection, query, top_k, mode, transform, rerank, metadata_filter or None
        )

    def answer(self, question: str, result: SearchResult) -> Answer:
        return answer(self.llm, question, result.hits)
