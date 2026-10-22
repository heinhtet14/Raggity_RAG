"""One SQLite file holds everything: collections, documents, chunks, vectors and a BM25 index.

- Vectors are float32 blobs. Search loads a collection's matrix and takes dot products
  (exact, brute force). That's fine up to ~100k chunks; past that you'd want an
  approximate index (HNSW in Qdrant, pgvector, LanceDB...). See docs/concepts.md.
- Keyword search uses SQLite's built-in FTS5, which ranks with BM25.
- Deleting a document deletes its chunks, vectors and FTS rows in one transaction, so a
  removed document can never come back in search results.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from raggity.ingest.chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    embedder TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    chunk_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_collection ON chunks(collection_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
-- Keep the FTS index in step with the chunks table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
"""


@dataclass(frozen=True)
class Collection:
    id: str
    name: str
    description: str
    embedder: str
    dim: int
    created_at: str


@dataclass(frozen=True)
class Document:
    id: str
    collection_id: str
    filename: str
    metadata: dict[str, Any]
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class StoredChunk:
    id: str
    document_id: str
    filename: str
    page: int | None
    index: int
    text: str
    metadata: dict[str, Any]


class NotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()  # one writer at a time; SQLite serialises anyway

    # -- collections ------------------------------------------------------------------

    def create_collection(self, name: str, description: str, embedder: str, dim: int) -> Collection:
        collection = Collection(uuid.uuid4().hex, name, description, embedder, dim, _now())
        with self._lock, self._db:
            try:
                self._db.execute(
                    "INSERT INTO collections VALUES (?, ?, ?, ?, ?, ?)",
                    (collection.id, name, description, embedder, dim, collection.created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a collection named {name!r} already exists") from exc
        return collection

    def get_collection(self, collection_id: str) -> Collection:
        row = self._db.execute(
            "SELECT * FROM collections WHERE id = ? OR name = ?", (collection_id, collection_id)
        ).fetchone()
        if row is None:
            raise NotFound(f"collection {collection_id!r} not found")
        return Collection(**dict(row))

    def list_collections(self) -> list[Collection]:
        rows = self._db.execute("SELECT * FROM collections ORDER BY created_at").fetchall()
        return [Collection(**dict(row)) for row in rows]

    def delete_collection(self, collection_id: str) -> None:
        collection = self.get_collection(collection_id)
        with self._lock, self._db:
            # Delete chunks explicitly so the FTS trigger fires for each row.
            self._db.execute("DELETE FROM chunks WHERE collection_id = ?", (collection.id,))
            self._db.execute("DELETE FROM collections WHERE id = ?", (collection.id,))

    # -- documents --------------------------------------------------------------------

    def add_document(
        self,
        collection: Collection,
        filename: str,
        metadata: dict[str, Any],
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> Document:
        if embeddings.shape != (len(chunks), collection.dim):
            raise ValueError(
                f"expected {len(chunks)} vectors of size {collection.dim}, got {embeddings.shape}"
            )
        document = Document(
            uuid.uuid4().hex, collection.id, filename, metadata, len(chunks), _now()
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    collection.id,
                    filename,
                    json.dumps(metadata, sort_keys=True),
                    len(chunks),
                    document.created_at,
                ),
            )
            self._db.executemany(
                "INSERT INTO chunks (id, document_id, collection_id, idx, page, text, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        f"{document.id}:{chunk.index}",
                        document.id,
                        collection.id,
                        chunk.index,
                        chunk.page,
                        chunk.text,
                        vector.astype(np.float32).tobytes(),
                    )
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )
        return document

    def list_documents(self, collection_id: str) -> list[Document]:
        rows = self._db.execute(
            "SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at", (collection_id,)
        ).fetchall()
        return [self._document(row) for row in rows]

    def get_document(self, collection_id: str, document_id: str) -> Document:
        row = self._db.execute(
            "SELECT * FROM documents WHERE id = ? AND collection_id = ?",
            (document_id, collection_id),
        ).fetchone()
        if row is None:
            raise NotFound(f"document {document_id!r} not found")
        return self._document(row)

    def delete_document(self, collection_id: str, document_id: str) -> None:
        self.get_document(collection_id, document_id)
        with self._lock, self._db:
            self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        values = dict(row)
        values["metadata"] = json.loads(values["metadata"])
        return Document(**values)

    # -- retrieval --------------------------------------------------------------------

    def vectors(self, collection_id: str) -> tuple[list[str], np.ndarray]:
        """All chunk ids and their vectors for a collection, for brute-force dense search."""
        rows = self._db.execute(
            "SELECT id, embedding FROM chunks WHERE collection_id = ?", (collection_id,)
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [row["id"] for row in rows]
        matrix = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        return ids, matrix

    def keyword_search(self, collection_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        """BM25 over the chunk text. Returns (chunk_id, score), higher is better."""
        match = _fts_query(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT c.id, bm25(chunks_fts) AS rank FROM chunks_fts"
            " JOIN chunks c ON c.rowid = chunks_fts.rowid"
            " WHERE chunks_fts MATCH ? AND c.collection_id = ?"
            " ORDER BY rank LIMIT ?",
            (match, collection_id, limit),
        ).fetchall()
        # SQLite's bm25() is negative (lower is better); flip it so higher is better.
        return [(row["id"], -row["rank"]) for row in rows]

    def chunks(self, chunk_ids: list[str]) -> dict[str, StoredChunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self._db.execute(
            "SELECT c.id, c.document_id, d.filename, c.page, c.idx, c.text, d.metadata"
            f" FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id IN ({marks})",
            chunk_ids,
        ).fetchall()
        return {
            row["id"]: StoredChunk(
                row["id"],
                row["document_id"],
                row["filename"],
                row["page"],
                row["idx"],
                row["text"],
                json.loads(row["metadata"]),
            )
            for row in rows
        }


def _fts_query(query: str) -> str:
    """Quote each word so user input can't use FTS5 syntax; OR them so any word can match."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words)
