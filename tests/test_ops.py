from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from raggity.ingest.chunking import Chunk
from raggity.main import create_app
from raggity.service import RagService
from raggity.store.sqlite_store import Store

# -- auth ----------------------------------------------------------------------------


@pytest.fixture
def locked(service: RagService) -> TestClient:
    service.settings.api_key = SecretStr("s3cret")
    return TestClient(create_app(service))


def test_requests_without_the_key_are_refused(locked: TestClient) -> None:
    with locked as c:
        assert c.get("/v1/collections").status_code == 401
        assert c.get("/v1/collections", headers={"x-api-key": "wrong"}).status_code == 401
        assert c.get("/metrics").status_code == 401


def test_either_header_form_is_accepted(locked: TestClient) -> None:
    with locked as c:
        assert c.get("/v1/collections", headers={"x-api-key": "s3cret"}).status_code == 200
        bearer = {"authorization": "Bearer s3cret"}
        assert c.get("/v1/collections", headers=bearer).status_code == 200
        assert c.get("/metrics", headers=bearer).status_code == 200


def test_health_stays_open_for_liveness_probes(locked: TestClient) -> None:
    with locked as c:
        assert c.get("/healthz").status_code == 200


# -- timings and metrics -------------------------------------------------------------


def test_search_reports_time_per_stage(client: TestClient) -> None:
    cid = client.post("/v1/collections", json={"name": "t"}).json()["id"]
    upload = client.post(
        f"/v1/collections/{cid}/documents/text", json={"filename": "a.txt", "text": "alpha beta"}
    ).json()
    assert {"chunk", "embed_documents", "store", "total"} <= upload["timings_ms"].keys()

    timings = client.post(f"/v1/collections/{cid}/search", json={"query": "alpha"}).json()[
        "timings_ms"
    ]
    expected = {"load_vectors", "embed_query", "dense_scoring", "bm25", "fusion", "total"}
    assert expected <= timings.keys()
    assert timings["total"] >= timings["embed_query"]


def test_answer_times_generation(client: TestClient) -> None:
    cid = client.post("/v1/collections", json={"name": "t"}).json()["id"]
    client.post(
        f"/v1/collections/{cid}/documents/text", json={"filename": "a.txt", "text": "alpha"}
    )
    timings = client.post(f"/v1/collections/{cid}/answer", json={"query": "alpha"}).json()[
        "timings_ms"
    ]
    assert "generate" in timings


def test_metrics_use_route_templates_not_raw_paths(client: TestClient) -> None:
    cid = client.post("/v1/collections", json={"name": "t"}).json()["id"]
    client.post(f"/v1/collections/{cid}/search", json={"query": "x"})
    body = client.get("/metrics").text
    assert 'route="/v1/collections/{collection_id}/search"' in body
    assert cid not in body
    assert 'raggity_stage_seconds_count{stage="embed_query"}' in body


# -- vector cache --------------------------------------------------------------------


def _add(store: Store, cid: str, text: str, vector: list[float]) -> str:
    collection = store.get_collection(cid)
    doc = store.add_document(
        collection, f"{text}.txt", {}, [Chunk(text, None, 0)], np.array([vector], np.float32)
    )
    return doc.id


def test_the_cache_sees_writes_from_another_process(tmp_path: Path) -> None:
    # Two Store objects on one file stand in for two uvicorn workers.
    reader, writer = Store(tmp_path / "db"), Store(tmp_path / "db")
    cid = writer.create_collection("c", "", "test", 2).id
    _add(writer, cid, "one", [1, 0])
    assert len(reader.vectors(cid)[0]) == 1  # fills the reader's cache

    doc = _add(writer, cid, "two", [0, 1])
    assert len(reader.vectors(cid)[0]) == 2

    writer.delete_document(cid, doc)
    assert len(reader.vectors(cid)[0]) == 1


def test_the_cache_returns_the_same_matrix_until_a_write(tmp_path: Path) -> None:
    store = Store(tmp_path / "db")
    cid = store.create_collection("c", "", "test", 2).id
    _add(store, cid, "one", [1, 0])
    first = store.vectors(cid)[1]
    assert store.vectors(cid)[1] is first
    _add(store, cid, "two", [0, 1])
    assert store.vectors(cid)[1] is not first


def test_with_the_cache_off_every_call_rereads(tmp_path: Path) -> None:
    store = Store(tmp_path / "db", vector_cache=False)
    cid = store.create_collection("c", "", "test", 2).id
    _add(store, cid, "one", [1, 0])
    assert store.vectors(cid)[1] is not store.vectors(cid)[1]


def test_a_database_from_before_versions_is_migrated(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "old")
    db.execute(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,"
        " description TEXT NOT NULL DEFAULT '', embedder TEXT NOT NULL, dim INTEGER NOT NULL,"
        " created_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO collections VALUES ('c1', 'old', '', 'test', 2, 'then')")
    db.commit()
    db.close()

    store = Store(tmp_path / "old")
    assert store.get_collection("c1").version == 0
    _add(store, "c1", "one", [1, 0])
    assert store.get_collection("c1").version == 1
