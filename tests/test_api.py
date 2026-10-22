from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeLLM


def _collection(client: TestClient, name: str = "notes") -> str:
    response = client.post("/v1/collections", json={"name": name})
    assert response.status_code == 201
    return response.json()["id"]


def _add(client: TestClient, cid: str, filename: str, text: str, **metadata) -> str:
    response = client.post(
        f"/v1/collections/{cid}/documents/text",
        json={"filename": filename, "text": text, "metadata": metadata},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "embedder": "hash:256", "llm": "fake"}


def test_collection_names_are_unique(client: TestClient) -> None:
    _collection(client)
    assert client.post("/v1/collections", json={"name": "notes"}).status_code == 409


def test_collections_can_be_fetched_by_name(client: TestClient) -> None:
    cid = _collection(client)
    assert client.get("/v1/collections/notes").json()["id"] == cid


def test_upload_a_file_and_search_it(client: TestClient) -> None:
    cid = _collection(client)
    response = client.post(
        f"/v1/collections/{cid}/documents",
        files={"file": ("guide.md", b"# Tea\n\nGreen tea brews at 80 degrees for two minutes.")},
        data={"metadata": '{"user_id": "u1"}'},
    )
    assert response.status_code == 201, response.text
    assert response.json()["metadata"] == {"user_id": "u1"}

    results = client.post(
        f"/v1/collections/{cid}/search", json={"query": "green tea temperature"}
    ).json()["results"]
    assert results[0]["filename"] == "guide.md"
    assert "80 degrees" in results[0]["text"]


def test_unsupported_files_are_refused(client: TestClient) -> None:
    cid = _collection(client)
    response = client.post(
        f"/v1/collections/{cid}/documents", files={"file": ("photo.png", b"\x89PNG")}
    )
    assert response.status_code == 415


def test_bad_metadata_is_refused(client: TestClient) -> None:
    cid = _collection(client)
    response = client.post(
        f"/v1/collections/{cid}/documents",
        files={"file": ("a.txt", b"hello")},
        data={"metadata": "[1, 2]"},
    )
    assert response.status_code == 422


def test_a_deleted_document_is_no_longer_found(client: TestClient) -> None:
    cid = _collection(client)
    doc = _add(client, cid, "secret.txt", "The vault code is 4417.")
    _add(client, cid, "other.txt", "Nothing interesting here.")

    assert client.delete(f"/v1/collections/{cid}/documents/{doc}").status_code == 204

    for mode in ("dense", "keyword", "hybrid"):
        results = client.post(
            f"/v1/collections/{cid}/search", json={"query": "vault code 4417", "mode": mode}
        ).json()["results"]
        assert all(r["document_id"] != doc for r in results), mode
    assert [d["filename"] for d in client.get(f"/v1/collections/{cid}/documents").json()] == [
        "other.txt"
    ]


def test_deleting_a_collection_removes_everything(client: TestClient) -> None:
    cid = _collection(client)
    _add(client, cid, "a.txt", "alpha")
    assert client.delete(f"/v1/collections/{cid}").status_code == 204
    assert client.get(f"/v1/collections/{cid}").status_code == 404


def test_answer_returns_the_cited_sources(client: TestClient, llm: FakeLLM) -> None:
    cid = _collection(client)
    _add(client, cid, "cats.txt", "Cats sleep twelve to sixteen hours a day.")
    _add(client, cid, "dogs.txt", "Dogs need daily walks.")

    body = client.post(
        f"/v1/collections/{cid}/answer", json={"query": "how much do cats sleep", "top_k": 2}
    ).json()

    assert body["answer"] == "Cats sleep a lot [1]."
    assert [c["filename"] for c in body["citations"]] == ["cats.txt"]
    assert len(body["sources"]) == 2
    system, prompt = llm.calls[-1]
    assert "only the numbered sources" in system
    assert '<source id="1" from="cats.txt">' in prompt


def test_answer_with_no_documents_does_not_call_the_model(client: TestClient, llm: FakeLLM) -> None:
    cid = _collection(client)
    body = client.post(f"/v1/collections/{cid}/answer", json={"query": "anything"}).json()
    assert body["citations"] == []
    assert llm.calls == []


def test_unknown_collection_is_404(client: TestClient) -> None:
    assert client.post("/v1/collections/nope/search", json={"query": "x"}).status_code == 404
