from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raggity.config import Settings
from raggity.main import create_app
from raggity.providers.embeddings import HashEmbedder
from raggity.retrieval.rerank import LLMReranker
from raggity.service import RagService
from raggity.store.sqlite_store import Store


class FakeLLM:
    """Stands in for every chat model so tests never touch the network."""

    name = "fake"

    def __init__(self) -> None:
        self.reply: Callable[[str, str], str] = lambda system, user: "Cats sleep a lot [1]."
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply(system, user)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(tmp_path: Path, llm: FakeLLM) -> RagService:
    settings = Settings(data_dir=tmp_path, chunk_size=200, chunk_overlap=40)
    return RagService(
        settings,
        store=Store(tmp_path / "test.sqlite3"),
        embedder=HashEmbedder(256),
        llm=llm,
        reranker=LLMReranker(llm),
    )


@pytest.fixture
def client(service: RagService) -> Iterator[TestClient]:
    with TestClient(create_app(service)) as c:
        yield c
