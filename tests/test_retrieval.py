from __future__ import annotations

import pytest

from raggity.providers.embeddings import HashEmbedder
from raggity.retrieval.search import reciprocal_rank_fusion
from raggity.service import RagService
from tests.conftest import FakeLLM

DOCS = {
    "cats.md": "Cats sleep for twelve to sixteen hours a day. Kittens sleep even more.",
    "dogs.md": "Dogs need daily walks. A border collie needs two hours of exercise.",
    "parts.md": "The replacement filter for the X-200 purifier has part number QF-7731.",
}


@pytest.fixture
def collection(service: RagService):
    c = service.create_collection("pets", "")
    for name, text in DOCS.items():
        service.ingest_text(c, name, text, {"topic": name.split(".")[0]})
    return c


def test_rrf_rewards_items_ranked_well_by_both_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] == fused["b"] > fused["c"] == fused["d"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("mode", ["dense", "keyword", "hybrid"])
def test_each_mode_finds_the_relevant_document(service: RagService, collection, mode) -> None:
    result = service.search(collection, "how long do cats sleep", 3, mode, "none", False, {})
    assert result.hits[0].chunk.filename == "cats.md"


def test_keyword_search_finds_exact_codes(service: RagService, collection) -> None:
    result = service.search(collection, "QF-7731", 3, "keyword", "none", False, {})
    assert result.hits[0].chunk.filename == "parts.md"


def test_hybrid_records_the_score_from_each_stage(service: RagService, collection) -> None:
    hit = service.search(collection, "cats sleep", 1, "hybrid", "none", False, {}).hits[0]
    assert {"dense[0]", "bm25[0]", "rrf"} <= hit.scores.keys()


def test_metadata_filter_restricts_results(service: RagService, collection) -> None:
    result = service.search(collection, "sleep", 5, "hybrid", "none", False, {"topic": "dogs"})
    assert {h.chunk.filename for h in result.hits} == {"dogs.md"}


def test_multi_query_searches_the_rewrites_too(
    service: RagService, collection, llm: FakeLLM
) -> None:
    llm.reply = lambda system, user: "feline nap length\nhours cats spend asleep\nkitten sleep"
    result = service.search(collection, "cat naps", 3, "hybrid", "multi_query", False, {})
    assert result.queries == [
        "cat naps",
        "feline nap length",
        "hours cats spend asleep",
        "kitten sleep",
    ]
    assert result.hits[0].chunk.filename == "cats.md"


def test_hyde_embeds_a_hypothetical_answer(service: RagService, collection, llm: FakeLLM) -> None:
    llm.reply = lambda system, user: "A border collie needs daily walks and exercise."
    result = service.search(collection, "what does my pet need", 1, "dense", "hyde", False, {})
    assert result.hits[0].chunk.filename == "dogs.md"


def test_rerank_reorders_by_the_reranker_score(
    service: RagService, collection, llm: FakeLLM
) -> None:
    def grade(system: str, user: str) -> str:
        # Score whichever passage mentions dogs highest.
        lines = []
        for i, passage in enumerate(user.split("<passage id=")[1:]):
            lines.append(f"{i}: {10 if 'Dogs' in passage else 1}")
        return "\n".join(lines)

    llm.reply = grade
    result = service.search(collection, "sleep", 3, "hybrid", "none", True, {})
    assert result.hits[0].chunk.filename == "dogs.md"
    assert result.hits[0].scores["rerank"] == 10


def test_a_collection_rejects_a_different_embedder(service: RagService, collection) -> None:
    service.retriever.embedder = HashEmbedder(128)
    with pytest.raises(ValueError, match="can't be compared"):
        service.search(collection, "cats", 3, "dense", "none", False, {})
