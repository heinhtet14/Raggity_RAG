"""Retrieval: find the chunks most likely to answer a query.

The pipeline, each stage switchable per request so you can compare them:

    query ─► [transform] ─► dense / keyword / hybrid ─► [filter] ─► [rerank] ─► top_k

- dense:   embed the query, cosine similarity against every chunk. Finds paraphrases.
- keyword: BM25 over the words. Finds exact names, codes and rare terms dense misses.
- hybrid:  run both and merge with Reciprocal Rank Fusion (RRF). Usually the best default.
- transform "multi_query": ask the LLM for rephrasings, search each, fuse the results.
- transform "hyde": ask the LLM to write a hypothetical answer and embed that instead,
  because an answer looks more like the stored chunks than a question does.
- rerank: re-score the shortlist with a stronger, slower model (see rerank.py).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from raggity.observability import Timings
from raggity.providers.embeddings import Embedder
from raggity.providers.llm import LLM
from raggity.retrieval.rerank import Reranker
from raggity.store.sqlite_store import Collection, Store, StoredChunk

Mode = Literal["dense", "keyword", "hybrid"]
Transform = Literal["none", "multi_query", "hyde"]

RRF_K = 60  # the constant from the original RRF paper; dampens the weight of top ranks


@dataclass
class Hit:
    chunk: StoredChunk
    score: float
    # How each stage scored this chunk, so you can see why it ranked where it did.
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    hits: list[Hit]
    queries: list[str]  # the queries actually searched, after any transform
    timings: Timings = field(default_factory=Timings)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Score each id by the sum of 1 / (k + rank) over every ranking it appears in.

    RRF only looks at positions, not raw scores, so it can merge BM25 scores (unbounded)
    with cosine similarities (-1..1) without any normalisation.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, llm: LLM, reranker: Reranker) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker

    def search(
        self,
        collection: Collection,
        query: str,
        top_k: int = 5,
        mode: Mode = "hybrid",
        transform: Transform = "none",
        rerank: bool = False,
        metadata_filter: dict[str, Any] | None = None,
        candidates: int = 30,
        timings: Timings | None = None,
    ) -> SearchResult:
        t = timings or Timings()
        if collection.embedder != self.embedder.name:
            raise ValueError(
                f"collection was embedded with {collection.embedder} but the service is "
                f"running {self.embedder.name}; vectors from different models can't be compared"
            )
        with t.stage("query_transform"):
            queries = self._transform(query, transform)
        with t.stage("metadata_filter"):
            allowed = self._allowed_documents(collection.id, metadata_filter)

        rankings: list[list[str]] = []
        stage_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for i, q in enumerate(queries):
            if mode in ("dense", "hybrid"):
                if transform == "hyde":
                    with t.stage("hyde"):
                        text = self._hypothetical_answer(q)
                else:
                    text = q
                ranked = self._dense(collection.id, text, candidates, allowed, t)
                rankings.append([cid for cid, _ in ranked])
                for cid, s in ranked:
                    stage_scores[cid].setdefault(f"dense[{i}]", s)
            if mode in ("keyword", "hybrid"):
                with t.stage("bm25"):
                    ranked = self._keyword(collection.id, q, candidates, allowed)
                rankings.append([cid for cid, _ in ranked])
                for cid, s in ranked:
                    stage_scores[cid].setdefault(f"bm25[{i}]", s)

        with t.stage("fusion"):
            if len(rankings) == 1:
                # A single ranking needs no fusion: keep its own scores.
                fused = {cid: next(iter(stage_scores[cid].values())) for cid in rankings[0]}
            else:
                fused = reciprocal_rank_fusion(rankings)
                for cid, s in fused.items():
                    stage_scores[cid]["rrf"] = s
            shortlist = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:candidates]

        with t.stage("fetch_chunks"):
            chunks = self.store.chunks(shortlist)
        hits = [
            Hit(chunks[cid], fused[cid], stage_scores[cid]) for cid in shortlist if cid in chunks
        ]

        if rerank and hits:
            with t.stage("rerank"):
                new_scores = self.reranker.score(query, [hit.chunk.text for hit in hits])
            for hit, s in zip(hits, new_scores, strict=True):
                hit.scores["rerank"] = s
                hit.score = s
            hits.sort(key=lambda hit: hit.score, reverse=True)

        return SearchResult(hits[:top_k], queries, t)

    # -- stages -----------------------------------------------------------------------

    def _dense(
        self, collection_id: str, text: str, limit: int, allowed: set[str] | None, t: Timings
    ) -> list[tuple[str, float]]:
        with t.stage("load_vectors"):
            ids, matrix = self.store.vectors(collection_id)
        if not ids:
            return []
        with t.stage("embed_query"):
            query_vector = self.embedder.embed_query(text)
        with t.stage("dense_scoring"):
            if allowed is not None:
                keep = [i for i, cid in enumerate(ids) if _document_of(cid) in allowed]
                ids, matrix = [ids[i] for i in keep], matrix[keep]
                if not ids:
                    return []
            similarities = matrix @ query_vector  # cosine: the vectors are unit length
            top = np.argsort(-similarities)[:limit]
            return [(ids[i], float(similarities[i])) for i in top]

    def _keyword(
        self, collection_id: str, query: str, limit: int, allowed: set[str] | None
    ) -> list[tuple[str, float]]:
        # Over-fetch when filtering, since the filter runs after BM25 ranks.
        fetch = limit * 5 if allowed is not None else limit
        ranked = self.store.keyword_search(collection_id, query, fetch)
        if allowed is not None:
            ranked = [(cid, s) for cid, s in ranked if _document_of(cid) in allowed]
        return ranked[:limit]

    def _allowed_documents(
        self, collection_id: str, metadata_filter: dict[str, Any] | None
    ) -> set[str] | None:
        if not metadata_filter:
            return None
        return {
            doc.id
            for doc in self.store.list_documents(collection_id)
            if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
        }

    def _transform(self, query: str, transform: Transform) -> list[str]:
        if transform != "multi_query":
            return [query]
        reply = self.llm.complete(
            "You rewrite search queries. Reply with exactly three alternative phrasings of the "
            "user's query, one per line, with no numbering or commentary.",
            query,
        )
        rewrites = [line.strip() for line in reply.splitlines() if line.strip()][:3]
        return [query, *rewrites]

    def _hypothetical_answer(self, query: str) -> str:
        return self.llm.complete(
            "Write a short passage that would plausibly answer the question, as if quoted from "
            "a document. It is used only to find similar real passages, so do not hedge.",
            query,
        )


def _document_of(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[0]
