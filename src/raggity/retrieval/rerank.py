"""Rerankers re-score a shortlist of chunks against the query.

First-stage retrieval embeds the query and each chunk separately (a "bi-encoder"), which is
fast enough to run over every chunk but only roughly accurate. A reranker reads the query and
one chunk together, so it can judge relevance properly, but it is too slow to run on
everything. Hence: retrieve ~30 cheaply, rerank them, keep the top 5.
"""

from __future__ import annotations

import re
from typing import Protocol

from raggity.config import Settings
from raggity.providers.llm import LLM


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class NoReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        raise ValueError("reranking is off; set RAGGITY_RERANKER to cross-encoder or llm")


class CrossEncoderReranker:
    """A small model trained for exactly this job (MS MARCO passage ranking)."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class LLMReranker:
    """Asks the chat model to grade each passage 0-10. Works with any provider, but costs a
    model call per search; compare its ordering with the cross-encoder's."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def score(self, query: str, texts: list[str]) -> list[float]:
        passages = "\n\n".join(f"<passage id={i}>\n{t}\n</passage>" for i, t in enumerate(texts))
        reply = self._llm.complete(
            "You grade how well each passage answers a query. For every passage output one "
            "line 'id: score' with a score from 0 (irrelevant) to 10 (fully answers it). "
            "Output nothing else.",
            f"Query: {query}\n\n{passages}",
        )
        scores = [0.0] * len(texts)
        for match in re.finditer(r"(\d+)\s*:\s*(\d+(?:\.\d+)?)", reply):
            i, s = int(match.group(1)), float(match.group(2))
            if 0 <= i < len(texts):
                scores[i] = s
        return scores


def make_reranker(settings: Settings, llm: LLM) -> Reranker:
    match settings.reranker:
        case "none":
            return NoReranker()
        case "cross-encoder":
            return CrossEncoderReranker(settings.reranker_model)
        case "llm":
            return LLMReranker(llm)
