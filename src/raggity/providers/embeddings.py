"""Embedders turn text into vectors; similar meaning should mean nearby vectors.

Every embedder returns L2-normalised float32 vectors, so cosine similarity is a dot product.
A collection remembers which embedder built it: vectors from two different models live in
different spaces and can't be compared, so mixing them would return nonsense.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

from raggity.config import Settings


class Embedder(Protocol):
    name: str  # "provider:model", stored on the collection
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Feature hashing over word unigrams and bigrams. No model, no network.

    It only captures shared words, not meaning ("car" and "automobile" are unrelated), so
    use it to run the service and tests, then switch to a real model to see the difference.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.lower())
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vec[value % self.dim] += 1.0 if (value >> 63) & 1 else -1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Runs a local model (downloaded from Hugging Face on first use)."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_embedding_dimension() or 0)
        self.name = f"sentence-transformers:{model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalise(self._model.encode(texts, batch_size=32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url)  # reads OPENAI_API_KEY
        self._model = model
        self.name = f"openai:{model}"
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 256):  # stay well under the per-request limit
            response = self._client.embeddings.create(
                model=self._model, input=texts[start : start + 256]
            )
            vectors.extend(item.embedding for item in response.data)
        return normalise(np.array(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=120)
        self._model = model
        self.name = f"ollama:{model}"
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        response = self._http.post("/api/embed", json={"model": self._model, "input": texts})
        response.raise_for_status()
        return normalise(np.array(response.json()["embeddings"]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def make_embedder(settings: Settings) -> Embedder:
    model = settings.embedding_model
    match settings.embedding_provider:
        case "hash":
            return HashEmbedder(int(model) if model else 512)
        case "sentence-transformers":
            return SentenceTransformerEmbedder(model or "BAAI/bge-small-en-v1.5")
        case "openai":
            return OpenAIEmbedder(model or "text-embedding-3-small", settings.openai_base_url)
        case "ollama":
            return OllamaEmbedder(model or "nomic-embed-text", settings.ollama_base_url)
