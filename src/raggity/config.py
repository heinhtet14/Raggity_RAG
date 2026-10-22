"""Settings, read from environment variables prefixed with RAGGITY_ (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["hash", "sentence-transformers", "openai", "ollama"]
LLMProvider = Literal["echo", "anthropic", "openai", "ollama"]
RerankerProvider = Literal["none", "cross-encoder", "llm"]
Chunker = Literal["fixed", "recursive"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGGITY_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # Shared secret callers send as `Authorization: Bearer <key>` or `x-api-key`. Unset means
    # no auth at all, which is only for local development.
    api_key: SecretStr | None = None

    # Keep each collection's vector matrix in memory between queries.
    vector_cache: bool = True

    # Embeddings: "hash" needs nothing installed and is only good for tests and demos.
    embedding_provider: EmbeddingProvider = "hash"
    embedding_model: str = ""  # empty = the provider's default (see providers/embeddings.py)

    # Generation: "echo" returns the retrieved context instead of calling a model.
    llm_provider: LLMProvider = "echo"
    llm_model: str = ""  # empty = the provider's default (see providers/llm.py)
    llm_max_tokens: int = 16000
    llm_effort: Effort = "medium"  # Anthropic only

    reranker: RerankerProvider = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Defaults for ingestion; each upload can override them.
    chunker: Chunker = "recursive"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120

    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()
