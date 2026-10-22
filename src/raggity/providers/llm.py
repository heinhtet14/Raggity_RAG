"""Chat models used for answer generation and, optionally, reranking and query rewriting.

Each provider takes a system prompt plus one user message and returns text. That is all
RAG needs from a model; the companion agent on top can do the multi-turn work.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from raggity.config import Effort, Settings


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class EchoLLM:
    """Returns the prompt it was given. Lets you inspect exactly what a model would see."""

    name = "echo"

    def complete(self, system: str, user: str) -> str:
        return f"[echo provider: no model was called]\n\n{user}"


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int, effort: Effort) -> None:
        import anthropic
        from anthropic.types.beta import BetaOutputConfigParam

        self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
        self._model = model
        self._max_tokens = max_tokens
        self._output_config: BetaOutputConfigParam = {"effort": effort}
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        # Streaming avoids HTTP timeouts on long answers; we only need the final message.
        # fallbacks="default": if a safety classifier declines, the API retries on a
        # fallback model inside the same call instead of returning a refusal.
        with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            output_config=self._output_config,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request")
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAILLM:
    def __init__(self, model: str, base_url: str, max_tokens: int) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url)  # reads OPENAI_API_KEY
        self._model = model
        self._max_tokens = max_tokens
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=300)
        self._model = model
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str) -> str:
        response = self._http.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def make_llm(settings: Settings) -> LLM:
    model = settings.llm_model
    match settings.llm_provider:
        case "echo":
            return EchoLLM()
        case "anthropic":
            return AnthropicLLM(
                model or "claude-opus-5-5", settings.llm_max_tokens, settings.llm_effort
            )
        case "openai":
            return OpenAILLM(
                model or "gpt-5-mini", settings.openai_base_url, settings.llm_max_tokens
            )
        case "ollama":
            return OllamaLLM(model or "llama3.2", settings.ollama_base_url)
