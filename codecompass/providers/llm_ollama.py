"""LLM provider for Ollama (local models, no API key needed).

Calls Ollama's OpenAI-compatible endpoint at localhost:11434/v1/chat/completions.
Install Ollama: https://ollama.ai  then: ollama pull llama3.2
"""

import logging

import httpx

from codecompass.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete(
        self, messages: list[dict], *, max_tokens: int = 2048, system: str | None = None
    ) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        try:
            response = httpx.post(
                f"{self._base_url}/v1/chat/completions",
                json={"model": self._model, "messages": all_messages, "max_tokens": max_tokens},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: https://ollama.ai"
            ) from e

        return response.json()["choices"][0]["message"]["content"]
