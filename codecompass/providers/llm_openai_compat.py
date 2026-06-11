"""LLM provider for any OpenAI-compatible endpoint.

Works with: LM Studio, vLLM, llama.cpp server, Llamafile, LiteLLM proxy, etc.
Config:
    codecompass config set llm.provider openai-compat
    codecompass config set llm.base_url http://localhost:1234/v1
    codecompass config set llm.model mistral-7b
    codecompass config set llm.api_key none   # required by openai SDK but not validated
"""

import logging

import httpx

from codecompass.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "none",
        timeout: int = 120,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def complete(
        self, messages: list[dict], *, max_tokens: int = 2048, system: str | None = None
    ) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json={"model": self._model, "messages": all_messages, "max_tokens": max_tokens},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible endpoint at {self._base_url}. "
                "Check llm.base_url in config."
            ) from e

        return response.json()["choices"][0]["message"]["content"]
