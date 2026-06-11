"""Universal LLM provider via LiteLLM.

Supports any provider — Anthropic, OpenAI, Google Gemini, Ollama (local, no key),
Azure, Cohere, Mistral, and 100+ others — with a single model string:

    claude-sonnet-4-5            Anthropic (ANTHROPIC_API_KEY)
    gpt-4o                       OpenAI    (OPENAI_API_KEY)
    gemini/gemini-1.5-pro        Google    (GEMINI_API_KEY)
    ollama/llama3                Ollama    (no key — local server)
    openai/gpt-4o                any OpenAI-compatible endpoint (set OPENAI_API_BASE)

Set provider API keys in .env or the environment; LiteLLM picks them up automatically.
"""

from __future__ import annotations

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from codecompass.providers.base import LLMProvider

litellm.drop_params = True  # ignore unsupported params per-provider silently


class LiteLLMProvider(LLMProvider):
    def __init__(self, model: str, **kwargs: object) -> None:
        self._model = model
        self._kwargs = kwargs

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        all_messages: list[dict] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = litellm.completion(
            model=self._model,
            messages=all_messages,
            max_tokens=max_tokens,
            **self._kwargs,
        )
        return response.choices[0].message.content or ""
