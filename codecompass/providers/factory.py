"""Provider factory with auto-detection.

Imports are done lazily inside functions to avoid pulling in heavy dependencies
at module import time.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import httpx

from codecompass.config.schema import Config
from codecompass.providers.base import EmbeddingProvider, LLMProvider


def get_llm_provider(config: Config) -> LLMProvider:
    """Return the configured LLM provider, auto-detecting if needed."""
    provider = config.llm.provider
    if provider == "auto":
        provider = _auto_detect(config)

    if provider == "claude-code":
        from codecompass.providers.llm_subprocess import ClaudeCodeProvider

        return ClaudeCodeProvider(model=config.llm.model, timeout=config.llm.timeout)

    elif provider == "subprocess":
        from codecompass.providers.llm_subprocess import SubprocessProvider

        if not config.llm.cmd_template:
            raise ValueError("llm.cmd_template must be set when llm.provider=subprocess")
        return SubprocessProvider(cmd_template=config.llm.cmd_template, timeout=config.llm.timeout)

    elif provider == "ollama":
        from codecompass.providers.llm_ollama import OllamaProvider

        return OllamaProvider(
            model=config.llm.model or "llama3.2",
            base_url=config.llm.base_url or "http://localhost:11434",
            timeout=config.llm.timeout,
        )

    elif provider == "openai-compat":
        from codecompass.providers.llm_openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model=config.llm.model or "gpt-4o",
            base_url=config.llm.base_url,
            api_key=config.llm.api_key or "none",
            timeout=config.llm.timeout,
        )

    elif provider == "litellm":
        from codecompass.providers.llm_litellm import LiteLLMProvider

        return LiteLLMProvider(model=config.llm.model or "claude-sonnet-4-5")

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Run: codecompass config set llm.provider <provider>"
        )


def get_embedding_provider(config: Config) -> EmbeddingProvider:
    """Return the configured embedding provider."""
    from codecompass.providers.embedding_fastembed import FastEmbedProvider

    return FastEmbedProvider(
        model_name=config.embedding.model, batch_size=config.embedding.batch_size
    )


def _auto_detect(config: Config) -> str:
    """Detect which provider is available without doing expensive I/O."""
    # 1. claude CLI
    if shutil.which("claude"):
        try:
            r = subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return "claude-code"
        except Exception:
            pass

    # 2. Ollama
    base = config.llm.base_url or "http://localhost:11434"
    try:
        httpx.get(f"{base}/api/tags", timeout=2.0)
        return "ollama"
    except Exception:
        pass

    # 3. Cloud API keys
    if os.environ.get("ANTHROPIC_API_KEY") or config.llm.api_key:
        return "litellm"
    if os.environ.get("OPENAI_API_KEY"):
        return "litellm"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "litellm"
    if config.llm.base_url:
        return "openai-compat"

    raise RuntimeError(
        "No LLM provider detected. Set one with:\n"
        "  codecompass config set llm.provider claude-code   # if you have Claude Code\n"
        "  codecompass config set llm.provider ollama         # if you have Ollama running\n"
        "  codecompass config set llm.provider litellm        # with an API key in env"
    )
