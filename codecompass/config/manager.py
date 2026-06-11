"""Config manager for code-compass.

Reads/writes ~/.config/codecompass/config.toml.
Environment variables override file values.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from codecompass.config.schema import (
    Config,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    StoreConfig,
)

CONFIG_PATH: Path = Path("~/.config/codecompass/config.toml").expanduser()

# Module-level cache; None means not yet loaded
_config_cache: Config | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _write_toml(data: dict[str, Any], path: Path) -> None:
    """Write a dict to a TOML file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a Config dataclass to a plain dict suitable for TOML serialisation."""
    return {
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key": config.llm.api_key,
            "cmd_template": config.llm.cmd_template,
            "timeout": config.llm.timeout,
        },
        "embedding": {
            "model": config.embedding.model,
            "batch_size": config.embedding.batch_size,
        },
        "store": {
            "data_dir": config.store.data_dir,
        },
        "retrieval": {
            "top_k": config.retrieval.top_k,
            "token_budget": config.retrieval.token_budget,
        },
    }


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Build a Config dataclass from a (possibly partial) dict."""
    llm_data = data.get("llm", {})
    embedding_data = data.get("embedding", {})
    store_data = data.get("store", {})
    retrieval_data = data.get("retrieval", {})

    llm = LLMConfig(
        provider=llm_data.get("provider", "auto"),
        model=llm_data.get("model", ""),
        base_url=llm_data.get("base_url", ""),
        api_key=llm_data.get("api_key", ""),
        cmd_template=llm_data.get("cmd_template", ""),
        timeout=llm_data.get("timeout", 120),
    )
    embedding = EmbeddingConfig(
        model=embedding_data.get("model", "jinaai/jina-embeddings-v2-base-code"),
        batch_size=embedding_data.get("batch_size", 32),
    )
    store = StoreConfig(
        data_dir=store_data.get("data_dir", "~/.codecompass"),
    )
    retrieval = RetrievalConfig(
        top_k=retrieval_data.get("top_k", 10),
        token_budget=retrieval_data.get("token_budget", 6000),
    )
    return Config(llm=llm, embedding=embedding, store=store, retrieval=retrieval)


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides on top of the loaded config."""
    # Primary CODECOMPASS_* vars
    if val := os.environ.get("CODECOMPASS_LLM_PROVIDER"):
        config.llm.provider = val
    if val := os.environ.get("CODECOMPASS_LLM_MODEL"):
        config.llm.model = val
    if val := os.environ.get("CODECOMPASS_LLM_BASE_URL"):
        config.llm.base_url = val
    if val := os.environ.get("CODECOMPASS_LLM_API_KEY"):
        config.llm.api_key = val
    if val := os.environ.get("CODECOMPASS_EMBEDDING_MODEL"):
        config.embedding.model = val
    if val := os.environ.get("CODECOMPASS_DATA_DIR"):
        config.store.data_dir = val

    # Legacy fallbacks — only set if not already configured
    if not config.llm.api_key and (
        (val := os.environ.get("ANTHROPIC_API_KEY")) or (val := os.environ.get("OPENAI_API_KEY"))
    ):
        config.llm.api_key = val

    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> Config:
    """Load config from TOML file, then apply env-var overrides.

    If the file does not exist, returns a Config with all defaults applied.
    """
    path = config_path if config_path is not None else CONFIG_PATH
    data = _read_toml(path)
    config = _dict_to_config(data)
    return _apply_env_overrides(config)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Write Config to the TOML file."""
    global _config_cache
    path = config_path if config_path is not None else CONFIG_PATH
    _write_toml(_config_to_dict(config), path)
    # Invalidate cache so the next get_config() re-reads
    _config_cache = None


def set_config_value(key: str, value: str, config_path: Path | None = None) -> None:
    """Set a dot-notation config key, e.g. ``set_config_value("llm.provider", "claude-code")``."""
    path = config_path if config_path is not None else CONFIG_PATH
    config = load_config(config_path=path)

    parts = key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Config key must be in 'section.field' format, got: {key!r}")
    section, field_name = parts

    section_obj = getattr(config, section, None)
    if section_obj is None:
        raise ValueError(f"Unknown config section: {section!r}")
    if not hasattr(section_obj, field_name):
        raise ValueError(f"Unknown config field: {field_name!r} in section {section!r}")

    # Coerce to the field's existing type
    existing = getattr(section_obj, field_name)
    if isinstance(existing, int):
        value = int(value)  # type: ignore[assignment]
    setattr(section_obj, field_name, value)

    save_config(config, config_path=path)


def get_config_value(key: str, config_path: Path | None = None) -> str:
    """Get a dot-notation config value as a string."""
    path = config_path if config_path is not None else CONFIG_PATH
    config = load_config(config_path=path)

    parts = key.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Config key must be in 'section.field' format, got: {key!r}")
    section, field_name = parts

    section_obj = getattr(config, section, None)
    if section_obj is None:
        raise ValueError(f"Unknown config section: {section!r}")
    if not hasattr(section_obj, field_name):
        raise ValueError(f"Unknown config field: {field_name!r} in section {section!r}")

    return str(getattr(section_obj, field_name))


def get_config() -> Config:
    """Return a cached singleton Config, loading from disk on first call."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache
