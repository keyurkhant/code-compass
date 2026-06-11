"""Tests for config manager — pure unit tests, no I/O to real config file."""
import os
from unittest.mock import patch

from codecompass.config.schema import Config


def test_default_config_has_auto_provider():
    config = Config()
    assert config.llm.provider == "auto"


def test_default_embedding_model_is_code_aware():
    config = Config()
    assert "jina" in config.embedding.model or "bge" in config.embedding.model


def test_store_resolved_data_dir_expands_home():
    config = Config()
    resolved = config.store.resolved_data_dir()
    assert not str(resolved).startswith("~")


def test_load_config_returns_defaults_when_no_file(tmp_path):
    from codecompass.config.manager import load_config
    with patch("codecompass.config.manager.CONFIG_PATH", tmp_path / "nonexistent.toml"):
        cfg = load_config()
    assert cfg.llm.provider == "auto"


def test_env_var_overrides_provider(tmp_path):
    from codecompass.config.manager import load_config
    with patch("codecompass.config.manager.CONFIG_PATH", tmp_path / "config.toml"), patch.dict(
        os.environ, {"CODECOMPASS_LLM_PROVIDER": "ollama"}
    ):
        cfg = load_config()
    assert cfg.llm.provider == "ollama"


def test_set_and_get_config_value(tmp_path):
    from codecompass.config.manager import get_config_value, set_config_value
    with patch("codecompass.config.manager.CONFIG_PATH", tmp_path / "config.toml"):
        set_config_value("llm.provider", "ollama", config_path=tmp_path / "config.toml")
        value = get_config_value("llm.provider", config_path=tmp_path / "config.toml")
    assert value == "ollama"
