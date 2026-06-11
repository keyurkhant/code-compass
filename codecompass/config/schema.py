from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "auto"
    # provider values: auto | claude-code | ollama | openai-compat | litellm | subprocess
    model: str = ""  # e.g. llama3.2, claude-sonnet-4-5, gpt-4o
    base_url: str = ""  # for ollama / openai-compat endpoints
    api_key: str = ""  # for litellm only (optional)
    cmd_template: str = ""  # for subprocess: e.g. "claude --bare -p {prompt} --output-format json"
    timeout: int = 120  # seconds


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 256


@dataclass
class StoreConfig:
    data_dir: str = "~/.codecompass"

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


@dataclass
class RetrievalConfig:
    top_k: int = 10
    token_budget: int = 6000


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
