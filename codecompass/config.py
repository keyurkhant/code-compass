from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model_name: str = "claude-sonnet-4-5"

    # Embeddings
    embedding_model_name: str = "microsoft/codebert-base"
    embedding_batch_size: int = 32

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_prefix: str = "codecompass"

    # BM25 index
    bm25_index_path: str = "./data/bm25_index.pkl"

    # Retrieval
    top_k_retrieve: int = 10
    token_budget: int = 6000

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
