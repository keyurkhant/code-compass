from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — LiteLLM model string; examples:
    #   claude-sonnet-4-5       (Anthropic — needs ANTHROPIC_API_KEY)
    #   gpt-4o                  (OpenAI   — needs OPENAI_API_KEY)
    #   gemini/gemini-1.5-pro   (Google   — needs GEMINI_API_KEY)
    #   ollama/llama3           (local Ollama — no key needed)
    llm_model: str = "claude-sonnet-4-5"

    # Embeddings — fastembed model name; code-optimised options:
    #   jinaai/jina-embeddings-v2-base-code  (768-dim, best for code)
    #   BAAI/bge-small-en-v1.5               (384-dim, fastest)
    embedding_model_name: str = "jinaai/jina-embeddings-v2-base-code"
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
