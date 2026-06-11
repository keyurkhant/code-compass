from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from codecompass.config import get_settings
from codecompass.generate.answerer import Answerer
from codecompass.index.bm25_indexer import BM25Index
from codecompass.providers.embedding_sentence_transformer import SentenceTransformerEmbeddingProvider
from codecompass.providers.llm_anthropic import AnthropicLLMProvider
from codecompass.providers.vector_store_chroma import ChromaVectorStore
from codecompass.api.routers import ask, health
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    embedder = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    llm = AnthropicLLMProvider(model_name=settings.llm_model_name, api_key=settings.anthropic_api_key)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=f"{settings.chroma_collection_prefix}_default",
    )

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)
        logger.info(f"Loaded BM25 index from {bm25_path}")

    app.state.answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings)
    logger.info("Answerer initialized")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="code-compass API", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(ask.router)
    return app


app = create_app()
