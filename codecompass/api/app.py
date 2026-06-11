import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from codecompass.api.routers import ask, health
from codecompass.config import get_settings
from codecompass.generate.answerer import Answerer
from codecompass.index.bm25_indexer import BM25Index
from codecompass.providers.embedding_fastembed import FastEmbedProvider
from codecompass.providers.llm_litellm import LiteLLMProvider
from codecompass.providers.vector_store_chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    embedder = FastEmbedProvider(settings.embedding_model_name)
    llm = LiteLLMProvider(settings.llm_model)
    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=f"{settings.chroma_collection_prefix}_default",
    )

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25.load(bm25_path)
        logger.info("Loaded BM25 index from %s", bm25_path)

    app.state.answerer = Answerer(
        llm=llm, embedder=embedder, store=store, bm25=bm25, settings=settings
    )
    logger.info("Answerer initialized (model=%s)", settings.llm_model)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="code-compass API", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(ask.router)
    return app


app = create_app()
