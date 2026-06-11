import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from codecompass.api.routers import ask, health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from codecompass.config.manager import load_config
    from codecompass.generate.answerer import Answerer
    from codecompass.index.bm25_indexer import BM25Index
    from codecompass.providers.factory import get_embedding_provider, get_llm_provider
    from codecompass.providers.vector_store_chroma import ChromaVectorStore

    config = load_config()
    data_dir = config.store.resolved_data_dir()

    llm = get_llm_provider(config)
    embedder = get_embedding_provider(config)
    store = ChromaVectorStore(
        persist_dir=str(data_dir / "chroma"),
        collection_name="codecompass_default",
    )

    bm25 = BM25Index()
    bm25_path = data_dir / "bm25_index.pkl"
    if bm25_path.exists():
        bm25.load(bm25_path)
        logger.info("Loaded BM25 index from %s", bm25_path)

    app.state.answerer = Answerer(llm=llm, embedder=embedder, store=store, bm25=bm25, config=config)
    logger.info("Answerer initialized (provider=%s)", config.llm.provider)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="code-compass API", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(ask.router)
    return app


app = create_app()
