from codecompass.providers.base import EmbeddingProvider, SearchResult, VectorStore


def dense_search(
    query: str,
    provider: EmbeddingProvider,
    store: VectorStore,
    top_k: int,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Embed the query and run a vector similarity search against the store."""
    [embedding] = provider.embed([query])
    return store.query(embedding=embedding, top_k=top_k, where=filters)
