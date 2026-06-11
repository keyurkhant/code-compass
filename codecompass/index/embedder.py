from codecompass.ingest.models import CodeChunk
from codecompass.providers.base import EmbeddingProvider


def embed_chunks(
    chunks: list[CodeChunk],
    provider: EmbeddingProvider,
    batch_size: int = 32,
) -> list[tuple[CodeChunk, list[float]]]:
    """Embed chunks in batches. Returns (chunk, embedding) pairs."""
    results: list[tuple[CodeChunk, list[float]]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.embed_text() for c in batch]
        embeddings = provider.embed(texts)
        results.extend(zip(batch, embeddings, strict=False))
    return results
