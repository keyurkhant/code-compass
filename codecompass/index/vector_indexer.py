from codecompass.ingest.models import CodeChunk
from codecompass.providers.base import VectorStore


def index_chunks(
    chunks_with_embeddings: list[tuple[CodeChunk, list[float]]],
    store: VectorStore,
    batch_size: int = 100,
) -> int:
    """Upsert embedded chunks into the vector store. Returns count of upserted docs."""
    total = 0
    for i in range(0, len(chunks_with_embeddings), batch_size):
        batch = chunks_with_embeddings[i : i + batch_size]
        ids = [c.id for c, _ in batch]
        embeddings = [e for _, e in batch]
        documents = [c.embed_text() for c, _ in batch]
        metadatas = [
            {
                "repo": c.repo,
                "path": c.path,
                "language": c.language,
                "symbol_name": c.symbol_name or "",
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c, _ in batch
        ]
        store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        total += len(batch)
    return total
