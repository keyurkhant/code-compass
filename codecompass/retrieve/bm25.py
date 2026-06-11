from codecompass.index.bm25_indexer import BM25Index
from codecompass.providers.base import SearchResult


def bm25_search(
    query: str,
    index: BM25Index,
    top_k: int,
) -> list[SearchResult]:
    """Return SearchResult list from BM25 index. document field is empty (no content stored)."""
    results = index.search(query, top_k)
    return [
        SearchResult(id=chunk_id, document="", metadata={}, score=score)
        for chunk_id, score in results
    ]
