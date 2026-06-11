from codecompass.index.bm25_indexer import BM25Index
from codecompass.providers.base import EmbeddingProvider, SearchResult, VectorStore
from codecompass.retrieve.dense import dense_search


def _reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse multiple ranked lists using RRF. Merges by id."""
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if result.id not in by_id or result.document:
                by_id[result.id] = result

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        SearchResult(id=cid, document=by_id[cid].document, metadata=by_id[cid].metadata, score=s)
        for cid, s in fused
    ]


def hybrid_search(
    query: str,
    provider: EmbeddingProvider,
    store: VectorStore,
    bm25_index: BM25Index | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
    filters: dict | None = None,
) -> list[SearchResult]:
    """
    Hybrid dense + BM25 search with RRF fusion.
    Falls back to dense-only if bm25_index is None or not loaded.
    """
    dense_results = dense_search(query, provider, store, top_k=top_k * 2, filters=filters)

    if bm25_index is None or not bm25_index.is_loaded:
        return dense_results[:top_k]

    from codecompass.retrieve.bm25 import bm25_search

    bm25_results = bm25_search(query, bm25_index, top_k=top_k * 2)
    fused = _reciprocal_rank_fusion([dense_results, bm25_results], rrf_k=rrf_k)
    return fused[:top_k]
