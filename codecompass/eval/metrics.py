import math


def recall_at_k(retrieved_paths: list[str], relevant_paths: list[str], k: int) -> float:
    """Fraction of relevant paths found in top-k retrieved paths."""
    if not relevant_paths:
        return 1.0
    top_k = retrieved_paths[:k]
    hits = sum(1 for p in relevant_paths if any(p in r or r in p for r in top_k))
    return hits / len(relevant_paths)


def reciprocal_rank(retrieved_paths: list[str], relevant_paths: list[str]) -> float:
    """1/rank of the first relevant result, 0 if none found."""
    relevant_set = set(relevant_paths)
    for rank, path in enumerate(retrieved_paths, 1):
        if path in relevant_set or any(rp in path or path in rp for rp in relevant_set):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(per_query_rr: list[float]) -> float:
    if not per_query_rr:
        return 0.0
    return sum(per_query_rr) / len(per_query_rr)


def ndcg_at_k(retrieved_paths: list[str], relevant_paths: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevant_set = set(relevant_paths)

    def _is_relevant(path: str) -> bool:
        return path in relevant_set or any(rp in path or path in rp for rp in relevant_set)

    def _dcg(paths: list[str]) -> float:
        return sum(
            (1.0 / math.log2(rank + 2)) if _is_relevant(p) else 0.0
            for rank, p in enumerate(paths[:k])
        )

    dcg = _dcg(retrieved_paths)
    ideal = _dcg(relevant_paths + [p for p in retrieved_paths[:k] if not _is_relevant(p)])
    return dcg / ideal if ideal > 0 else 0.0
