from collections.abc import Callable

from codecompass.providers.base import SearchResult


def pack_context(
    results: list[SearchResult],
    token_budget: int,
    tokenizer: Callable[[str], int],
) -> list[SearchResult]:
    """
    Greedily select highest-scoring results that fit within the token budget.
    Deduplicates by id.
    """
    seen: set[str] = set()
    kept: list[SearchResult] = []
    used_tokens = 0

    for result in results:
        if result.id in seen:
            continue
        seen.add(result.id)
        cost = tokenizer(result.document)
        if used_tokens + cost > token_budget and kept:
            break
        kept.append(result)
        used_tokens += cost

    return kept
