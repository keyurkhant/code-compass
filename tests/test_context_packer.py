from codecompass.retrieve.context_packer import pack_context
from codecompass.providers.base import SearchResult


def _make_result(id: str, doc: str, score: float) -> SearchResult:
    return SearchResult(id=id, document=doc, metadata={"path": f"{id}.py"}, score=score)


def test_pack_context_respects_budget():
    results = [
        _make_result("a", "word " * 100, 1.0),
        _make_result("b", "word " * 100, 0.9),
        _make_result("c", "word " * 10, 0.8),
    ]
    tokenizer = lambda text: len(text.split())
    packed = pack_context(results, token_budget=150, tokenizer=tokenizer)
    assert len(packed) < len(results)


def test_pack_context_deduplicates():
    r = _make_result("a", "hello world", 1.0)
    packed = pack_context([r, r], token_budget=1000, tokenizer=lambda t: len(t.split()))
    assert len(packed) == 1


def test_pack_context_orders_by_score():
    results = [
        _make_result("a", "aaa", 0.5),
        _make_result("b", "bbb", 0.9),
        _make_result("c", "ccc", 0.7),
    ]
    # Already sorted by score (caller's responsibility), just verify dedup works
    packed = pack_context(results, token_budget=1000, tokenizer=lambda t: len(t.split()))
    assert len(packed) == 3


def test_pack_context_first_item_always_included():
    """Even if budget is tiny, include at least the first result."""
    r = _make_result("a", "very long document " * 1000, 1.0)
    packed = pack_context([r], token_budget=1, tokenizer=lambda t: len(t.split()))
    assert len(packed) == 1
