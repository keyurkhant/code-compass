from codecompass.eval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_perfect():
    assert recall_at_k(["a.py", "b.py", "c.py"], ["a.py", "b.py"], k=5) == 1.0


def test_recall_at_k_none_found():
    assert recall_at_k(["x.py", "y.py"], ["a.py", "b.py"], k=5) == 0.0


def test_recall_at_k_partial():
    result = recall_at_k(["a.py", "x.py", "b.py"], ["a.py", "b.py"], k=2)
    assert result == 0.5  # only a.py in top-2


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a.py", "b.py"], ["a.py"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["x.py", "a.py"], ["a.py"]) == 0.5


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x.py", "y.py"], ["a.py"]) == 0.0


def test_mean_reciprocal_rank():
    mrr = mean_reciprocal_rank([1.0, 0.5, 0.0])
    assert abs(mrr - 0.5) < 1e-9


def test_ndcg_at_k_perfect():
    retrieved = ["a.py", "b.py", "c.py"]
    relevant = ["a.py", "b.py"]
    score = ndcg_at_k(retrieved, relevant, k=3)
    assert score > 0.9


def test_recall_empty_relevant():
    assert recall_at_k(["a.py"], [], k=5) == 1.0
