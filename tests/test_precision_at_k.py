from src.evaluation.precision_at_k import precision_at_k, reciprocal_rank


def test_precision_at_k_counts_relevant_in_top_k_only():
    ranked = ["a", "x", "b", "y"]
    assert precision_at_k(ranked, {"a", "b", "y"}, k=3) == 2 / 3


def test_precision_at_k_penalises_short_result_lists():
    assert precision_at_k(["a"], {"a"}, k=2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
