import numpy as np

from scripts.run_weight_sweep import score_queries, weight_grid


def test_weight_grid_sums_to_one_and_is_complete():
    grid = weight_grid(0.5)
    assert all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0 for w in grid)
    assert len(grid) == 6  # (a,b) with a+b<=2 on a 0.5 step
    assert (1.0, 0.0, 0.0) in grid and (0.0, 0.0, 1.0) in grid


def test_score_queries_follows_the_weighted_modality():
    ids = ["a", "b"]
    # one query; text prefers "a", audio prefers "b"; image is neutral
    cos = np.array([[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]])
    relevant = [{"a"}]
    assert score_queries(cos, (1.0, 0.0, 0.0), ids, relevant)[0] == 1 / 3   # a ranked first -> hit in top-3
    assert list(np.argsort(-np.tensordot(np.array((0, 0, 1.0)), cos, axes=([0], [1]))[0])) == [1, 0]
