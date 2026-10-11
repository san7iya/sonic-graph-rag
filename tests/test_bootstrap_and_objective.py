import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci, fmt_ci, paired_diff_ci
from src.evaluation.objective_queries import album_queries, artist_queries


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = [0.0, 1.0] * 20
    mean, lo, hi = bootstrap_ci(values)
    assert mean == pytest.approx(0.5) and lo < 0.5 < hi
    assert bootstrap_ci(values) == (mean, lo, hi)          # fixed seed


def test_bootstrap_ci_is_degenerate_for_constant_scores_and_wider_with_fewer_queries():
    assert bootstrap_ci([0.7] * 10) == pytest.approx((0.7, 0.7, 0.7))
    rng = np.random.default_rng(1)
    few, many = rng.random(8), rng.random(200)
    assert (lambda c: c[2] - c[1])(bootstrap_ci(few)) > (lambda c: c[2] - c[1])(bootstrap_ci(many))


def test_paired_diff_detects_consistent_gap_but_not_noise():
    base = np.random.default_rng(2).random(60)
    _, lo, hi = paired_diff_ci(base + 0.1, base)
    assert lo > 0                                            # consistently better -> excludes 0
    noise = np.random.default_rng(3).normal(0, 0.1, 60)
    _, lo, hi = paired_diff_ci(base + noise, base)
    assert lo < 0 < hi


def test_fmt_ci_format():
    assert fmt_ci([0.5, 0.5]) == "0.500 [0.500, 0.500]"


TRACKS = [
    {"track_id": "a1", "artist": "A", "album": "X"}, {"track_id": "a2", "artist": "A", "album": "X"},
    {"track_id": "a3", "artist": "A", "album": "Y"}, {"track_id": "b1", "artist": "B", "album": "X"},
]


def test_artist_queries_use_min_tracks_and_exact_relevant_sets():
    qs = artist_queries(TRACKS, min_tracks=3)
    assert len(qs) == 1 and qs[0]["query"] == "songs by A" and qs[0]["relevant"] == {"a1", "a2", "a3"}


def test_album_queries_key_by_artist_and_album():
    qs = album_queries(TRACKS, min_tracks=2)
    assert len(qs) == 1                                      # album X exists for A and B but only A has 2 tracks
    assert qs[0]["relevant"] == {"a1", "a2"} and "by A" in qs[0]["query"]
