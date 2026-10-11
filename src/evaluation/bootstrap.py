"""Confidence intervals for retrieval metrics, by resampling queries.

Why: with a few dozen queries, a gap of 0.03 in mean precision can be pure
luck of which queries were asked. Resampling the per-query scores with
replacement and recomputing the mean many times shows how much the mean
would wobble, giving a 95% interval instead of a bare number.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_resamples: int = 5000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the mean of per-query `values`.

    Percentile bootstrap; the fixed seed keeps reported intervals
    reproducible between runs.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_diff_ci(a, b, n_resamples: int = 5000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """CI for mean(a - b) where a[i] and b[i] are two systems' scores on
    the same query i.

    Pairing matters: both systems face the same easy and hard queries, so
    the per-query difference is far less noisy than comparing two
    independent intervals. If the interval excludes 0 the gap is unlikely
    to be chance.
    """
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_resamples, confidence, seed)


def fmt_ci(values, digits: int = 3) -> str:
    """'0.643 [0.500, 0.790]' for table cells."""
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
