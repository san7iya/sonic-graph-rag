"""Retrieval metrics against manually labelled relevant sets."""

from __future__ import annotations


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant.

    Divides by k (not by len(top_k)) so a retriever that returns fewer
    than k results isn't rewarded for returning less.
    """
    top_k = ranked_ids[:k]
    return sum(1 for track_id in top_k if track_id in relevant_ids) / k


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant result (0 if none retrieved).

    Complements precision@k: P@k says how many good results are in the
    top k, reciprocal rank says how far down you must read to find the
    first good one.
    """
    for rank, track_id in enumerate(ranked_ids, start=1):
        if track_id in relevant_ids:
            return 1.0 / rank
    return 0.0
