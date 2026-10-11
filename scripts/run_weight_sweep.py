"""Weighted-fusion experiment (stretch goal): does weighting the three
modalities beat the equal-weight baseline?

How: the baseline fused inner product is cos_text + cos_image + cos_audio
(each block is unit-norm). A weighted variant is w_t*cos_text +
w_i*cos_image + w_a*cos_audio, which is exactly what concatenating the
blocks scaled by sqrt(w) would give. So every candidate weighting can be
scored from cached vectors with no re-encoding and no FAISS.

Overfitting guard: with only a few labelled queries, picking the best grid
point and reporting its score on the same queries would be optimistic. So
the headline number is leave-one-query-out: for each query, choose the
weights that are best on all the *other* queries, then score that query.
Compare that figure (not the in-sample best) with the equal-weight
baseline before believing any improvement.

Run scripts/run_evaluation.py first (it caches the query vectors).
Appends a "Weight sweep" section to results.md.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import cache_path, eval_query_cache_path
from scripts.run_evaluation import EVAL_QUERIES_JSON, K, MODALITIES, RESULTS_MD, unit
from src.evaluation.precision_at_k import precision_at_k, reciprocal_rank

SECTION_MARK = "## Weight sweep"
GRID_STEP = 0.1


def weight_grid(step: float = GRID_STEP) -> list[tuple[float, float, float]]:
    """All (w_text, w_image, w_audio) with non-negative entries summing to 1."""
    n = round(1 / step)
    return [(a / n, b / n, (n - a - b) / n)
            for a, b in itertools.product(range(n + 1), repeat=2) if a + b <= n]


def score_queries(cos: np.ndarray, weights: tuple, ids: list[str], relevant: list[set],
                  k: int = K) -> np.ndarray:
    """Per-query P@k under one weighting.

    cos has shape (queries, modalities, tracks); the weighted score is a
    plain weighted sum over the modality axis.
    """
    scores = np.tensordot(np.array(weights), cos, axes=([0], [1]))  # (queries, tracks)
    out = []
    for q in range(cos.shape[0]):
        ranked = [ids[j] for j in np.argsort(-scores[q])]
        out.append(precision_at_k(ranked, relevant[q], k))
    return np.array(out)


def mean_rr(cos: np.ndarray, weights: tuple, ids: list[str], relevant: list[set]) -> float:
    scores = np.tensordot(np.array(weights), cos, axes=([0], [1]))
    return float(np.mean([
        reciprocal_rank([ids[j] for j in np.argsort(-scores[q])], relevant[q]) for q in range(cos.shape[0])
    ]))


if __name__ == "__main__":
    eval_set = json.loads(EVAL_QUERIES_JSON.read_text(encoding="utf-8"))
    corpus = {m: dict(np.load(cache_path(m))) for m in MODALITIES}
    queries = {m: dict(np.load(eval_query_cache_path(m))) for m in MODALITIES}

    ids = sorted(corpus["text"].keys())
    # Only queries whose labelled tracks all exist in this corpus are usable.
    usable = [i for i, e in enumerate(eval_set) if set(e["relevant"]) <= set(ids)]
    relevant = [set(eval_set[i]["relevant"]) for i in usable]

    cos = np.stack([
        np.stack([np.stack([unit(corpus[m][t]) for t in ids]) @ unit(queries[m][f"q{i}"]) for m in MODALITIES])
        for i in usable
    ])  # (queries, modalities, tracks)

    grid = weight_grid()
    per_query = np.stack([score_queries(cos, w, ids, relevant) for w in grid])  # (grid, queries)
    baseline = (1 / 3, 1 / 3, 1 / 3)
    base_p = float(score_queries(cos, baseline, ids, relevant).mean())
    base_rr = mean_rr(cos, baseline, ids, relevant)

    best_idx = int(per_query.mean(axis=1).argmax())
    best_w, best_in_sample = grid[best_idx], float(per_query[best_idx].mean())

    # Leave-one-query-out: pick weights on the other queries, score the held-out one.
    held_out = []
    for q in range(len(usable)):
        others = np.delete(per_query, q, axis=1).mean(axis=1)
        held_out.append(per_query[int(others.argmax()), q])
    loo_p = float(np.mean(held_out))

    singles = {m: score_queries(cos, tuple(1.0 if j == k else 0.0 for j in range(3)), ids, relevant).mean()
               for k, m in enumerate(MODALITIES)}

    lines = [SECTION_MARK, "",
             f"{len(usable)} labelled queries over {len(ids)} tracks; weights searched on a grid of step "
             f"{GRID_STEP} ({len(grid)} combinations). Weights are (text, image, audio), summing to 1; "
             f"(0.33, 0.33, 0.33) is the equal-weight baseline.", "",
             "| Setting | mean P@%d |" % K, "|---|---|",
             f"| Baseline equal weights | {base_p:.3f} |",
             f"| Best grid point, in-sample {tuple(round(x, 2) for x in best_w)} (optimistic) | {best_in_sample:.3f} |",
             f"| Leave-one-query-out (honest estimate) | {loo_p:.3f} |"]
    lines += [f"| {m} only | {singles[m]:.3f} |" for m in MODALITIES]
    lines += ["", f"Baseline MRR {base_rr:.3f}.", "",
              "Reading it: a real improvement must show up in the leave-one-query-out row. "
              "If that row is not clearly above the baseline, weighting is not justified on this data; "
              "with this few queries differences under about 0.05 are noise."]

    print("\n".join(lines))
    existing = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    head = existing.split(SECTION_MARK)[0].rstrip()
    tail = ""
    if SECTION_MARK in existing:
        rest = existing.split(SECTION_MARK, 1)[1]
        tail = ("\n\n## " + rest.split("\n## ", 1)[1]) if "\n## " in rest else ""
    RESULTS_MD.write_text(head + "\n\n" + "\n".join(lines) + tail.rstrip() + "\n", encoding="utf-8")
