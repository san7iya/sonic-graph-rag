"""Does keyword query routing beat equal-weight fusion?

Two query sets, both scored from cached vectors (no models, no Gemini):
  A. the 14 hand-labelled queries over the 18 seed tracks (P@3)
  B. the genre queries over the scaled corpus (P@10), if
     scripts/run_scale_evaluation.py has already cached their vectors

For each set: P@k with equal weights vs with weights chosen by
query_router.route_query, how many queries the router actually changed,
and the per-query difference. The router is only worth enabling if the
routed score is clearly higher on BOTH sets; set A alone is too small
to trust. Writes results_routing.md.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import eval_query_cache_path
from scripts.run_scale_evaluation import GENRE_QUERIES
from src.evaluation.precision_at_k import precision_at_k
from src.preprocessing.records import load_valid_tracks
from src.rag import similar
from src.rag.query_router import EQUAL, route_query

OUT = config.ROOT_DIR / "results_routing.md"
EVAL_QUERIES_JSON = config.PROCESSED_DIR / "eval_queries.json"
MODS = similar.MODALITIES


def unit(v):
    return v / np.linalg.norm(v)


def score_set(queries, relevant, ids, mats, qvec_files, k):
    """Return per-query (equal P@k, routed P@k, weights) for one query set."""
    qv = {m: dict(np.load(qvec_files[m])) for m in MODS}
    out = []
    for i, q in enumerate(queries):
        cos = np.stack([mats[m] @ unit(qv[m][f"q{i}"]) for m in MODS])    # (modalities, tracks)
        results = []
        for w in (EQUAL, route_query(q)):
            ranked = [ids[j] for j in np.argsort(-(np.array(w) @ cos))]
            results.append(precision_at_k(ranked, relevant[i], k))
        out.append((results[0], results[1], route_query(q)))
    return out


def section(title, queries, results, k):
    eq = np.mean([r[0] for r in results])
    ro = np.mean([r[1] for r in results])
    changed = [(q, r) for q, r in zip(queries, results) if r[2] != EQUAL]
    lines = [f"## {title}", "", f"{len(queries)} queries, P@{k}. Router changed the weights on {len(changed)}.", "",
             "| | mean P@%d |" % k, "|---|---|", f"| equal weights | {eq:.3f} |", f"| routed | {ro:.3f} |", ""]
    if changed:
        lines += ["| Query | weights (text, image, audio) | equal | routed |", "|---|---|---|---|"]
        lines += [f"| {q} | {tuple(round(x, 1) for x in r[2])} | {r[0]:.2f} | {r[1]:.2f} |" for q, r in changed]
        lines.append("")
    return lines, eq, ro


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tracks = {t["track_id"]: t for t in load_valid_tracks()}
    lines = ["# Query routing vs equal weights", ""]

    # Set A: hand-labelled, restricted to the 18 seed tracks the labels were written for.
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        seed_ids = [r["track_id"] for r in csv.DictReader(f)]
    ids, mats = similar.load_corpus_vectors([t for t in seed_ids if t in tracks])
    eval_set = json.loads(EVAL_QUERIES_JSON.read_text(encoding="utf-8"))
    res_a = score_set([e["query"] for e in eval_set], [set(e["relevant"]) for e in eval_set], ids, mats,
                      {m: eval_query_cache_path(m) for m in MODS}, k=3)
    sec_a, eq_a, ro_a = section("A. Hand-labelled queries (18 seed tracks)", [e["query"] for e in eval_set], res_a, 3)
    lines += sec_a

    # Set B: genre queries over the full encoded corpus.
    scale_files = {m: config.CACHE_DIR / f"scale_query_{m}_vectors.npz" for m in MODS}
    if all(p.exists() for p in scale_files.values()):
        ids, mats = similar.load_corpus_vectors(list(tracks))
        queries_b = [q for qs in GENRE_QUERIES.values() for q in qs]
        genres_b = [g for g, qs in GENRE_QUERIES.items() for _ in qs]
        rel_b = [{t for t in ids if tracks[t]["genre"] == g} for g in genres_b]
        res_b = score_set(queries_b, rel_b, ids, mats, scale_files, k=10)
        sec_b, eq_b, ro_b = section(f"B. Genre queries ({len(ids)} tracks)", queries_b, res_b, 10)
        lines += sec_b
    else:
        lines += ["## B. Genre queries", "", "Not run yet: run `python -m scripts.run_scale_evaluation` first.", ""]

    lines += ["Reading it: enable routing only if it is clearly higher on both sets. "
              "Differences of a few hundredths on set A are within noise."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
