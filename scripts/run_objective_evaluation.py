"""Retrieval evaluation on metadata-derived queries (artist and album
lookups), with bootstrap confidence intervals.

Relevance comes from metadata, not from hand labels or genre labels, so
there are far more queries than the 14 hand-labelled ones and the result
is independent of both. Metric: R-precision (precision at k = number of
relevant songs for that query), so queries with 3 or 8 relevant songs are
scored on the same footing. Compares fused / text / image / audio and
reports the paired fused-vs-text difference with its interval.

Needs preprocessing and encoding done. Writes results_objective.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import run_encoder_subprocess
from src.evaluation.bootstrap import fmt_ci, paired_diff_ci
from src.evaluation.objective_queries import album_queries, artist_queries
from src.evaluation.precision_at_k import precision_at_k, reciprocal_rank
from src.preprocessing.records import load_valid_tracks
from src.rag import similar

OUT = config.ROOT_DIR / "results_objective.md"
MODS = similar.MODALITIES
SYSTEMS = {"fused (equal weights)": (1, 1, 1), "text only": (1, 0, 0), "image only": (0, 1, 0), "audio only": (0, 0, 1)}


def unit(v):
    return v / np.linalg.norm(v)


def evaluate(queries, ids, mats, qvecs):
    """Per-system arrays of per-query R-precision and reciprocal rank."""
    out = {name: ([], []) for name in SYSTEMS}
    for i, q in enumerate(queries):
        cos = np.stack([mats[m] @ unit(qvecs[m][f"q{i}"]) for m in MODS])
        rel = q["relevant"] & set(ids)
        for name, w in SYSTEMS.items():
            ranked = [ids[j] for j in np.argsort(-(np.array(w) @ cos))]
            out[name][0].append(precision_at_k(ranked, rel, len(rel)))
            out[name][1].append(reciprocal_rank(ranked, rel))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tracks = load_valid_tracks()
    ids, mats = similar.load_corpus_vectors([t["track_id"] for t in tracks])
    in_corpus = [t for t in tracks if t["track_id"] in set(ids)]

    queries = artist_queries(in_corpus) + album_queries(in_corpus)
    queries_file = config.CACHE_DIR / "objective_queries.json"
    queries_file.write_text(json.dumps([q["query"] for q in queries]), encoding="utf-8")
    paths = {m: config.CACHE_DIR / f"objective_query_{m}_vectors.npz" for m in MODS}
    for m in MODS:
        run_encoder_subprocess([m, "--queries-file", str(queries_file), "--out", str(paths[m])])
    qvecs = {m: dict(np.load(paths[m])) for m in MODS}

    lines = ["# Objective (metadata-derived) retrieval evaluation", "",
             f"{len(ids)} tracks. Relevance = the track's artist (or artist+album) matches the query; "
             "metric = R-precision with 95% bootstrap intervals over queries. These are name-lookup queries: "
             "they show how well each modality connects a name to its songs, not mood or sound.", ""]
    for kind in ("artist", "album"):
        sel = [i for i, q in enumerate(queries) if q["kind"] == kind]
        if not sel:
            continue
        res = evaluate([queries[i] for i in sel], ids, mats, {m: {f"q{j}": qvecs[m][f"q{i}"] for j, i in enumerate(sel)} for m in MODS})
        lines += [f"## {kind.capitalize()} queries (n={len(sel)})", "", "| System | R-precision [95% CI] | MRR [95% CI] |", "|---|---|---|"]
        lines += [f"| {n} | {fmt_ci(res[n][0])} | {fmt_ci(res[n][1])} |" for n in SYSTEMS]
        d, lo, hi = paired_diff_ci(res["fused (equal weights)"][0], res["text only"][0])
        lines += ["", f"Fused minus text-only (paired): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]. "
                      + ("Interval excludes 0: the difference is unlikely to be chance." if lo > 0 or hi < 0
                         else "Interval includes 0: no reliable difference."), ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
