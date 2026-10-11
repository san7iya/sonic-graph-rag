"""Retrieval evaluation on the scaled corpus, using genre labels.

Hand-labelling relevance for hundreds of tracks is not practical, so this
uses the one label that exists for every track: the curated genre. Each
genre gets several differently-phrased queries; a retrieved track is
relevant if its genre matches the query's genre. This measures genre-level
retrieval only (not finer attributes like mood), so it complements, not
replaces, the hand-labelled evaluation in run_evaluation.py.

Compares fused / text / image / audio retrieval and the weighted-fusion
grid (with the leave-one-query-out honesty check, see run_weight_sweep.py).

Needs: scripts.preprocess run, all three modalities encoded
(scripts.encode_modality), then run `python -m scripts.run_scale_evaluation`.
Writes results_scale.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import cache_path, run_encoder_subprocess
from scripts.run_weight_sweep import weight_grid
from src.evaluation.bootstrap import fmt_ci
from src.evaluation.precision_at_k import precision_at_k, reciprocal_rank
from src.preprocessing.records import load_valid_tracks

K = 10
MODALITIES = ("text", "image", "audio")
RESULTS_SCALE_MD = config.ROOT_DIR / "results_scale.md"

GENRE_QUERIES = {
    "rock": ["rock music with electric guitars and drums", "a classic rock band anthem", "loud guitar-driven rock song"],
    "pop": ["catchy mainstream pop song", "radio-friendly pop hit with a big chorus", "danceable pop music"],
    "hiphop": ["hip-hop track with rapping", "rap song with strong beats and rhymes", "hip hop and rap music"],
    "jazz": ["jazz music with improvisation", "smooth jazz with saxophone or trumpet", "swing and bebop jazz standards"],
    "classical": ["classical orchestral music", "a symphony or concerto by a classical composer", "baroque or romantic era classical piece"],
    "reggae": ["reggae music with an offbeat rhythm", "Jamaican reggae and ska", "laid-back reggae groove"],
}


def unit_rows(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tracks = load_valid_tracks()
    genre_of = {t["track_id"]: t["genre"] for t in tracks}

    queries, query_genres = [], []
    for genre, qs in GENRE_QUERIES.items():
        queries += qs
        query_genres += [genre] * len(qs)

    queries_file = config.CACHE_DIR / "scale_queries.json"
    queries_file.write_text(json.dumps(queries), encoding="utf-8")
    query_paths = {m: config.CACHE_DIR / f"scale_query_{m}_vectors.npz" for m in MODALITIES}
    for m in MODALITIES:
        run_encoder_subprocess([m, "--queries-file", str(queries_file), "--out", str(query_paths[m])])

    corpus = {m: dict(np.load(cache_path(m))) for m in MODALITIES}
    ids = sorted(t for t in genre_of if all(t in corpus[m] for m in MODALITIES))
    print(f"{len(ids)} tracks encoded in all three modalities (of {len(tracks)} valid).")
    relevant = [{t for t in ids if genre_of[t] == g} for g in query_genres]

    corpus_mat = {m: unit_rows(np.stack([corpus[m][t] for t in ids])) for m in MODALITIES}
    qvecs = {m: dict(np.load(query_paths[m])) for m in MODALITIES}
    cos = np.stack([
        np.stack([corpus_mat[m] @ (qvecs[m][f"q{i}"] / np.linalg.norm(qvecs[m][f"q{i}"])) for m in MODALITIES])
        for i in range(len(queries))
    ])  # (queries, modalities, tracks)

    def per_query(weights):
        scores = np.tensordot(np.array(weights), cos, axes=([0], [1]))
        out_p, out_rr = [], []
        for q in range(len(queries)):
            ranked = [ids[j] for j in np.argsort(-scores[q])]
            out_p.append(precision_at_k(ranked, relevant[q], K))
            out_rr.append(reciprocal_rank(ranked, relevant[q]))
        return np.array(out_p), np.array(out_rr)

    systems = {"fused (equal weights)": (1, 1, 1), "text only": (1, 0, 0),
               "image only": (0, 1, 0), "audio only": (0, 0, 1)}
    results = {name: per_query(w) for name, w in systems.items()}

    grid = weight_grid()
    grid_p = np.stack([per_query(w)[0] for w in grid])           # (grid, queries)
    best = int(grid_p.mean(axis=1).argmax())
    held_out = [grid_p[int(np.delete(grid_p, q, axis=1).mean(axis=1).argmax()), q] for q in range(len(queries))]

    per_genre_lines = []
    for g in GENRE_QUERIES:
        idx = [i for i, qg in enumerate(query_genres) if qg == g]
        per_genre_lines.append(f"| {g} | " + " | ".join(
            f"{results[name][0][idx].mean():.3f}" for name in systems) + " |")

    # Cosine spread per modality across all queries x tracks (scale check).
    spread = {m: (float(cos[:, j].mean()), float(cos[:, j].std())) for j, m in enumerate(MODALITIES)}

    lines = ["# Scaled-corpus evaluation (genre labels)", "",
             f"Corpus: {len(ids)} tracks. {len(queries)} genre queries (3 phrasings x 6 genres); a hit is relevant "
             f"if the track's curated genre equals the query's genre. Metrics at k={K}. "
             f"Random-retrieval P@{K} would be about {np.mean([len(r) for r in relevant]) / len(ids):.2f}.", "",
             f"| System | mean P@{K} [95% CI] | MRR [95% CI] |", "|---|---|---|"]
    lines += [f"| {n} | {fmt_ci(results[n][0])} | {fmt_ci(results[n][1])} |" for n in systems]
    lines += ["", f"## Per-genre P@{K}", "", "| Genre | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)] + per_genre_lines
    lines += ["", "## Weighted fusion (weights text, image, audio)", "",
              f"- Best grid point in-sample: {tuple(round(x, 2) for x in grid[best])} -> {grid_p[best].mean():.3f} (optimistic)",
              f"- Leave-one-query-out estimate: {np.mean(held_out):.3f}",
              f"- Equal-weight baseline: {results['fused (equal weights)'][0].mean():.3f}", "",
              "## Per-modality cosine scale (mean / std)", "", "| Modality | mean | std |", "|---|---|---|"]
    lines += [f"| {m} | {spread[m][0]:.3f} | {spread[m][1]:.3f} |" for m in MODALITIES]
    RESULTS_SCALE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
