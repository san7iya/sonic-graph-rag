"""Neighbour-agreement evaluation: for each song, do its k nearest
neighbours share its curated genre? Compared across the text, image,
audio and fused spaces, against the random-neighbour chance rate.

Uses cached vectors only (no models, no Gemini). Writes
results_neighbors.md. Rerun after encoding more tracks.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation.neighbor_agreement import chance_same_genre, same_genre_at_k
from src.preprocessing.records import load_valid_tracks
from src.rag import similar

K = 5
OUT = config.ROOT_DIR / "results_neighbors.md"

if __name__ == "__main__":
    tracks = {t["track_id"]: t for t in load_valid_tracks()}
    ids, mats = similar.load_corpus_vectors(list(tracks))
    genres = [tracks[t]["genre"] for t in ids]
    counts = {g: genres.count(g) for g in sorted(set(genres))}

    # Equal-weight fused similarity = sum of the three cosines, which is
    # the same as the inner product of the concatenated (unit-block) vectors.
    fused = np.concatenate([mats[m] for m in similar.MODALITIES], axis=1)

    rows = {m: same_genre_at_k(mats[m], genres, K) for m in similar.MODALITIES}
    rows["fused (equal weights)"] = same_genre_at_k(fused, genres, K)
    chance = chance_same_genre(genres)
    # The knowledge graph's SIMILAR_TO edges are each song's top-3 fused neighbours.
    edge_k = 3
    edge_rate = same_genre_at_k(fused, genres, edge_k)

    lines = ["# Neighbour agreement (same-genre@%d)" % K, "",
             f"{len(ids)} tracks with all three modalities encoded. Genre counts: {counts}. "
             f"For each song, the fraction of its {K} nearest other songs that share its curated genre. "
             f"Random neighbours would score {chance:.3f}.", "",
             "| Space | same-genre@%d |" % K, "|---|---|"]
    lines += [f"| {name} | {score:.3f} |" for name, score in rows.items()]
    lines += [f"| SIMILAR_TO graph edges (top-{edge_k}, fused) | {edge_rate:.3f} |"]
    if len(counts) < 6 or min(counts.values()) < 10:
        lines += ["", "**Note: the encoded corpus is still small or unbalanced across genres; treat these "
                      "numbers as provisional.**"]
    # Does a shared artist-level bio inflate same-artist matches in text space?
    # Same-artist@K in the text space, split by where each track's bio came from.
    artists = [tracks[t]["artist"] for t in ids]
    sources = [tracks[t].get("bio_source", "seed") for t in ids]
    groups = {src: [i for i, s in enumerate(sources) if s == src] for src in sorted(set(sources))}
    lines += ["", "## Bio source breakdown (text space and fused)", "",
              "Tracks whose bio is the artist's article share text with that artist's other tracks, which could "
              "inflate same-artist neighbours. Same-genre and same-artist neighbour rates by the query song's bio source:",
              "", "| Bio source | tracks | text same-genre@%d | text same-artist@%d | fused same-artist@%d |" % (K, K, K),
              "|---|---|---|---|---|"]
    for src, rows_idx in groups.items():
        lines.append(f"| {src} | {len(rows_idx)} | {same_genre_at_k(mats['text'], genres, K, rows_idx):.3f} | "
                     f"{same_genre_at_k(mats['text'], artists, K, rows_idx):.3f} | "
                     f"{same_genre_at_k(fused, artists, K, rows_idx):.3f} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
