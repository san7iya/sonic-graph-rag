"""Do a song's nearest neighbours share its genre?

A label-free-ish sanity check for embedding quality at any corpus size:
for every song, take its k nearest other songs and measure how often
their curated genre matches. Compared with the chance rate (what random
neighbours would score), it shows whether a space (text, image, audio,
fused) has learned anything about genre structure. Also used to judge the
SIMILAR_TO edges added to the knowledge graph.
"""

from __future__ import annotations

import numpy as np


def nearest_neighbors(matrix: np.ndarray, k: int) -> np.ndarray:
    """Row i -> indices of its k most similar other rows (inner product).

    Rows are expected to be (near-)unit vectors, so this is cosine
    similarity; a track is never its own neighbour.
    """
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    k = min(k, matrix.shape[0] - 1)
    return np.argsort(-sims, axis=1)[:, :k]


def same_genre_at_k(matrix: np.ndarray, genres: list[str], k: int) -> float:
    """Mean fraction of each song's k nearest neighbours sharing its genre."""
    labels = np.array(genres)
    neighbors = nearest_neighbors(matrix, k)
    return float(np.mean(labels[neighbors] == labels[:, None]))


def chance_same_genre(genres: list[str]) -> float:
    """Expected same-genre rate if neighbours were drawn at random."""
    labels = np.array(genres)
    n = len(labels)
    same_other = np.array([(labels == g).sum() - 1 for g in labels])
    return float(np.mean(same_other / (n - 1)))
