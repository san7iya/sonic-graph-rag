"""Query-by-example scoring and neighbour-agreement metrics on tiny,
hand-checkable matrices (no models, no cache files).
"""

import numpy as np
import pytest

from scripts.similar_to import resolve_seed
from src import config
from src.evaluation.neighbor_agreement import chance_same_genre, nearest_neighbors, same_genre_at_k
from src.rag.similar import similar_songs, split_fused


def unit_rows(rows):
    m = np.array(rows, dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


IDS = ["seed", "close_audio", "far_audio", "text_match"]
MATS = {
    "audio": unit_rows([[1, 0], [0.9, 0.1], [0, 1], [0, 1]]),
    "text": unit_rows([[0, 1], [0, 1], [0, 1], [1, 0]]),
    "image": unit_rows([[1, 0]] * 4),
}


def test_default_ranks_by_audio_similarity_and_excludes_seed():
    hits = similar_songs("seed", IDS, MATS, top_k=3)
    assert [h["track_id"] for h in hits][0] == "close_audio"
    assert "seed" not in [h["track_id"] for h in hits]


def test_text_modifier_can_pull_a_track_up():
    hits = similar_songs("seed", IDS, MATS, top_k=3, modifier_vecs={"text": np.array([1.0, 0.0])},
                         modifier_weights={"text": 2.0})
    assert hits[0]["track_id"] == "text_match"
    assert "modifier_text" in hits[0] and "seed_audio" in hits[0]


def test_split_fused_uses_config_block_sizes():
    v = np.arange(config.FUSED_EMBED_DIM, dtype=float)
    parts = split_fused(v)
    assert [len(parts[m]) for m in ("text", "image", "audio")] == [
        config.TEXT_EMBED_DIM, config.IMAGE_EMBED_DIM, config.AUDIO_EMBED_DIM]
    assert parts["text"][0] == 0 and parts["audio"][-1] == config.FUSED_EMBED_DIM - 1


def test_nearest_neighbors_excludes_self():
    m = unit_rows([[1, 0], [0.9, 0.1], [0, 1]])
    nn = nearest_neighbors(m, 1)
    assert nn[0, 0] == 1 and nn[2, 0] != 2


def test_same_genre_at_k_and_chance():
    m = unit_rows([[1, 0], [0.95, 0.05], [0, 1], [0.05, 0.95]])
    genres = ["a", "a", "b", "b"]
    assert same_genre_at_k(m, genres, 1) == 1.0
    assert chance_same_genre(genres) == pytest.approx(1 / 3)


def test_resolve_seed_exact_unique_and_ambiguous():
    tracks = [{"track_id": "a_x", "title": "Alpha"}, {"track_id": "a_y", "title": "Alpine"}]
    assert resolve_seed("a_x", tracks)["title"] == "Alpha"
    assert resolve_seed("alpha", tracks)["track_id"] == "a_x"
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_seed("alp", tracks)
    with pytest.raises(ValueError, match="no match"):
        resolve_seed("zzz", tracks)
