"""FusedVectorIndex is a thin wrapper around faiss.IndexFlatIP -- these
tests use synthetic unit vectors (no model loading) to check the two
things retrieval actually depends on: closer vectors rank higher, and
similarity scores behave like cosine similarity.
"""

import numpy as np
import pytest

from src.indexing.faiss_index import FusedVectorIndex

DIM = 8


def unit_vector(seed: np.ndarray) -> np.ndarray:
    return seed / np.linalg.norm(seed)


def test_search_ranks_closer_vector_first():
    index = FusedVectorIndex(dim=DIM)
    close = unit_vector(np.array([1.0, 0.1, 0, 0, 0, 0, 0, 0]))
    far = unit_vector(np.array([0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    index.add("close_track", close)
    index.add("far_track", far)

    query = unit_vector(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    results = index.search(query, top_k=2)

    assert [track_id for track_id, _ in results] == ["close_track", "far_track"]


def test_identical_vector_scores_near_one():
    index = FusedVectorIndex(dim=DIM)
    vec = unit_vector(np.array([1.0, 2.0, 3.0, 0, 0, 0, 0, 0]))
    index.add("track_a", vec)

    results = index.search(vec, top_k=1)

    track_id, similarity = results[0]
    assert track_id == "track_a"
    assert similarity == pytest.approx(1.0, abs=1e-5)


def test_top_k_larger_than_index_size_returns_all_entries():
    index = FusedVectorIndex(dim=DIM)
    index.add("only_track", unit_vector(np.array([1.0, 0, 0, 0, 0, 0, 0, 0])))

    results = index.search(unit_vector(np.array([1.0, 0, 0, 0, 0, 0, 0, 0])), top_k=5)

    assert len(results) == 1


def test_add_rejects_wrong_shape():
    index = FusedVectorIndex(dim=DIM)
    with pytest.raises(AssertionError):
        index.add("bad_track", np.zeros(DIM + 1))
