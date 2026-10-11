"""Saved index/graph must round-trip exactly, be discarded when their inputs
change, and the FAISS-based SIMILAR_TO edges must equal a brute-force
reference (so the memory-saving rewrite changed no results).
"""

import numpy as np
import pytest

from src import config
from src.graph.build_graph import add_similarity_edges, build_graph
from src.indexing import store
from src.indexing.faiss_index import FusedVectorIndex


def unit_rows(rng, n, d):
    m = rng.normal(size=(n, d))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def small_index(rng, n=6, d=8):
    ids = [f"t{i}" for i in range(n)]
    vecs = unit_rows(rng, n, d).astype(np.float32)
    index = FusedVectorIndex(dim=d)
    for tid, v in zip(ids, vecs):
        index.add(tid, v)
    return index, ids, vecs


def test_index_save_load_round_trip(tmp_path):
    index, ids, vecs = small_index(np.random.default_rng(0))
    index.save(tmp_path / "idx.bin")
    loaded = FusedVectorIndex.load(tmp_path / "idx.bin")
    assert loaded.track_ids == ids
    assert loaded.search(vecs[2], 3) == index.search(vecs[2], 3)
    assert np.allclose(loaded.vectors(), vecs)


def test_similarity_edges_match_brute_force():
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(40)]
    vecs = unit_rows(rng, 40, 16).astype(np.float32)
    rows = [{"track_id": t, "title": t, "artist": f"a{i % 5}", "genre": "g", "album": f"al{i % 7}"}
            for i, t in enumerate(ids)]
    graph = build_graph(rows)
    assert add_similarity_edges(graph, ids, vecs, k=3) == 40 * 3

    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    for i, tid in enumerate(ids):
        expected = {ids[j] for j in np.argsort(-sims[i])[:3]}
        got = {graph.nodes[v]["track_id"] for _, v, d in graph.out_edges(f"Song:{tid}", data=True)
               if d["relation"] == "SIMILAR_TO"}
        assert got == expected


@pytest.fixture
def store_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", tmp_path / "faiss_index.bin")
    monkeypatch.setattr(config, "GRAPH_EXPORT_PATH", tmp_path / "graph.gpickle")
    monkeypatch.setattr(config, "TRACK_RECORDS_CSV", tmp_path / "track_records.csv")
    monkeypatch.setattr(store, "FINGERPRINT_PATH", tmp_path / "index_fingerprint.txt")
    for name in ("text_vectors.npz", "image_vectors.npz", "audio_vectors.npz", "track_records.csv"):
        (tmp_path / name).write_bytes(b"v1")
    return tmp_path


def test_load_cached_requires_a_saved_copy(store_paths):
    assert store.load_cached() is None


def test_saved_copy_is_reused_until_an_input_changes(store_paths):
    index, ids, _ = small_index(np.random.default_rng(2))
    graph = build_graph([{"track_id": t, "title": t, "artist": "a", "genre": "g", "album": "al"} for t in ids])
    store.save_cached(index, graph)

    cached = store.load_cached(ids)
    assert cached is not None
    loaded_index, loaded_graph = cached
    assert loaded_index.track_ids == ids and loaded_graph.number_of_nodes() == graph.number_of_nodes()

    (store_paths / "audio_vectors.npz").write_bytes(b"a re-encoded cache with different content")
    assert store.load_cached(ids) is None                       # input changed -> stale


def test_saved_copy_is_rejected_for_a_different_track_set(store_paths):
    index, ids, _ = small_index(np.random.default_rng(3))
    store.save_cached(index, build_graph([]))
    assert store.load_cached(ids) is not None
    assert store.load_cached(ids[:-1]) is None
