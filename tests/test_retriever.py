"""retrieve() has two responsibilities: rank by the fused index, then
attach each hit's graph metadata. Passing query_vec directly (rather than
letting retrieve() call embed_query()) tests that logic without loading
MiniLM/CLIP/CLAP -- embed_query() itself is exercised live by
scripts/run_pipeline.py and scripts/answer_query.py instead, since it
needs the real subprocess-per-model pipeline this machine's RAM budget
requires (see scripts/encode_modality.py).
"""

import numpy as np

from src.indexing.faiss_index import FusedVectorIndex
from src.rag.retriever import retrieve

DIM = 4


def unit_vector(seed: np.ndarray) -> np.ndarray:
    return seed / np.linalg.norm(seed)


def build_index_and_metadata() -> tuple[FusedVectorIndex, dict[str, dict]]:
    index = FusedVectorIndex(dim=DIM)
    index.add("song_a", unit_vector(np.array([1.0, 0, 0, 0])))
    index.add("song_b", unit_vector(np.array([0, 1.0, 0, 0])))

    track_metadata = {
        "song_a": {"title": "Song A", "artist": "Artist A", "genre": "rock", "bio": "bio a"},
        "song_b": {"title": "Song B", "artist": "Artist B", "genre": "jazz", "bio": "bio b"},
    }
    return index, track_metadata


def test_retrieve_ranks_by_similarity_and_respects_top_k():
    index, track_metadata = build_index_and_metadata()
    query_vec = unit_vector(np.array([1.0, 0, 0, 0]))  # matches song_a exactly

    hits = retrieve("irrelevant query text", index, track_metadata, top_k=1, query_vec=query_vec)

    assert len(hits) == 1
    assert hits[0]["track_id"] == "song_a"


def test_retrieve_attaches_graph_metadata_to_each_hit():
    index, track_metadata = build_index_and_metadata()
    query_vec = unit_vector(np.array([1.0, 0, 0, 0]))

    hits = retrieve("irrelevant query text", index, track_metadata, top_k=2, query_vec=query_vec)

    top_hit = hits[0]
    assert top_hit["title"] == "Song A"
    assert top_hit["artist"] == "Artist A"
    assert top_hit["genre"] == "rock"
    assert top_hit["bio"] == "bio a"
    assert isinstance(top_hit["similarity"], float)


def test_retrieve_with_graph_adds_one_hop_expansion():
    from src.graph.build_graph import build_graph

    index, track_metadata = build_index_and_metadata()
    track_metadata["song_c"] = {"title": "Song C", "artist": "Artist A", "genre": "rock", "bio": "bio c"}
    for tid, m in track_metadata.items():
        m["album"] = f"Album {tid}"
    rows = [{"track_id": t, **{k: m[k] for k in ("title", "artist", "genre", "album")}}
            for t, m in track_metadata.items()]
    graph = build_graph(rows)

    hits = retrieve("q", index, track_metadata, top_k=1,
                    query_vec=unit_vector(np.array([1.0, 0, 0, 0])), graph=graph)

    assert hits[0]["graph"]["album"] == "Album song_a"
    assert hits[0]["graph"]["same_artist_tracks"] == ["Song C"]


def test_modality_scores_split_sums_to_fused_similarity():
    from src import config
    from src.embeddings import fusion
    from src.rag.retriever import modality_scores

    rng = np.random.default_rng(0)
    def blocks():
        return (rng.normal(size=config.TEXT_EMBED_DIM), rng.normal(size=config.IMAGE_EMBED_DIM),
                rng.normal(size=config.AUDIO_EMBED_DIM))

    index = FusedVectorIndex(dim=config.FUSED_EMBED_DIM)
    index.add("t", fusion.fuse_embeddings(*blocks()))
    query_vec = fusion.fuse_embeddings(*blocks())

    parts = modality_scores(index, "t", query_vec)
    fused_similarity = index.search(query_vec, 1)[0][1]
    assert set(parts) == {"text", "image", "audio"}
    assert abs(sum(parts.values()) - fused_similarity) < 1e-4
