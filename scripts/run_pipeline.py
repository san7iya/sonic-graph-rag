"""Milestone 0 end-to-end runner: the script demoed at Review 1.

For every seed track: read its real bio/image/audio, run each through
its encoder, fuse into one vector, add to a FAISS index. Then take one
real query, retrieve top-k, and pass the retrieved context to Gemini for
a grounded answer.

Nothing here is a placeholder: every vector, similarity score, and the
final answer come from actually running the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import cache_path, run_encoder_subprocess

# faiss and google-genai are imported lazily, only after every encoder
# subprocess (corpus modalities *and* the query) has already been
# spawned. This machine has tight free RAM; once faiss/google-genai are
# loaded in this parent process, spawning a fresh ~600MB CLIP/CLAP
# subprocess reliably crashes (confirmed by reproducible access
# violations). Keeping the parent minimal until all subprocess spawning
# is done avoids that.


def valid_tracks_with_bios() -> tuple[list[dict], dict[str, str]]:
    """Tracks that passed preprocessing, with their cleaned bios (cheap --
    no ML imports needed). Run `python -m scripts.preprocess` first.
    """
    from src.preprocessing.records import bio_path, load_valid_tracks

    valid_rows = load_valid_tracks()
    bios = {row["track_id"]: bio_path(row["track_id"]).read_text(encoding="utf-8") for row in valid_rows}
    return valid_rows, bios


def assemble_index(valid_rows: list[dict], bios: dict[str, str]) -> tuple[FusedVectorIndex, dict[str, dict]]:
    """Build the FAISS index from each modality's cached vectors (written
    by encode_corpus_modalities). Prints real vector shapes/values as it
    goes (Milestone 0 step 2-3). Imports faiss/fusion here, not at module
    load time — see the note at the top of this file.
    """
    from src.embeddings import fusion
    from src.indexing.faiss_index import FusedVectorIndex

    text_vecs = dict(np.load(cache_path("text")))
    image_vecs = dict(np.load(cache_path("image")))
    audio_vecs = dict(np.load(cache_path("audio")))

    index = FusedVectorIndex(dim=config.FUSED_EMBED_DIM)
    for row in valid_rows:
        track_id = row["track_id"]
        text_vec, image_vec, audio_vec = text_vecs[track_id], image_vecs[track_id], audio_vecs[track_id]
        fused_vec = fusion.fuse_embeddings(text_vec, image_vec, audio_vec)

        index.add(track_id, fused_vec)
        print(f"[OK] {track_id}: text{text_vec.shape} image{image_vec.shape} "
              f"audio{audio_vec.shape} -> fused{fused_vec.shape}")

    return index, build_track_metadata(valid_rows, bios)


def build_track_metadata(valid_rows: list[dict], bios: dict[str, str]) -> dict[str, dict]:
    """track_id -> the fields retrieval attaches to each hit (cheap; always rebuilt)."""
    return {
        row["track_id"]: {"title": row["title"], "artist": row["artist"], "genre": row["genre"],
                          "album": row["album"], "bio": bios[row["track_id"]]}
        for row in valid_rows
    }


def load_index_and_graph(valid_rows: list[dict], bios: dict[str, str]):
    """Return (index, graph, track_metadata), reusing the saved index and
    graph when they are still fresh and rebuilding (then saving) otherwise.

    Why: building the index and the SIMILAR_TO graph from the vector caches
    on every query is wasted work that grows with the corpus; the saved
    copies are validated by src/indexing/store.py before being trusted.
    """
    from src.graph.build_graph import build_graph_with_similarity
    from src.indexing import store

    track_metadata = build_track_metadata(valid_rows, bios)
    cached = store.load_cached([r["track_id"] for r in valid_rows])
    if cached is not None:
        index, graph = cached
        print(f"[cache] loaded saved index and graph ({len(index.track_ids)} tracks)")
        return index, graph, track_metadata

    index, _ = assemble_index(valid_rows, bios)
    graph = build_graph_with_similarity(valid_rows, index)
    store.save_cached(index, graph)
    print(f"[cache] built and saved index and graph ({len(index.track_ids)} tracks)")
    return index, graph, track_metadata


def run_query(index: FusedVectorIndex, graph, track_metadata: dict[str, dict], query: str, top_k: int,
              query_vec: np.ndarray) -> None:
    from src.rag import llm_client, prompt_templates, retriever

    print(f"\n=== Query: {query!r} ===")
    hits = retriever.retrieve(query, index, track_metadata, top_k, query_vec=query_vec, graph=graph)
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit['title']} — {hit['artist']} (similarity={hit['similarity']:.4f})")
        print(f"      graph: album={hit['graph']['album']!r}, "
              f"same artist={hit['graph']['same_artist_tracks']}")

    prompt = prompt_templates.build_rag_prompt(query, hits)
    answer = llm_client.generate_answer(prompt)
    print("\n--- Gemini answer (grounded in retrieved context above) ---")
    print(answer)


if __name__ == "__main__":
    query_text = "Which songs have an upbeat, danceable feel?"

    print("Building fused index over seed tracks...\n")
    valid_rows, bios = valid_tracks_with_bios()

    for modality in ("text", "image", "audio"):
        print(f"-- encoding corpus {modality} --")
        run_encoder_subprocess([modality])

    # Embed the query now too, still before faiss/google-genai are
    # imported anywhere in this process (see module docstring note).
    from src.rag import retriever
    query_vec = retriever.embed_query(query_text)

    index, graph, track_metadata = load_index_and_graph(valid_rows, bios)
    print(f"\nIndexed {len(track_metadata)} tracks.")

    run_query(index, graph, track_metadata, query_text, top_k=5, query_vec=query_vec)
