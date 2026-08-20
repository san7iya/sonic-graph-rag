"""Query -> embed -> top-k retrieval against the fused FAISS index.

For Milestone 0 this is embed + top-k only. Graph expansion (pulling in
each retrieved song's artist/album/genre neighborhood from the knowledge
graph) is Milestone 4 — the graph doesn't exist yet at this stage, so
there is nothing to expand into.

Query embedding mirrors fusion.py's shape: a 384+512+512 vector built
from each model's own *text* tower (MiniLM directly, CLIP's and CLAP's
text encoders for the image/audio sub-spaces). This is what makes a
query like "sounds like a slow ballad" score against real audio clips
via CLAP's cross-modal alignment, and not just against other bios.
"""

import numpy as np

from src.embeddings import audio_encoder, fusion, image_encoder, text_encoder
from src.indexing.faiss_index import FusedVectorIndex


def embed_query(query_text: str) -> np.ndarray:
    """Build a fused query vector from one piece of text, using each
    modality model's own text tower so all three sub-spaces are non-zero.
    """
    text_vec = text_encoder.encode_text(query_text)
    image_vec = image_encoder.encode_text_for_image_space(query_text)
    audio_vec = audio_encoder.encode_text_for_audio_space(query_text)
    return fusion.fuse_embeddings(text_vec, image_vec, audio_vec)


def retrieve(query_text: str, index: FusedVectorIndex, track_metadata: dict[str, dict],
             top_k: int) -> list[dict]:
    """Return the top_k retrieved tracks as dicts ready for
    prompt_templates.build_context_block, most similar first.
    """
    query_vec = embed_query(query_text)
    hits = index.search(query_vec, top_k)

    results = []
    for track_id, similarity in hits:
        meta = track_metadata[track_id]
        results.append({
            "track_id": track_id,
            "title": meta["title"],
            "artist": meta["artist"],
            "genre": meta["genre"],
            "bio": meta["bio"],
            "similarity": similarity,
        })
    return results
