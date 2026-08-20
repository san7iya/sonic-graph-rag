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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.embeddings import fusion
from scripts.encode_modality import query_cache_path, run_encoder_subprocess

if TYPE_CHECKING:
    # Only needed for the type hint below, not at runtime: importing
    # faiss_index (and therefore faiss) is deferred so callers that need
    # to embed_query() before it's safe to also load faiss (see
    # run_pipeline.py) can import this module without paying that cost.
    from src.indexing.faiss_index import FusedVectorIndex


def embed_query(query_text: str) -> np.ndarray:
    """Build a fused query vector from one piece of text, using each
    modality model's own text tower so all three sub-spaces are non-zero.

    Each modality is encoded in its own subprocess (see
    scripts/encode_modality.py): this machine can't hold MiniLM, CLIP,
    and CLAP loaded in the same process without crashing, so even a
    single query is embedded one model-load at a time.
    """
    sub_vecs = []
    for modality in ("text", "image", "audio"):
        run_encoder_subprocess([modality, "--query", query_text])
        sub_vecs.append(np.load(query_cache_path(modality)))
    return fusion.fuse_embeddings(*sub_vecs)


def retrieve(query_text: str, index: FusedVectorIndex, track_metadata: dict[str, dict],
             top_k: int, query_vec: np.ndarray | None = None) -> list[dict]:
    """Return the top_k retrieved tracks as dicts ready for
    prompt_templates.build_context_block, most similar first.

    query_vec lets a caller pass an already-embedded query instead of
    embedding it here — run_pipeline.py uses this to embed the query
    *before* importing faiss, since this machine can't have faiss and
    google-genai resident in the parent process while still spawning the
    memory-heavy encoder subprocesses reliably.
    """
    if query_vec is None:
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
