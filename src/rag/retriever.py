"""Query -> embed -> top-k retrieval against the fused FAISS index.

Retrieval is embed + top-k, optionally followed by 1-hop graph expansion
(each retrieved song's album and same-artist / same-genre songs from the
knowledge graph in src/graph).

Query embedding mirrors fusion.py's shape: a 384+512+512 vector built
from each model's own *text* tower (MiniLM directly, CLIP's and CLAP's
text encoders for the image/audio sub-spaces). This is what makes a
query like "sounds like a slow ballad" score against real audio clips
via CLAP's cross-modal alignment, and not just against other bios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src import config
from src.embeddings import fusion
from src.graph.graph_utils import song_neighborhood
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


MAX_GENRE_EXAMPLES = 5


def modality_scores(index: FusedVectorIndex, track_id: str, query_vec: np.ndarray) -> dict[str, float]:
    """Split one hit's fused similarity into its text / image / audio parts.

    Why: the fused score is the sum of three per-modality cosines (each
    block is unit-normalized), so showing the parts reveals *which*
    modality made a song match -- the main thing a reviewer will ask
    about a multi-modal retriever. Fusion and this split share the same
    block boundaries from config, so they cannot drift apart.
    """
    track_vec = index.index.reconstruct(index.track_ids.index(track_id))
    bounds = np.cumsum([0, config.TEXT_EMBED_DIM, config.IMAGE_EMBED_DIM, config.AUDIO_EMBED_DIM])
    return {
        name: float(np.dot(track_vec[lo:hi], query_vec[lo:hi]))
        for name, lo, hi in zip(("text", "image", "audio"), bounds[:-1], bounds[1:])
    }


def retrieve(query_text: str, index: FusedVectorIndex, track_metadata: dict[str, dict],
             top_k: int, query_vec: np.ndarray | None = None,
             graph: nx.DiGraph | None = None,
             modality_weights: tuple[float, float, float] | None = None) -> list[dict]:
    """Return the top_k retrieved tracks as dicts ready for
    prompt_templates.build_context_block, most similar first.

    If a knowledge graph is passed, each hit also gets a "graph" entry
    (1-hop expansion: album plus other songs by the same artist / in the
    same genre, as titles) so the LLM can answer relational questions
    from explicit edges rather than from embedding similarity alone.

    query_vec lets a caller pass an already-embedded query instead of
    embedding it here — run_pipeline.py uses this to embed the query
    *before* importing faiss, since this machine can't have faiss and
    google-genai resident in the parent process while still spawning the
    memory-heavy encoder subprocesses reliably.
    """
    if query_vec is None:
        query_vec = embed_query(query_text)
    if modality_weights is not None:
        # Opt-in per-query weighting (see query_router.py); default is the baseline.
        query_vec = fusion.apply_query_weights(query_vec, modality_weights)
    hits = index.search(query_vec, top_k)

    results = []
    for track_id, similarity in hits:
        meta = track_metadata[track_id]
        result = {
            "track_id": track_id,
            "title": meta["title"],
            "artist": meta["artist"],
            "genre": meta["genre"],
            "bio": meta["bio"],
            "similarity": similarity,
        }
        if graph is not None:
            hood = song_neighborhood(graph, track_id)
            titles = lambda ids: [track_metadata[t]["title"] for t in ids if t in track_metadata]
            result["graph"] = {
                "album": hood["album"],
                "same_artist_tracks": titles(hood["same_artist_tracks"]),
                # A genre can hold hundreds of songs; the prompt only needs a few
                # examples and the total, so the list is capped here.
                "same_genre_tracks": titles(hood["same_genre_tracks"])[:MAX_GENRE_EXAMPLES],
                "same_genre_count": len(hood["same_genre_tracks"]),
                "similar_tracks": titles(hood["similar_tracks"]),
            }
        results.append(result)
    return results
