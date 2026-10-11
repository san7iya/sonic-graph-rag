"""Baseline multi-modal fusion: normalize each modality, then concatenate.

Why concatenate instead of average: MiniLM, CLIP, and CLAP each carve out
their own vector space, trained on different objectives and data — there
is no reason a text axis and an audio axis should mean the same thing, so
averaging them would blend unrelated dimensions together. Concatenation
keeps each modality's dimensions distinct and lets the shared FAISS index
compute one combined cosine similarity per node, weighting all three
modalities equally (since each was L2-normalized before concatenation).

This is the Milestone 0-3 target per CLAUDE.md. A learned/weighted fusion
network is an explicit stretch goal, only after this baseline's retrieval
quality is measured and a concrete weakness is found — not before.
"""

import numpy as np

from src import config


def fuse_embeddings(text_vec: np.ndarray, image_vec: np.ndarray, audio_vec: np.ndarray) -> np.ndarray:
    """Concatenate three independently L2-normalized modality vectors into
    one fused node vector (384 + 512 + 512 = 1408-dim).

    Each input is re-normalized here (not just trusted to already be unit
    length) so fusion behaves correctly even if an encoder's output was
    passed in without normalization by mistake.
    """
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    fused = np.concatenate([
        _l2_normalize(text_vec),
        _l2_normalize(image_vec),
        _l2_normalize(audio_vec),
    ])
    assert fused.shape[0] == config.FUSED_EMBED_DIM, (
        f"expected fused dim {config.FUSED_EMBED_DIM}, got {fused.shape[0]}"
    )
    return fused


def apply_query_weights(query_vec: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Scale a fused *query* vector's three blocks by modality weights.

    Why only the query: the indexed track vectors stay the plain baseline
    concatenation, and the inner product of (w_m * q_m) with t_m summed
    over blocks is exactly sum_m w_m * cos(q_m, t_m). So weighting needs no
    re-indexing and the baseline can be recovered by passing equal weights.
    Weights are multiplied by 3 so equal weights (1/3 each) are the
    identity, keeping scores on the same scale as the baseline.
    """
    scaled = query_vec.copy()
    bounds = np.cumsum([0, config.TEXT_EMBED_DIM, config.IMAGE_EMBED_DIM, config.AUDIO_EMBED_DIM])
    for w, lo, hi in zip(weights, bounds[:-1], bounds[1:]):
        scaled[lo:hi] *= 3 * w
    return scaled
