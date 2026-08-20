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
