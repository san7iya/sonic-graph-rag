"""Fusion is pure arithmetic (normalize each modality, concatenate) --
these tests use synthetic vectors so they run instantly with no model
loading, and check the two properties the whole retrieval pipeline
depends on: the fused vector has the right shape, and each modality's
block within it is independently unit-length.
"""

import numpy as np
import pytest

from src import config
from src.embeddings.fusion import fuse_embeddings


def test_fused_vector_has_expected_shape():
    text_vec = np.random.rand(config.TEXT_EMBED_DIM)
    image_vec = np.random.rand(config.IMAGE_EMBED_DIM)
    audio_vec = np.random.rand(config.AUDIO_EMBED_DIM)

    fused = fuse_embeddings(text_vec, image_vec, audio_vec)

    assert fused.shape == (config.FUSED_EMBED_DIM,)


def test_each_modality_block_is_independently_unit_norm():
    # Deliberately unnormalized and on different scales, to check fusion
    # normalizes each block itself rather than trusting its inputs.
    text_vec = np.random.rand(config.TEXT_EMBED_DIM) * 10
    image_vec = np.random.rand(config.IMAGE_EMBED_DIM) * 0.01
    audio_vec = np.random.rand(config.AUDIO_EMBED_DIM) * 100

    fused = fuse_embeddings(text_vec, image_vec, audio_vec)

    text_block = fused[:config.TEXT_EMBED_DIM]
    image_block = fused[config.TEXT_EMBED_DIM:config.TEXT_EMBED_DIM + config.IMAGE_EMBED_DIM]
    audio_block = fused[config.TEXT_EMBED_DIM + config.IMAGE_EMBED_DIM:]

    assert np.linalg.norm(text_block) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(image_block) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(audio_block) == pytest.approx(1.0, abs=1e-6)


def test_zero_vector_input_does_not_raise():
    # fuse_embeddings guards divide-by-zero for an all-zero modality
    # (e.g. a missing/failed encoder output) rather than crashing.
    text_vec = np.zeros(config.TEXT_EMBED_DIM)
    image_vec = np.random.rand(config.IMAGE_EMBED_DIM)
    audio_vec = np.random.rand(config.AUDIO_EMBED_DIM)

    fused = fuse_embeddings(text_vec, image_vec, audio_vec)

    assert fused.shape == (config.FUSED_EMBED_DIM,)
    assert np.all(fused[:config.TEXT_EMBED_DIM] == 0)
