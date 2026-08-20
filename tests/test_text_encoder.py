"""chunk_text() is pure logic (no model), tested directly. encode_text()
is tested with a real MiniLM inference call -- this is the one encoder
this suite loads for real, since MiniLM alone is well within this
machine's RAM budget; CLIP/CLAP are deliberately not exercised in the
same pytest run (see test_retriever.py's docstring and
scripts/encode_modality.py for why loading multiple heavy models in one
process crashes here).
"""

import numpy as np
import pytest

from src import config
from src.embeddings.text_encoder import chunk_text, encode_text


def test_chunk_text_short_text_returns_single_chunk():
    text = " ".join(f"word{i}" for i in range(50))  # well under CHUNK_SIZE_TOKENS

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_overlaps_correctly():
    word_count = config.CHUNK_SIZE_TOKENS * 3
    words = [f"word{i}" for i in range(word_count)]
    text = " ".join(words)
    step = max(1, int(config.CHUNK_SIZE_TOKENS * (1 - config.CHUNK_OVERLAP_RATIO)))

    chunks = chunk_text(text)

    # Each chunk after the first should start `step` words into the
    # previous one -- i.e. share the same overlap the encoder relies on
    # to avoid losing a fact split across a chunk boundary.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[step:] == second_words[:len(first_words) - step]

    # The last chunk must reach the end of the text -- no trailing words
    # silently dropped.
    assert chunks[-1].split()[-1] == words[-1]


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_encode_text_returns_normalized_384dim_vector():
    vec = encode_text("A short test bio about an upbeat rock song.")

    assert vec.shape == (config.TEXT_EMBED_DIM,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
