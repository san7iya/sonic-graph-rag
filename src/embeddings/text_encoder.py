"""Wraps MiniLM (sentence-transformers) to turn track bios into vectors.

Pipeline, spelled out because it's what gets asked about in review:
  chunk (long bios split into overlapping windows)
  -> tokenize (subword, via the model's own tokenizer)
  -> encode (transformer self-attention -> contextual per-token vectors)
  -> mean-pool (collapse token vectors into one fixed-size vector)
  -> L2-normalize (so cosine similarity == inner product downstream)

MiniLM is trained with a contrastive objective (similar sentence pairs
pulled together, dissimilar pushed apart), which is *why* cosine distance
between its output vectors tracks semantic similarity — it isn't just an
API we call, the training objective is what makes retrieval work at all.

Embeddings here are contextual: the same word gets a different vector
depending on surrounding context, unlike static Word2Vec/GloVe embeddings.
"""

from src import config  # noqa: E402 -- must set HF_HOME before HF libs import

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the model once per process; loading is slow (reads
    weights from disk) so we don't want to repeat it per call.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(config.TEXT_MODEL_NAME)
    return _model


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE_TOKENS,
               overlap_ratio: float = config.CHUNK_OVERLAP_RATIO) -> list[str]:
    """Split long text into overlapping word-count windows.

    We approximate "tokens" with whitespace-split words rather than
    running the real subword tokenizer here, since the encoder tokenizes
    internally anyway — this chunking step only needs to keep each piece
    short enough for the model's context window and give neighboring
    chunks some shared context (the overlap) so a fact split across a
    chunk boundary isn't silently lost.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def encode_text(text: str) -> np.ndarray:
    """Encode a single piece of text into a 384-dim, L2-normalized vector.

    For a bio longer than one chunk, we mean-pool across chunk embeddings
    first (chunk-of-chunks pooling) so the node still ends up with one
    vector, then re-normalize — normalizing twice is required because
    averaging two unit vectors does not itself produce a unit vector.
    """
    model = _get_model()
    chunks = chunk_text(text) or [text]
    chunk_vectors = model.encode(chunks, normalize_embeddings=True)
    pooled = np.mean(chunk_vectors, axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


if __name__ == "__main__":
    sample = (config.RAW_TEXT_DIR / "queen_bohemian_rhapsody.txt").read_text(encoding="utf-8")
    chunks = chunk_text(sample)
    print(f"chunked bio ({len(sample.split())} words) into {len(chunks)} chunk(s)")

    vec = encode_text(sample)
    print("vector shape:", vec.shape)
    print("first 8 values:", np.round(vec[:8], 4))
    print("L2 norm (should be ~1.0):", round(float(np.linalg.norm(vec)), 6))
