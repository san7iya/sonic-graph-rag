"""Flat FAISS index over fused node vectors, using inner product.

Why IndexFlatIP + normalized vectors: inner product between two unit
vectors equals their cosine similarity. We normalize every fused vector
in fusion.py, so plugging them into an inner-product index gives cosine
similarity search without computing it separately per pair. IndexFlatIP
does an exact (not approximate) search — fine at 15-2000 nodes, where
approximate indexes would only add complexity with no speed benefit.
"""

import faiss
import numpy as np


class FusedVectorIndex:
    """Thin wrapper around faiss.IndexFlatIP that also remembers which
    track_id each row corresponds to, since FAISS itself only returns
    integer row positions.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.track_ids: list[str] = []

    def add(self, track_id: str, vector: np.ndarray) -> None:
        assert vector.shape == (self.dim,), f"expected shape ({self.dim},), got {vector.shape}"
        self.index.add(vector.reshape(1, -1).astype(np.float32))
        self.track_ids.append(track_id)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return the top_k (track_id, cosine_similarity) pairs for a
        query vector, most similar first.
        """
        scores, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.track_ids[idx], float(score)))
        return results
