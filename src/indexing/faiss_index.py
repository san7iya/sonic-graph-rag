"""Flat FAISS index over fused node vectors, using inner product.

Why IndexFlatIP + normalized vectors: inner product between two unit
vectors equals their cosine similarity. We normalize every fused vector
in fusion.py, so plugging them into an inner-product index gives cosine
similarity search without computing it separately per pair. IndexFlatIP
does an exact (not approximate) search — fine at 15-2000 nodes, where
approximate indexes would only add complexity with no speed benefit.
"""

import json
from pathlib import Path

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


    def vectors(self) -> np.ndarray:
        """All stored vectors, row i belonging to self.track_ids[i]."""
        return self.index.reconstruct_n(0, self.index.ntotal)

    def save(self, path: Path) -> None:
        """Write the FAISS index to `path` and the row -> track_id mapping
        next to it (FAISS itself stores only the vectors, not our ids).
        """
        faiss.write_index(self.index, str(path))
        path.with_suffix(".ids.json").write_text(json.dumps(self.track_ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FusedVectorIndex":
        """Read back an index written by save()."""
        index = faiss.read_index(str(path))
        loaded = cls(dim=index.d)
        loaded.index = index
        loaded.track_ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
        return loaded
