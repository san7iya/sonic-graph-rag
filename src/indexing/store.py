"""Persist the FAISS index and the knowledge graph so a query does not
rebuild them from the encoder caches every time.

The saved copies are only trusted while the inputs they were built from are
unchanged. A fingerprint over the three vector caches and the validated
track list is stored next to them; if any of those files changes (a new
encode, a re-run of preprocessing), the saved index/graph are treated as
stale and rebuilt instead of silently serving old vectors.
"""

from __future__ import annotations

import hashlib
import pickle

from src import config

FORMAT_VERSION = "v1"                 # bump if the index/graph build logic changes
FINGERPRINT_PATH = config.CACHE_DIR / "index_fingerprint.txt"


def corpus_fingerprint() -> str:
    """Hash of (name, size, mtime) of every file the index and graph depend on."""
    parts = [FORMAT_VERSION]
    for path in (config.CACHE_DIR / "text_vectors.npz", config.CACHE_DIR / "image_vectors.npz",
                 config.CACHE_DIR / "audio_vectors.npz", config.TRACK_RECORDS_CSV):
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            parts.append(f"{path.name}:missing")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def load_cached(expected_track_ids: list[str] | None = None):
    """Return (index, graph) if the saved copies exist and are fresh, else None.

    `expected_track_ids`, when given, must match the saved index's track set
    exactly -- a second guard in case the fingerprint inputs did not change
    but the caller is asking about a different corpus.
    """
    from src.indexing.faiss_index import FusedVectorIndex

    needed = (config.FAISS_INDEX_PATH, config.FAISS_INDEX_PATH.with_suffix(".ids.json"),
              config.GRAPH_EXPORT_PATH, FINGERPRINT_PATH)
    if not all(p.exists() for p in needed):
        return None
    if FINGERPRINT_PATH.read_text(encoding="utf-8").strip() != corpus_fingerprint():
        return None
    index = FusedVectorIndex.load(config.FAISS_INDEX_PATH)
    if expected_track_ids is not None and set(index.track_ids) != set(expected_track_ids):
        return None
    with open(config.GRAPH_EXPORT_PATH, "rb") as f:
        graph = pickle.load(f)
    return index, graph


def save_cached(index, graph) -> None:
    """Write the index, graph and the fingerprint they were built from."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index.save(config.FAISS_INDEX_PATH)
    with open(config.GRAPH_EXPORT_PATH, "wb") as f:
        pickle.dump(graph, f)
    FINGERPRINT_PATH.write_text(corpus_fingerprint(), encoding="utf-8")
