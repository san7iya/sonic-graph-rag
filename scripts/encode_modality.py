"""Encode one modality (text/image/audio) for all valid seed tracks and
cache the vectors to disk as an .npz file.

Why a standalone script invoked as a subprocess rather than a function
called in-process: this machine has ~3GB free RAM and a system drive
that's nearly full (constraining how much the Windows pagefile can grow),
which isn't enough to hold MiniLM, CLIP, and CLAP loaded at once —
confirmed by reproducible segfaults when all three are resident in one
process. PyTorch's allocator also doesn't reliably hand freed memory back
to the OS within a process even after dropping references and calling
gc.collect(), so "load, use, unload" in-process isn't enough either.
Running each modality in its own process guarantees the OS reclaims that
memory when the process exits. This also anticipates Milestone 3's
"cache to disk, never recompute live" requirement.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def run_encoder_subprocess(args: list[str], retries: int = 2, delay_seconds: float = 3.0) -> None:
    """Run this script as a subprocess (`python -m scripts.encode_modality ...`),
    retrying on failure.

    On this machine, free RAM is tight enough that loading a ~600MB
    transformer model occasionally hits a transient allocation failure
    (observed as a Windows access violation) even though the exact same
    load reliably succeeds when retried in a fresh process a few seconds
    later. A short retry-with-backoff is cheap insurance against that.
    """
    full_args = [sys.executable, "-m", "scripts.encode_modality", *args]
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            subprocess.run(full_args, cwd=config.ROOT_DIR, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"[retry] encode subprocess failed (attempt {attempt}): {e}")
            time.sleep(delay_seconds)
    raise last_error


def _valid_tracks() -> list[dict]:
    """Tracks that passed scripts/preprocess.py validation in all three
    modalities (shared with run_pipeline so both always see the same corpus).
    """
    from src.preprocessing.records import load_valid_tracks
    return load_valid_tracks()


def encode_text_modality(rows: list[dict]) -> dict[str, np.ndarray]:
    from src.embeddings import text_encoder
    vecs = {}
    for row in rows:
        from src.preprocessing.records import bio_path
        bio = bio_path(row["track_id"]).read_text(encoding="utf-8")
        vecs[row["track_id"]] = text_encoder.encode_text(bio)
    return vecs


def encode_image_modality(rows: list[dict]) -> dict[str, np.ndarray]:
    from src.embeddings import image_encoder
    vecs = {}
    for row in rows:
        image_path = config.RAW_IMAGE_DIR / f"{row['track_id']}.jpg"
        vecs[row["track_id"]] = image_encoder.encode_image(image_path)
    return vecs


def encode_audio_modality(rows: list[dict]) -> dict[str, np.ndarray]:
    from src.embeddings import audio_encoder
    vecs = {}
    for row in rows:
        audio_path = config.RAW_AUDIO_DIR / f"{row['track_id']}.m4a"
        vecs[row["track_id"]] = audio_encoder.encode_audio(audio_path)
    return vecs


ENCODERS = {
    "text": encode_text_modality,
    "image": encode_image_modality,
    "audio": encode_audio_modality,
}


CHECKPOINT_EVERY = 25  # tracks encoded between cache saves


def cache_path(modality: str) -> Path:
    return config.CACHE_DIR / f"{modality}_vectors.npz"


def query_cache_path(modality: str) -> Path:
    return config.CACHE_DIR / f"query_{modality}_vector.npy"


def eval_query_cache_path(modality: str) -> Path:
    return config.CACHE_DIR / f"eval_query_{modality}_vectors.npz"


def _encode_query_text(modality: str, query: str) -> np.ndarray:
    """Encode one query string through the given modality's own *text*
    tower (see retriever.py for why: this is what makes a text query
    score against real image/audio vectors, not just against bios).
    """
    if modality == "text":
        from src.embeddings import text_encoder
        return text_encoder.encode_text(query)
    if modality == "image":
        from src.embeddings import image_encoder
        return image_encoder.encode_text_for_image_space(query)
    from src.embeddings import audio_encoder
    return audio_encoder.encode_text_for_audio_space(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modality", choices=list(ENCODERS.keys()))
    parser.add_argument("--query", default=None,
                         help="Encode this single query string (through the modality's text "
                              "tower) instead of batch-encoding the seed track corpus.")
    parser.add_argument("--queries-file", default=None,
                         help="JSON list of query strings; encode them all in this one process "
                              "(model loaded once) and write eval_query_<modality>_vectors.npz, "
                              "keyed q0, q1, ... Used by scripts/run_evaluation.py.")
    parser.add_argument("--out", default=None,
                         help="With --queries-file: write the .npz here instead of the default "
                              "eval_query_<modality>_vectors.npz, so a second query set does not "
                              "overwrite the first.")
    parser.add_argument("--force", action="store_true",
                         help="Re-encode every track instead of only the ones missing from the cache "
                              "(use after changing a model or the cleaned input).")
    args = parser.parse_args()

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.queries_file is not None:
        import json
        queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
        vecs = {f"q{i}": _encode_query_text(args.modality, q) for i, q in enumerate(queries)}
        out_path = Path(args.out) if args.out else eval_query_cache_path(args.modality)
        np.savez(out_path, **vecs)
        print(f"[{args.modality}] wrote {len(vecs)} query vectors to {out_path}")
    elif args.query is not None:
        vec = _encode_query_text(args.modality, args.query)
        np.save(query_cache_path(args.modality), vec)
        print(f"[{args.modality}] wrote {query_cache_path(args.modality)}")
    else:
        rows = _valid_tracks()
        path = cache_path(args.modality)
        cached = dict(np.load(path)) if (path.exists() and not args.force) else {}
        todo = [r for r in rows if r["track_id"] not in cached]
        print(f"[{args.modality}] {len(rows)} valid tracks, {len(cached)} already cached, "
              f"{len(todo)} to encode...")
        # Checkpoint after every batch so a crash or interruption in a long
        # CPU run (CLAP over hundreds of clips) loses at most one batch.
        for start in range(0, len(todo), CHECKPOINT_EVERY):
            cached.update(ENCODERS[args.modality](todo[start:start + CHECKPOINT_EVERY]))
            np.savez(path, **cached)
            print(f"[{args.modality}] {min(start + CHECKPOINT_EVERY, len(todo))}/{len(todo)} encoded")
        if not todo:
            np.savez(path, **cached)
        print(f"[{args.modality}] wrote {path} ({len(cached)} vectors)")
