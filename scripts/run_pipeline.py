"""Milestone 0 end-to-end runner: the script demoed at Review 1.

For every seed track: read its real bio/image/audio, run each through
its encoder, fuse into one vector, add to a FAISS index. Then take one
real query, retrieve top-k, and pass the retrieved context to Gemini for
a grounded answer.

Nothing here is a placeholder: every vector, similarity score, and the
final answer come from actually running the pipeline.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.embeddings import audio_encoder, fusion, image_encoder, text_encoder
from src.indexing.faiss_index import FusedVectorIndex
from src.rag import llm_client, prompt_templates, retriever


def load_seed_tracks() -> list[dict]:
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_index() -> tuple[FusedVectorIndex, dict[str, dict]]:
    """Encode every seed track's three modalities, fuse, and index it.
    Prints real vector shapes/values as it goes (Milestone 0 step 2-3).
    """
    rows = load_seed_tracks()
    index = FusedVectorIndex(dim=config.FUSED_EMBED_DIM)
    track_metadata: dict[str, dict] = {}

    for row in rows:
        track_id = row["track_id"]
        bio_path = config.RAW_TEXT_DIR / f"{track_id}.txt"
        image_path = config.RAW_IMAGE_DIR / f"{track_id}.jpg"
        audio_path = config.RAW_AUDIO_DIR / f"{track_id}.m4a"
        if not (bio_path.exists() and image_path.exists() and audio_path.exists()):
            print(f"[SKIP] {track_id}: missing one or more modality files")
            continue

        bio = bio_path.read_text(encoding="utf-8")
        t0 = time.time()
        text_vec = text_encoder.encode_text(bio)
        image_vec = image_encoder.encode_image(image_path)
        audio_vec = audio_encoder.encode_audio(audio_path)
        fused_vec = fusion.fuse_embeddings(text_vec, image_vec, audio_vec)
        elapsed = time.time() - t0

        index.add(track_id, fused_vec)
        track_metadata[track_id] = {
            "title": row["title"], "artist": row["artist"], "genre": row["genre"], "bio": bio,
        }
        print(f"[OK] {track_id}: text{text_vec.shape} image{image_vec.shape} "
              f"audio{audio_vec.shape} -> fused{fused_vec.shape} ({elapsed:.1f}s)")

    return index, track_metadata


def run_query(index: FusedVectorIndex, track_metadata: dict[str, dict], query: str, top_k: int) -> None:
    print(f"\n=== Query: {query!r} ===")
    hits = retriever.retrieve(query, index, track_metadata, top_k)
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit['title']} — {hit['artist']} (similarity={hit['similarity']:.4f})")

    prompt = prompt_templates.build_rag_prompt(query, hits)
    answer = llm_client.generate_answer(prompt)
    print("\n--- Gemini answer (grounded in retrieved context above) ---")
    print(answer)


if __name__ == "__main__":
    print("Building fused index over seed tracks...\n")
    index, track_metadata = build_index()
    print(f"\nIndexed {len(track_metadata)} tracks.")

    run_query(index, track_metadata, "Which songs have an upbeat, danceable feel?", top_k=5)
