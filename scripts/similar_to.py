"""CLI for query-by-example: songs that sound like a given song,
optionally steered by a text modifier.

    python -m scripts.similar_to "three little birds" --top-k 5
    python -m scripts.similar_to bob_marley_three_little_birds --modifier "songs about sadness"

The seed is matched on audio (CLAP) by default; the modifier phrase is
matched against bios (MiniLM). No LLM call and no quota needed. Without
--modifier no model is loaded at all (cached vectors only).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.records import load_valid_tracks
from src.rag import similar


def resolve_seed(query: str, tracks: list[dict]) -> dict:
    """Find one track from an exact id or a unique title/artist substring.

    Raises ValueError listing candidates if it is missing or ambiguous,
    rather than silently choosing one.
    """
    for t in tracks:
        if t["track_id"] == query:
            return t
    q = query.lower()
    matches = [t for t in tracks if q in t["title"].lower() or q in t["track_id"]]
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(f"{t['track_id']}" for t in matches[:8]) or "none"
    raise ValueError(f"{'ambiguous' if matches else 'no'} match for {query!r}: {names}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="track_id, or a unique part of the title")
    parser.add_argument("--modifier", default=None, help='text steer, e.g. "happier lyrics"')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-modalities", default="audio",
                        help="comma list from text,image,audio used to compare against the seed")
    args = parser.parse_args()

    tracks = load_valid_tracks()
    info = {t["track_id"]: t for t in tracks}
    seed = resolve_seed(args.seed, tracks)
    ids, mats = similar.load_corpus_vectors(list(info))

    modifier_vecs = None
    if args.modifier:
        from src.rag import retriever  # loads encoders in subprocesses, only when needed
        modifier_vecs = similar.split_fused(retriever.embed_query(args.modifier))

    seed_weights = {m: 1.0 for m in args.seed_modalities.split(",")}
    hits = similar.similar_songs(seed["track_id"], ids, mats, args.top_k,
                                 seed_weights=seed_weights, modifier_vecs=modifier_vecs)
    print(f"Seed: {seed['title']} - {seed['artist']} ({seed['genre']})"
          + (f"  | modifier: {args.modifier!r}" if args.modifier else ""))
    for rank, h in enumerate(hits, start=1):
        t = info[h["track_id"]]
        parts = ", ".join(f"{k}={v:.3f}" for k, v in h.items() if k.startswith(("seed_", "modifier_")))
        print(f"  [{rank}] {t['title']} - {t['artist']} ({t['genre']})  score={h['score']:.3f}  [{parts}]")
