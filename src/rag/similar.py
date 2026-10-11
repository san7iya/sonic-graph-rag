"""Query-by-example retrieval: "songs that sound like X, but ...".

The project's motivating question cannot be answered by a text query
alone: "sounds like X" is about a *song*, "happier lyrics" is about
*text*. Because every modality keeps its own unit-normalized vector block,
the two parts can be scored in the right spaces and added:

    score(track) = sum over seed modalities  w_m * cos(seed_m, track_m)
                 + sum over modifier spaces  v_m * cos(modifier_m, track_m)

Default: audio similarity to the seed song (CLAP audio-audio) plus the
modifier phrase matched against bios (MiniLM text-text). Everything runs
on cached vectors with numpy; no model is loaded here.
"""

from __future__ import annotations

import numpy as np

from src import config

MODALITIES = ("text", "image", "audio")


def split_fused(vec: np.ndarray) -> dict[str, np.ndarray]:
    """Cut a fused 1408-dim vector back into its text / image / audio
    blocks, using the same boundaries fusion.py concatenated them with.
    """
    bounds = np.cumsum([0, config.TEXT_EMBED_DIM, config.IMAGE_EMBED_DIM, config.AUDIO_EMBED_DIM])
    return {m: vec[lo:hi] for m, lo, hi in zip(MODALITIES, bounds[:-1], bounds[1:])}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def load_corpus_vectors(track_ids: list[str] | None = None) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load cached per-modality vectors as row-normalized matrices whose
    rows are aligned to one shared list of track ids.

    Only tracks present in all three caches are kept, so a half-encoded
    corpus never yields matrices of different lengths.
    """
    from scripts.encode_modality import cache_path

    raw = {m: dict(np.load(cache_path(m))) for m in MODALITIES}
    ids = sorted(set.intersection(*(set(raw[m]) for m in MODALITIES)))
    if track_ids is not None:
        wanted = set(track_ids)
        ids = [t for t in ids if t in wanted]
    mats = {m: np.stack([_unit(raw[m][t]) for t in ids]) for m in MODALITIES}
    return ids, mats


def similar_songs(seed_id: str, ids: list[str], mats: dict[str, np.ndarray], top_k: int = 5,
                  seed_weights: dict[str, float] | None = None,
                  modifier_vecs: dict[str, np.ndarray] | None = None,
                  modifier_weights: dict[str, float] | None = None) -> list[dict]:
    """Rank all tracks except the seed by the combined score above.

    seed_weights    modality -> weight on cos(seed, track); default audio only.
    modifier_vecs   modality -> query vector for the modifier phrase, in that
                    modality's space (from retriever.embed_query + split_fused).
    modifier_weights modality -> weight; default text only, so "happier
                    lyrics" matches bios. Modalities with no vector are skipped.

    Each result carries its per-component scores so the ranking can be
    explained, not just reported.
    """
    seed_weights = seed_weights or {"audio": 1.0}
    modifier_weights = modifier_weights or {"text": 1.0}
    seed_idx = ids.index(seed_id)

    total = np.zeros(len(ids))
    components: dict[str, np.ndarray] = {}
    for m, w in seed_weights.items():
        part = w * (mats[m] @ mats[m][seed_idx])
        components[f"seed_{m}"] = part
        total += part
    for m, vec in (modifier_vecs or {}).items():
        w = modifier_weights.get(m, 0.0)
        if w == 0.0:
            continue
        part = w * (mats[m] @ _unit(vec))
        components[f"modifier_{m}"] = part
        total += part

    total[seed_idx] = -np.inf  # a song is trivially most similar to itself
    results = []
    for j in np.argsort(-total)[:top_k]:
        results.append({"track_id": ids[j], "score": float(total[j]),
                        **{name: float(c[j]) for name, c in components.items()}})
    return results
