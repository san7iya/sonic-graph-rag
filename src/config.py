"""Single source of truth for paths, model names, and fusion constants.

Kept in one place so every module (encoders, fusion, indexing, RAG) imports
the same values instead of redefining them, which would let them drift out
of sync silently.
"""

import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# The default Hugging Face cache lives under the user's home directory,
# which on this machine is on a system drive that's nearly full (see
# CLAUDE.md-adjacent note: C: had ~256MB free). Redirect it into our own
# cache dir on this project's drive before transformers/sentence-
# transformers get imported anywhere, since they read this at import time.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

RAW_TEXT_DIR = RAW_DIR / "text"
RAW_IMAGE_DIR = RAW_DIR / "images"
RAW_AUDIO_DIR = RAW_DIR / "audio"

PROCESSED_TEXT_DIR = PROCESSED_DIR / "text"   # cleaned bios (raw ones stay untouched in RAW_TEXT_DIR)

SEED_TRACKS_CSV = PROCESSED_DIR / "seed_tracks.csv"
CATALOG_CSV = PROCESSED_DIR / "catalog.csv"             # scaled catalogue (src/ingestion/build_catalog.py)
TRACK_RECORDS_CSV = PROCESSED_DIR / "track_records.csv"  # per-track validation result (scripts/preprocess.py)
INGESTION_REPORT_CSV = PROCESSED_DIR / "ingestion_report.csv"

FAISS_INDEX_PATH = CACHE_DIR / "faiss_index.bin"
GRAPH_EXPORT_PATH = CACHE_DIR / "graph.gpickle"

# Contact info required by MusicBrainz's API usage policy for the User-Agent
# header; requests without one get rate-limited harder or rejected.
CONTACT_EMAIL = "saniya.goyal2023@vitstudent.ac.in"
USER_AGENT = f"music-kg-rag/0.1 ( {CONTACT_EMAIL} )"

# --- Model names (pretrained inference only, no fine-tuning) ---
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"          # sentence-transformers, 384-dim
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"   # CLIP, 512-dim
AUDIO_MODEL_NAME = "laion/clap-htsat-unfused"       # CLAP, 512-dim
LLM_MODEL_NAME = "gemini-3.6-flash"  # gemini-2.5/2.0-flash from the original spec are
# now blocked for this API key ("no longer available to new users"); this is
# the exact replacement Google's own API error names. Flagged and confirmed
# with the user 2026-08-20 rather than swapped silently.

# Groundedness judge (scripts/run_groundedness.py). A separate, smaller model
# so judging has its own free-tier quota (20 requests/day per model) and the
# judge is not literally the same model that wrote the answer. Chosen with
# the user 2026-10-11 after gemini-3.6-flash's daily quota ran out.
JUDGE_MODEL_NAME = "gemini-3.5-flash-lite"

# --- Embedding dimensions (must match the models above) ---
TEXT_EMBED_DIM = 384
IMAGE_EMBED_DIM = 512
AUDIO_EMBED_DIM = 512
FUSED_EMBED_DIM = TEXT_EMBED_DIM + IMAGE_EMBED_DIM + AUDIO_EMBED_DIM  # 1408

# --- Chunking (for long text like bios) ---
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_RATIO = 0.175  # midpoint of the 15-20% spec

# Prepend "Title by Artist, from the album Album." to every cleaned bio
# (scripts/preprocess.py). Gives tracks that share an artist-level bio
# distinct text, at the cost of making name-lookup queries easier by
# construction. Switched on at the user's request 2026-10-11; set False and
# rerun preprocess + `encode_modality text` to compare.
# Environment override (PREPEND_METADATA_TO_BIO=0) lets the on/off comparison run without editing this file.
PREPEND_METADATA_TO_BIO = os.environ.get("PREPEND_METADATA_TO_BIO", "1") != "0"

# --- Retrieval ---
DEFAULT_TOP_K = 5


# --- Offline model loading ---
# transformers/sentence-transformers send a network HEAD request per model
# file on every load, even when the weights are already cached. On a flaky
# connection those requests retry with backoff and made each query take
# 35-60 s instead of ~25 s (measured). Once all three models are in the
# local cache there is nothing to download, so tell the hub library to use
# the cache only. Before the first download the cache is incomplete and
# this stays off, so initial setup still works.
def _all_models_cached() -> bool:
    hub = CACHE_DIR / "huggingface" / "hub"
    names = ["sentence-transformers/" + TEXT_MODEL_NAME, IMAGE_MODEL_NAME, AUDIO_MODEL_NAME]
    return all(list((hub / ("models--" + n.replace("/", "--")) / "snapshots").glob("*/*")) for n in names)


if _all_models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
