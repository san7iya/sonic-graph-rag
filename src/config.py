"""Single source of truth for paths, model names, and fusion constants.

Kept in one place so every module (encoders, fusion, indexing, RAG) imports
the same values instead of redefining them, which would let them drift out
of sync silently.
"""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

RAW_TEXT_DIR = RAW_DIR / "text"
RAW_IMAGE_DIR = RAW_DIR / "images"
RAW_AUDIO_DIR = RAW_DIR / "audio"

SEED_TRACKS_CSV = PROCESSED_DIR / "seed_tracks.csv"
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
LLM_MODEL_NAME = "gemini-2.5-flash"

# --- Embedding dimensions (must match the models above) ---
TEXT_EMBED_DIM = 384
IMAGE_EMBED_DIM = 512
AUDIO_EMBED_DIM = 512
FUSED_EMBED_DIM = TEXT_EMBED_DIM + IMAGE_EMBED_DIM + AUDIO_EMBED_DIM  # 1408

# --- Chunking (for long text like bios) ---
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_RATIO = 0.175  # midpoint of the 15-20% spec

# --- Retrieval ---
DEFAULT_TOP_K = 5
