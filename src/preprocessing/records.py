"""Read the per-track validation records written by scripts/preprocess.py.

Single place that decides which tracks the encoders and the pipeline may
use, so they can never disagree about the corpus (a mismatch would make
the FAISS index and the graph describe different sets of songs).
"""

import csv

from src import config

STATUS_OK = "ok"


def load_valid_tracks() -> list[dict]:
    """Return metadata rows for every track whose three modalities all
    passed validation.

    Raises FileNotFoundError (with the fix) rather than silently falling
    back to unvalidated data when preprocessing has not been run.
    """
    if not config.TRACK_RECORDS_CSV.exists():
        raise FileNotFoundError(
            f"{config.TRACK_RECORDS_CSV} not found. Run `python -m scripts.preprocess` first."
        )
    with open(config.TRACK_RECORDS_CSV, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["status"] == STATUS_OK]


def bio_path(track_id: str):
    """Path of a track's cleaned bio (the text the encoder should read)."""
    return config.PROCESSED_TEXT_DIR / f"{track_id}.txt"
