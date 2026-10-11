"""Milestone 1 preprocessing: validate every track's three modalities and
write data/processed/track_records.csv (one row per track, with an
explicit status and the reasons for any flagged modality).

Done-check from the spec: "data/processed/ has a consistent record per
track with all three modalities present or explicitly flagged missing".
Raw files are never modified; cleaned bios go to data/processed/text/.

Usage: python -m scripts.preprocess [--catalog PATH]
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.preprocessing.clean_audio import validate_audio
from src.preprocessing.clean_images import validate_image
from src.preprocessing.clean_text import clean_bio, validate_text
from src.preprocessing.records import STATUS_OK

RECORD_FIELDS = ["track_id", "title", "artist", "genre", "album", "status",
                 "audio_seconds", "bio_chars", "issues"]


def process_track(row: dict) -> dict:
    """Validate one track's modalities and return its record row."""
    tid = row["track_id"]
    issues = []

    text_path = config.RAW_TEXT_DIR / f"{tid}.txt"
    bio_chars = 0
    if text_path.exists():
        bio = clean_bio(text_path.read_text(encoding="utf-8"))
        bio_chars = len(bio)
        problems = validate_text(bio)
        issues += [f"text: {p}" for p in problems]
        if not problems:
            config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
            (config.PROCESSED_TEXT_DIR / f"{tid}.txt").write_text(bio, encoding="utf-8")
    else:
        issues.append("text: missing")

    image_path = config.RAW_IMAGE_DIR / f"{tid}.jpg"
    issues += [f"image: {p}" for p in validate_image(image_path)] if image_path.exists() else ["image: missing"]

    audio_path = config.RAW_AUDIO_DIR / f"{tid}.m4a"
    duration = 0.0
    if audio_path.exists():
        problems, duration = validate_audio(audio_path)
        issues += [f"audio: {p}" for p in problems]
    else:
        issues.append("audio: missing")

    return {
        "track_id": tid, "title": row["title"], "artist": row["artist"], "genre": row["genre"],
        "album": row["album"], "status": STATUS_OK if not issues else "flagged",
        "audio_seconds": f"{duration:.1f}", "bio_chars": bio_chars, "issues": "; ".join(issues),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=None,
                        help="Track list CSV (default: catalog.csv if present, else seed_tracks.csv)")
    args = parser.parse_args()
    catalog = Path(args.catalog) if args.catalog else (
        config.CATALOG_CSV if config.CATALOG_CSV.exists() else config.SEED_TRACKS_CSV)

    with open(catalog, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Validating {len(rows)} tracks from {catalog.name}...")

    records = [process_track(r) for r in rows]
    with open(config.TRACK_RECORDS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    ok = [r for r in records if r["status"] == STATUS_OK]
    print(f"\n{len(ok)}/{len(records)} tracks valid in all three modalities.")
    print("Valid per genre:", dict(Counter(r["genre"] for r in ok)))
    problem_kinds = Counter(i.split(" (")[0] for r in records for i in r["issues"].split("; ") if i)
    print("Flag reasons:", dict(problem_kinds))
    print(f"Wrote {config.TRACK_RECORDS_CSV}")
