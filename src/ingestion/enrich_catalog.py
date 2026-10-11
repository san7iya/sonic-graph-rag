"""Attach iTunes' own genre tag and release year to each catalogue track.

Separate from build_catalog.py on purpose: the catalogue defines the track
ids that ingestion and every cache are keyed by, so it must not change
underneath them. This script only reads the catalogue and writes a side
table, data/processed/catalog_extra.csv (track_id, itunes_genre,
release_year, matched), used to cross-check the curated genre labels.

Tracks are matched to iTunes results by normalised title within the
artist's search results; unmatched tracks are recorded as matched=no
rather than guessed.
"""

import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
from src.ingestion.build_catalog import ITUNES_SEARCH_URL, normalize_title

OUT_CSV = config.PROCESSED_DIR / "catalog_extra.csv"
FIELDS = ["track_id", "itunes_genre", "release_year", "matched"]


def search_artist(artist: str, attempts: int = 4) -> list[dict]:
    """iTunes song results for an artist, retried on network errors."""
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(ITUNES_SEARCH_URL, params={"term": artist, "media": "music",
                                                           "entity": "song", "limit": 50}, timeout=15)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as e:
            if attempt == attempts:
                raise
            print(f"[retry {attempt}] {artist}: {type(e).__name__}")
            time.sleep(8 * attempt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with open(config.CATALOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_artist: dict[str, list[dict]] = {}
    for r in rows:
        by_artist.setdefault(r["artist"], []).append(r)

    out = []
    for n, (artist, tracks) in enumerate(sorted(by_artist.items()), start=1):
        try:
            results = search_artist(artist)
        except requests.RequestException as e:
            print(f"[FAIL] {artist}: {type(e).__name__}")
            results = []
        by_title = {}
        for res in results:
            by_title.setdefault(normalize_title(res.get("trackName", "")), res)
        for t in tracks:
            hit = by_title.get(normalize_title(t["title"]))
            out.append({"track_id": t["track_id"],
                        "itunes_genre": hit["primaryGenreName"] if hit else "",
                        "release_year": hit.get("releaseDate", "")[:4] if hit else "",
                        "matched": "yes" if hit else "no"})
        if n % 20 == 0:
            print(f"[{n}/{len(by_artist)}] artists")
        time.sleep(3.2)  # iTunes allows roughly 20 requests/minute

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {len(out)} rows ({sum(r['matched'] == 'yes' for r in out)} matched) to {OUT_CSV}")
