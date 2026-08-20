"""Fetch a real bio/description for a track from Wikipedia.

Why Wikipedia: it's a free, key-less public API and gives us unstructured
prose text per track, which is exactly what the text encoder (MiniLM) needs
to demonstrate chunk -> tokenize -> encode -> pool on real, non-synthetic
input rather than a stub sentence.
"""

import csv
import time

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_summary(artist: str, title: str) -> str | None:
    """Return the plain-text intro of the Wikipedia article best matching
    "<title> <artist> song", or None if nothing usable is found.

    Uses a single generator=search call with prop=extracts so the search
    and the content fetch happen in one request instead of two.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{title} {artist} song",
        "gsrlimit": 1,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "format": "json",
    }
    resp = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        headers={"User-Agent": config.USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    return extract or None


def _run_for_seed_tracks() -> None:
    """CLI entry point: fetch bios for every row in seed_tracks.csv and
    write one .txt file per track to data/raw/text/, so the encoder step
    has real files to read instead of needing network access every run.
    """
    config.RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        track_id, title, artist = row["track_id"], row["title"], row["artist"]
        out_path = config.RAW_TEXT_DIR / f"{track_id}.txt"
        try:
            summary = fetch_wikipedia_summary(artist, title)
        except requests.RequestException as e:
            print(f"[FAIL] {track_id}: request error ({e})")
            continue

        if summary:
            out_path.write_text(summary, encoding="utf-8")
            print(f"[OK]   {track_id}: {len(summary)} chars")
        else:
            print(f"[MISS] {track_id}: no Wikipedia summary found")

        time.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    _run_for_seed_tracks()
