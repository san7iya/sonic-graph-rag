"""Download real 30-second audio previews via the iTunes Search API.

Why iTunes instead of the GTZAN dataset for Milestone 0: GTZAN's clips
carry only a genre label, not a real song identity, so they can't be
paired with a specific track's real album art and bio (see CLAUDE.md
section 6). The iTunes Search API is free, needs no key, and returns a
genuine preview clip for the exact song we searched for. GTZAN is used
later (Milestone 1+) for bulk genre-classification scale, where per-song
identity isn't load-bearing.
"""

import csv
import time

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def fetch_audio_preview(artist: str, title: str, album: str | None = None) -> bytes | None:
    """Return the bytes of a 30s audio preview for a track, or None if no
    match with a preview URL was found.

    iTunes often returns several versions of a song (studio, live,
    compilation). When an album is given, we prefer the result whose
    collectionName matches it, to avoid grabbing a live/remix version.
    """
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{artist} {title}", "media": "music", "entity": "song", "limit": 5},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    results = [r for r in results if r.get("previewUrl")]
    if not results:
        return None

    if album:
        for r in results:
            if r.get("collectionName", "").strip().lower() == album.strip().lower():
                chosen = r
                break
        else:
            chosen = results[0]
    else:
        chosen = results[0]

    preview_resp = requests.get(chosen["previewUrl"], timeout=30)
    preview_resp.raise_for_status()
    return preview_resp.content


def _run_for_seed_tracks() -> None:
    """CLI entry point: download previews for every row in seed_tracks.csv
    and write one .m4a file per track to data/raw/audio/.
    """
    config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        track_id, title, artist, album = row["track_id"], row["title"], row["artist"], row["album"]
        out_path = config.RAW_AUDIO_DIR / f"{track_id}.m4a"
        try:
            audio_bytes = fetch_audio_preview(artist, title, album)
        except requests.RequestException as e:
            print(f"[FAIL] {track_id}: request error ({e})")
            continue

        if audio_bytes:
            out_path.write_bytes(audio_bytes)
            print(f"[OK]   {track_id}: {len(audio_bytes)} bytes")
        else:
            print(f"[MISS] {track_id}: no preview found")

        time.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    _run_for_seed_tracks()
