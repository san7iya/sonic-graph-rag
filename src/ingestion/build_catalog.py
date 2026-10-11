"""Build the scaled track catalogue (Milestone 1) from curated artist
lists via the iTunes Search API.

Why artist lists, not a genre search: the genre label is part of the
structured metadata the knowledge graph and evaluation rely on, so it is
curated by hand per artist (data/processed/artist_seeds.json) rather than
trusting iTunes' own noisy genre tags ("Alternative", "Soundtrack", ...).
iTunes supplies the rest in the same response: real title, album, a
30-second preview URL and artwork URL for that exact song, which keeps
all three modalities tied to one real recording.

Output: data/processed/catalog.csv. The 18 seed tracks are kept first,
with their original ids, so earlier results stay reproducible.
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ARTIST_SEEDS_JSON = config.PROCESSED_DIR / "artist_seeds.json"
SONGS_PER_ARTIST = 4
# Alternate recordings of the same song would add near-duplicate nodes.
SKIP_WORDS = ("live", "remix", "karaoke", "instrumental", "demo", "acoustic", "version", "edit", "mix")

FIELDS = ["track_id", "title", "artist", "genre", "album", "preview_url", "artwork_url"]


def slugify(text: str) -> str:
    """Lowercase, ASCII-ish id fragment: 'Guns N' Roses' -> 'guns_n_roses'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_title(title: str) -> str:
    """Strip '(Remastered 2011)'-style suffixes so duplicates compare equal."""
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip().lower()


def is_alternate_version(title: str, album: str) -> bool:
    """True for live/remix/karaoke-style recordings, judged from the
    title text and the album name, not the parenthesised remaster tag
    ('Remastered' is fine: same recording).
    """
    text = f"{title} {album}".lower()
    return any(re.search(rf"\b{w}\b", text) for w in SKIP_WORDS)


def songs_for_artist(artist: str, genre: str) -> list[dict]:
    """Return up to SONGS_PER_ARTIST catalogue rows for one artist.

    Classical is the exception to the artist-name check: iTunes lists
    the *performer* as artistName, so the composer only appears in the
    track/album text.
    """
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": artist, "media": "music", "entity": "song", "limit": 40},
        timeout=15,
    )
    resp.raise_for_status()
    name_key = artist.lower().replace("&", "and")
    rows, seen = [], set()
    for r in resp.json().get("results", []):
        if not r.get("previewUrl") or not r.get("trackName"):
            continue
        haystack = " ".join([r["artistName"], r["trackName"], r.get("collectionName", "")]).lower().replace("&", "and")
        if genre == "classical":
            if name_key.split()[-1] not in haystack:
                continue
        elif name_key not in r["artistName"].lower().replace("&", "and") and \
                r["artistName"].lower().replace("&", "and") not in name_key:
            continue
        album = r.get("collectionName", "")
        if is_alternate_version(r["trackName"], album):
            continue
        key = normalize_title(r["trackName"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "track_id": f"{slugify(artist)}_{slugify(r['trackName'])}"[:80],
            "title": r["trackName"], "artist": artist, "genre": genre, "album": album,
            "preview_url": r["previewUrl"], "artwork_url": r.get("artworkUrl100", ""),
        })
        if len(rows) == SONGS_PER_ARTIST:
            break
    return rows


def songs_for_artist_with_retry(artist: str, genre: str, attempts: int = 4) -> list[dict]:
    """songs_for_artist, retried with growing pauses: a dropped connection
    or rate-limit hiccup should cost a few seconds, not that artist's songs
    (an earlier run silently lost ~90 artists to a network outage).
    """
    for attempt in range(1, attempts + 1):
        try:
            return songs_for_artist(artist, genre)
        except requests.RequestException as e:
            if attempt == attempts:
                raise
            print(f"[retry {attempt}] {artist}: {type(e).__name__}")
            time.sleep(8 * attempt)


def build_catalog() -> list[dict]:
    seeds = json.loads(ARTIST_SEEDS_JSON.read_text(encoding="utf-8"))
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        catalog = [{**row, "preview_url": "", "artwork_url": ""} for row in csv.DictReader(f)]
    taken_ids = {r["track_id"] for r in catalog}
    taken_titles = {(normalize_title(r["title"]), r["artist"].lower()) for r in catalog}

    for genre, artists in seeds.items():
        for artist in artists:
            try:
                found = songs_for_artist_with_retry(artist, genre)
            except requests.RequestException as e:
                print(f"[FAIL] {artist}: {type(e).__name__}")
                continue
            added = 0
            for row in found:
                if row["track_id"] in taken_ids or (normalize_title(row["title"]), artist.lower()) in taken_titles:
                    continue
                taken_ids.add(row["track_id"])
                catalog.append(row)
                added += 1
            print(f"[{genre}] {artist}: +{added}")
            time.sleep(3.2)  # iTunes allows roughly 20 requests/minute
    return catalog


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # artist names like Dvořák crash cp1252 consoles
    rows = build_catalog()
    out = config.PROCESSED_DIR / "catalog.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_genre = {}
    for r in rows:
        by_genre[r["genre"]] = by_genre.get(r["genre"], 0) + 1
    print(f"Wrote {len(rows)} tracks to {out}: {by_genre}")
