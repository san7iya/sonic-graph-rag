"""Bulk ingestion for the scaled catalogue (Milestone 1): download audio,
album art and a bio for every row of data/processed/catalog.csv, then
write data/processed/ingestion_report.csv saying exactly which modalities
each track has and where its bio came from.

Resumable: a file that already exists in data/raw/ is not fetched again,
so an interrupted run (network drop, rate limit) can just be re-run.

Bio source honesty: the song's own Wikipedia article is preferred, but
many songs have none. Rather than silently accepting whatever article the
search returns first (which can be about a different song), a bio is only
kept if the text actually mentions the song title or the artist, and the
report records whether it came from the "song" article, the "artist"
article (shared by all of that artist's tracks), or is "missing".
"""

import csv
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
from src.ingestion.fetch_album_art import fetch_cover_art
from src.ingestion.fetch_text_metadata import WIKIPEDIA_API_URL

CATALOG_CSV = config.PROCESSED_DIR / "catalog.csv"
MIN_BIO_CHARS = 150


def _wikipedia_intro(search_text: str) -> str | None:
    """Plain-text intro of the top Wikipedia search hit, or None."""
    resp = requests.get(
        WIKIPEDIA_API_URL,
        params={"action": "query", "generator": "search", "gsrsearch": search_text, "gsrlimit": 1,
                "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json"},
        headers={"User-Agent": config.USER_AGENT}, timeout=15,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    if not pages:
        return None
    return next(iter(pages.values())).get("extract", "").strip() or None


def _clean_title(title: str) -> str:
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip().lower()


def fetch_validated_bio(artist: str, title: str) -> tuple[str | None, str]:
    """Return (bio_text, source) with source in {"song", "artist", "missing"}.

    Song article is accepted only if it mentions the song title (the
    artist's name alone isn't enough: the search could have returned the
    artist's page). The fallback artist article must mention the artist.
    """
    song_text = _wikipedia_intro(f"{title} {artist} song")
    if song_text and len(song_text) >= MIN_BIO_CHARS and _clean_title(title) in song_text.lower():
        return song_text, "song"
    artist_core = artist.split("&")[0].strip().lower()
    artist_text = _wikipedia_intro(f"{artist} musician band")
    if artist_text and len(artist_text) >= MIN_BIO_CHARS and artist_core.split()[-1] in artist_text.lower():
        return artist_text, "artist"
    return None, "missing"


def _download(url: str) -> bytes | None:
    resp = requests.get(url, timeout=30)
    return resp.content if resp.status_code == 200 else None


def ingest_row(row: dict, art_cache: dict, artist_bio_cache: dict) -> dict:
    """Fetch whatever is still missing for one track; return its report row."""
    tid = row["track_id"]
    audio_path = config.RAW_AUDIO_DIR / f"{tid}.m4a"
    image_path = config.RAW_IMAGE_DIR / f"{tid}.jpg"
    text_path = config.RAW_TEXT_DIR / f"{tid}.txt"
    source_path = config.RAW_TEXT_DIR / f"{tid}.source"

    if not audio_path.exists() and row.get("preview_url"):
        data = _download(row["preview_url"])
        if data:
            audio_path.write_bytes(data)

    if not image_path.exists():
        key = (row["artist"], row["album"])
        if key not in art_cache:  # tracks on one album share a cover
            art_cache[key] = fetch_cover_art(row["artist"], row["album"], row["title"])
            time.sleep(1.0)  # MusicBrainz asks for ~1 request/second
        if art_cache[key]:
            image_path.write_bytes(art_cache[key])

    if not text_path.exists():
        bio, source = fetch_validated_bio(row["artist"], row["title"])
        if bio:
            text_path.write_text(bio, encoding="utf-8")
        source_path.write_text(source, encoding="utf-8")
        time.sleep(0.3)

    return {
        "track_id": tid, "has_audio": audio_path.exists(), "has_image": image_path.exists(),
        "has_text": text_path.exists(),
        "bio_source": source_path.read_text(encoding="utf-8") if source_path.exists() else "missing",
    }


if __name__ == "__main__":
    for d in (config.RAW_AUDIO_DIR, config.RAW_IMAGE_DIR, config.RAW_TEXT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    report, art_cache = [], {}
    for n, row in enumerate(rows, start=1):
        result = None
        for attempt in range(1, 5):
            try:
                result = ingest_row(row, art_cache, {})
                break
            except requests.RequestException as e:
                # ingest_row skips anything already downloaded, so retrying
                # after a network blip only repeats what is still missing.
                print(f"[retry {attempt}] {row['track_id']}: {type(e).__name__}")
                time.sleep(10 * attempt)
        if result is None:
            print(f"[FAIL] {row['track_id']}: gave up after retries")
            result = {"track_id": row["track_id"], "has_audio": (config.RAW_AUDIO_DIR / f"{row['track_id']}.m4a").exists(),
                      "has_image": (config.RAW_IMAGE_DIR / f"{row['track_id']}.jpg").exists(),
                      "has_text": (config.RAW_TEXT_DIR / f"{row['track_id']}.txt").exists(), "bio_source": "error"}
        report.append(result)
        if n % 25 == 0:
            print(f"[{n}/{len(rows)}] last: {result}")

    with open(config.INGESTION_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    complete = sum(r["has_audio"] and r["has_image"] and r["has_text"] for r in report)
    print(f"{complete}/{len(report)} tracks have all three modalities. "
          f"Report: {config.INGESTION_REPORT_CSV}")
