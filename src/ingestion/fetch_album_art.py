"""Fetch real album cover art via MusicBrainz + the Cover Art Archive.

Why two APIs: Cover Art Archive stores images keyed by MusicBrainz release
or release-group ID, not by artist/album name, so we first resolve the
name to an ID via MusicBrainz's search API, then fetch the image.

We search at the release-group level (artist + album) rather than the
recording level: a release-group aggregates every pressing of an album,
so it has a much higher chance of having art attached, and it matches the
actual studio album rather than an arbitrary compilation the track also
appears on.
"""

import csv
import time

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

MUSICBRAINZ_API_URL = "https://musicbrainz.org/ws/2/release-group/"
COVER_ART_ARCHIVE_URL = "https://coverartarchive.org/release-group/{mbid}/front"
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def _find_release_group_id(artist: str, album: str) -> str | None:
    resp = requests.get(
        MUSICBRAINZ_API_URL,
        params={
            "query": f'releasegroup:"{album}" AND artist:"{artist}"',
            "fmt": "json",
            "limit": 1,
        },
        headers={"User-Agent": config.USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    groups = resp.json().get("release-groups", [])
    return groups[0]["id"] if groups else None


def _fetch_cover_art_via_itunes(artist: str, title: str) -> bytes | None:
    """Fallback for the rare album that has no MusicBrainz release-group
    or no Cover Art Archive image (e.g. some soundtrack compilations).
    iTunes returns artwork per track rather than per album, but since we
    search by the exact artist+title it's still real art for this song,
    just a different real source than the CLAUDE.md-listed primary.
    """
    resp = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{artist} {title}", "media": "music", "entity": "song", "limit": 1},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results or not results[0].get("artworkUrl100"):
        return None

    # artworkUrl100 points at a 100x100 thumbnail; iTunes serves larger
    # sizes from the same path if you swap the size in the filename.
    large_url = results[0]["artworkUrl100"].replace("100x100bb", "600x600bb")
    art_resp = requests.get(large_url, timeout=15)
    if art_resp.status_code != 200:
        return None
    return art_resp.content


def fetch_cover_art(artist: str, album: str, title: str | None = None) -> bytes | None:
    """Return the front-cover image bytes for an artist's album.

    Tries MusicBrainz + Cover Art Archive first (the CLAUDE.md-listed
    source). Falls back to iTunes artwork, keyed by track title, only if
    that fails and a title was given.
    """
    try:
        mbid = _find_release_group_id(artist, album)
        if mbid is not None:
            resp = requests.get(
                COVER_ART_ARCHIVE_URL.format(mbid=mbid),
                headers={"User-Agent": config.USER_AGENT},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.content
    except requests.RequestException:
        pass  # MusicBrainz is occasionally flaky (503s under load); fall through to iTunes

    if title:
        return _fetch_cover_art_via_itunes(artist, title)
    return None


def _run_for_seed_tracks() -> None:
    """CLI entry point: fetch cover art for every row in seed_tracks.csv
    and write one .jpg file per track to data/raw/images/.
    """
    config.RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SEED_TRACKS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        track_id, artist, album, title = row["track_id"], row["artist"], row["album"], row["title"]
        out_path = config.RAW_IMAGE_DIR / f"{track_id}.jpg"
        try:
            image_bytes = fetch_cover_art(artist, album, title)
        except requests.RequestException as e:
            print(f"[FAIL] {track_id}: request error ({e})")
            continue

        if image_bytes:
            out_path.write_bytes(image_bytes)
            print(f"[OK]   {track_id}: {len(image_bytes)} bytes")
        else:
            print(f"[MISS] {track_id}: no cover art found")

        # MusicBrainz's usage policy asks for ~1 request/second unauthenticated.
        time.sleep(1.0)


if __name__ == "__main__":
    _run_for_seed_tracks()
