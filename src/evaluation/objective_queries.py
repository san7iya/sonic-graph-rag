"""Evaluation queries whose correct answers come straight from metadata.

Hand-labelled relevance does not scale and genre labels are one person's
opinion. But "which songs are by this artist?" and "which songs are on
this album?" have unambiguous answers already stored with every track, so
hundreds of queries can be generated with no labelling effort and no
dependence on the genre labels.

These queries are mostly *name lookups*: they test whether retrieval can
connect a name in the query to the right songs (largely via bio text,
which mentions the artist and album). They do not test mood or sound, so
they complement, not replace, the other evaluations.
"""

from __future__ import annotations

from collections import defaultdict


def artist_queries(tracks: list[dict], min_tracks: int = 3) -> list[dict]:
    """One query per artist with at least `min_tracks` tracks; relevant =
    all of that artist's tracks. Artists with fewer tracks give too few
    positives for precision@k to be meaningful.
    """
    by_artist: dict[str, set[str]] = defaultdict(set)
    for t in tracks:
        by_artist[t["artist"]].add(t["track_id"])
    return [{"query": f"songs by {artist}", "relevant": ids, "kind": "artist"}
            for artist, ids in sorted(by_artist.items()) if len(ids) >= min_tracks]


def album_queries(tracks: list[dict], min_tracks: int = 2) -> list[dict]:
    """One query per (artist, album) with at least `min_tracks` tracks.

    Keyed by artist and album together because album titles repeat across
    artists. iTunes lists reissues under different album names ("Deluxe
    Edition"), so tracks from one real album may be split; those splits make
    the task slightly harder, not easier, so the score is if anything
    pessimistic.
    """
    by_album: dict[tuple[str, str], set[str]] = defaultdict(set)
    for t in tracks:
        by_album[(t["artist"], t["album"])].add(t["track_id"])
    return [{"query": f"songs from the album {album} by {artist}", "relevant": ids, "kind": "album"}
            for (artist, album), ids in sorted(by_album.items()) if len(ids) >= min_tracks]
