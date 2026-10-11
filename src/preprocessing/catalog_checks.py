"""Catalogue-level checks: is this track really the song we meant?

The per-modality validators (audio/image/text) cannot notice a perfectly
good audio clip, cover and bio that belong to the *wrong song*. These
checks use the iTunes side table (data/processed/catalog_extra.csv) to
catch the two failure modes seen in the first catalogue build:
  - a different artist's song with the same or similar title / name
    (e.g. Lil Nas X's "Panini" filed under Nas);
  - for classical, where iTunes credits the performer so the artist test
    cannot apply, a title match that is actually a pop/rap/electronic track
    (iTunes tags it with a clearly non-classical genre).
"""

from src.evaluation.label_check import coarse_genre


def catalog_issues(row: dict, extra: dict | None) -> list[str]:
    """Problems with a track's identity; empty means no evidence against it.

    `extra` is the track's row from catalog_extra.csv, or None when the
    enrichment has not been run (then nothing can be checked).
    """
    if extra is None:
        return []
    issues = []
    if extra.get("artist_ok") == "no":
        issues.append(f"catalog: iTunes credits '{extra.get('itunes_artist', '')}', not '{row['artist']}'")
    if row["genre"] == "classical":
        guess = coarse_genre(extra.get("itunes_genre", ""))
        if guess not in (None, "classical"):
            issues.append(f"catalog: classical track tagged '{extra['itunes_genre']}' by iTunes "
                          "(probably a different recording)")
    return issues
