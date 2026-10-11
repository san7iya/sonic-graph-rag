"""Cross-check the hand-curated genre labels against iTunes' own genre tags.

Not a ground truth: iTunes tags are noisy and per-release ("Pop" for a
Fleetwood Mac compilation, "Rock" for the studio album). The point is to
surface *disagreements* for a human to review, and to give a rough
agreement rate, so the curated labels are not taken on faith.
"""

from __future__ import annotations

# (substring of the lower-cased iTunes genre) -> curated genre. First match wins,
# so more specific families come before the broad "rock"/"pop" ones.
_RULES = [
    (("hip-hop", "hip hop", "rap"), "hiphop"),
    (("reggae", "lovers rock", "dancehall", "ska", "dub", "roots"), "reggae"),
    (("jazz", "bebop", "big band", "swing"), "jazz"),
    (("classical", "opera", "baroque", "orchestral", "chamber", "symphon"), "classical"),
    (("rock", "metal", "punk", "grunge", "alternative", "indie"), "rock"),
    (("pop", "r&b", "soul", "dance", "electronic"), "pop"),
]


def coarse_genre(itunes_genre: str) -> str | None:
    """Map an iTunes genre string onto one of the six curated genres, or
    None when it says nothing useful (Soundtrack, Country, Folk, ...).
    """
    g = (itunes_genre or "").lower()
    for needles, genre in _RULES:
        if any(n in g for n in needles):
            return genre
    return None


def compare_labels(rows: list[dict]) -> dict:
    """rows have `track_id`, `genre` (curated) and `itunes_genre`.

    Returns coverage (tracks where iTunes' tag maps to a curated genre),
    the agreement rate among those, per-genre agreement, and the list of
    disagreements to review.
    """
    mapped, disagreements = [], []
    for r in rows:
        guess = coarse_genre(r.get("itunes_genre", ""))
        if guess is None:
            continue
        mapped.append((r["genre"], guess))
        if guess != r["genre"]:
            disagreements.append({"track_id": r["track_id"], "curated": r["genre"],
                                  "itunes": r["itunes_genre"], "itunes_maps_to": guess})
    per_genre = {}
    for curated, guess in mapped:
        total, agree = per_genre.get(curated, (0, 0))
        per_genre[curated] = (total + 1, agree + (curated == guess))
    return {"rows": len(rows), "mapped": len(mapped),
            "agreement": (sum(a for _, a in per_genre.values()) / len(mapped)) if mapped else float("nan"),
            "per_genre": per_genre, "disagreements": disagreements}
