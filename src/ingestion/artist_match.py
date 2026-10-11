"""Decide whether an iTunes result really belongs to the intended artist.

The first catalogue build used a substring test, so searching "Nas" also
accepted Lil Nas X's songs, and "Culture" accepted Culture Club. The fix is
to split the iTunes credit into its individual credited artists (on
"feat.", "&", "and", ",", "x", "with") and require the intended artist to
equal one of them exactly. Collaborations still count ("Stan Getz & Bill
Evans" is a credited Bill Evans track; "Toots & The Maytals" matches the
seed "Toots and the Maytals") but near-miss names do not.
"""

import re

_SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|with|vs\.?|&|and|x|,)\s+|\s*,\s*", flags=re.IGNORECASE)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", " and ")).strip()


def credited_artists(credit: str) -> list[str]:
    """Normalised individual artists in a credit string.

    "Jeremih feat. 50 Cent" -> ["jeremih", "50 cent"]. A standalone "x"
    needs whitespace on both sides, so the trailing X of "Lil Nas X" is not
    treated as a separator.
    """
    return [_normalize(part) for part in _SPLIT.split(credit.strip()) if _normalize(part)]


def is_credited_artist(seed_artist: str, itunes_credit: str) -> bool:
    """True if the seed artist (its first-named artist) is one of the artists
    credited on the iTunes result."""
    seed = credited_artists(seed_artist)
    return bool(seed) and seed[0] in credited_artists(itunes_credit)
