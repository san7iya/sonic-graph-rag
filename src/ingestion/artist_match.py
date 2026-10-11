"""Decide whether an iTunes result really belongs to the intended artist.

The first catalogue build used a substring test, so searching "Nas" also
accepted Lil Nas X, and "50 Cent" accepted a Jeremih song that merely
features him. Comparing the *primary* artist of each side (the first name
before "feat.", "&", "and", ",", "x", "with") fixes both while still letting
"Bob Marley & The Wailers" or "Toots & The Maytals" match their seeds.
"""

import re

_SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|with|&|and|x|,)\s+|\s*,\s*", flags=re.IGNORECASE)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", " and ")).strip()


def primary_artist(name: str) -> str:
    """First credited artist of a credit string, normalised for comparison.

    "Jeremih feat. 50 Cent" -> "jeremih"; "Toots & The Maytals" -> "toots";
    "Lil Nas X" -> "lil nas x" (the standalone "x" needs spaces on both sides,
    so a trailing "X" in a name is not treated as a separator).
    """
    return _normalize(_SPLIT.split(name.strip(), maxsplit=1)[0])


def same_primary_artist(seed_artist: str, itunes_artist: str) -> bool:
    return primary_artist(seed_artist) == primary_artist(itunes_artist)
