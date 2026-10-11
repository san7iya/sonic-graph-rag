"""Clean and validate Wikipedia bios before they reach the text encoder."""

import re

MIN_BIO_CHARS = 150


def clean_bio(text: str) -> str:
    """Strip markup left over from Wikipedia plain-text extracts.

    Why: citation markers like "[1]" and "[citation needed]" and stray
    "== Heading ==" lines are not content about the song, but MiniLM would
    embed them as if they were, nudging every vector toward Wikipedia
    boilerplate. Whitespace is also normalised so chunk boundaries (which
    count words) are not skewed by runs of blank lines.
    """
    text = re.sub(r"\[(?:\d+|[a-z]|citation needed|clarification needed)\]", "", text)
    text = re.sub(r"^=+\s*.*?\s*=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


_EDITION_TAG = re.compile(
    r"\s*[\(\[][^\)\]]*(?:remaster|deluxe|edition|version|anniversary|expanded|bonus|mono|stereo)[^\)\]]*[\)\]]",
    flags=re.IGNORECASE)


def strip_edition_tags(name: str) -> str:
    """Drop release-packaging suffixes such as "(Remastered 2011)" or
    "(Super Deluxe Edition)" from a title or album name, keeping meaningful
    parentheses like "(The Four Seasons)".
    """
    return _EDITION_TAG.sub("", name).strip()


def with_metadata_prefix(bio: str, title: str, artist: str, album: str) -> str:
    """Prepend a one-line "Title by Artist, from the album Album." header.

    Why: tracks whose bio is the artist's article would otherwise share
    identical text with that artist's other tracks, so their text vectors
    cannot tell the songs apart. The header is true, track-specific
    metadata, so every track gets distinct text. Trade-off, worth saying out
    loud: it puts the artist and album names into each track's text, so
    name-lookup queries ("songs by X") become easier partly by construction;
    compare results with the header on and off before drawing conclusions.
    """
    header = f"{strip_edition_tags(title)} by {artist}, from the album {strip_edition_tags(album)}."
    return f"{header}\n\n{bio}"


def validate_text(text: str) -> list[str]:
    """Return a list of problems (empty list means the bio is usable)."""
    issues = []
    if len(text) < MIN_BIO_CHARS:
        issues.append(f"bio too short ({len(text)} chars < {MIN_BIO_CHARS})")
    return issues
