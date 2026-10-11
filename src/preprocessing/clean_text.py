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


def validate_text(text: str) -> list[str]:
    """Return a list of problems (empty list means the bio is usable)."""
    issues = []
    if len(text) < MIN_BIO_CHARS:
        issues.append(f"bio too short ({len(text)} chars < {MIN_BIO_CHARS})")
    return issues
