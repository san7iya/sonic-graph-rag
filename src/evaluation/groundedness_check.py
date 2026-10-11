"""Groundedness checks for generated answers: does every claim trace to
the retrieved context?

Two complementary checks, because each alone has a blind spot:
  1. Deterministic: any catalogue song title the answer mentions that was
     NOT in the retrieved context is an unambiguous leak of outside
     knowledge or a hallucination. Cheap and reproducible, but it only
     catches song-name leaks, not wrong facts about retrieved songs.
  2. LLM judge: Gemini is shown the context and the answer and must list
     each factual claim with a supported / unsupported verdict. Covers
     facts, but it is the same model family that wrote the answer, so it
     can share its blind spots -- treat it as an estimate, not ground truth.
"""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are a strict fact-checker. Below is CONTEXT (the only allowed evidence) and an ANSWER.
Split the ANSWER into its individual factual claims. For each claim decide whether the CONTEXT directly supports it.
Statements like "the context does not contain enough information" are not claims; skip them.

Reply with ONLY a JSON list, no prose, no code fences, in this form:
[{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false}}]
Use an empty list [] if the answer contains no factual claims.

--- CONTEXT ---
{context}
--- ANSWER ---
{answer}"""


def unretrieved_titles_mentioned(answer: str, retrieved_titles: list[str], catalog_titles: list[str]) -> list[str]:
    """Catalogue titles that appear in the answer but were not retrieved.

    Case-insensitive whole-phrase match. A title that is a substring of a
    retrieved title (e.g. 'Spring' vs 'Spring (The Four Seasons)') is not
    flagged, since mentioning it is just a shortened reference.
    """
    answer_lower = answer.lower()
    retrieved_lower = [t.lower() for t in retrieved_titles]
    leaked = []
    for title in catalog_titles:
        t = title.lower()
        if t in retrieved_lower or any(t in r for r in retrieved_lower):
            continue
        if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", answer_lower):
            leaked.append(title)
    return leaked


def parse_judge_output(raw: str) -> list[dict]:
    """Parse the judge's JSON list, tolerating stray code fences.

    Raises ValueError if no valid list is found, so a malformed judge
    reply is counted as a failed judgement instead of silently as '0 claims'.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON list in judge output: {raw[:200]!r}")
    claims = json.loads(text[start:end + 1])
    return [{"claim": str(c["claim"]), "supported": bool(c["supported"])} for c in claims]


def judge_claims(answer: str, context: str, generate) -> list[dict]:
    """Ask the LLM (`generate` = prompt -> text) to verify each claim."""
    return parse_judge_output(generate(JUDGE_PROMPT.format(context=context, answer=answer)))


# Capitalised words that start sentences or are generic labels, not names
# worth verifying against the context.
_NOT_NAMES = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "based", "according", "both",
              "however", "also", "while", "for", "in", "on", "of", "and", "but", "as", "from", "context",
              "answer", "yes", "no", "retrieved", "song", "songs", "track", "tracks", "genre", "album", "bio",
              "graph", "similar", "other", "one", "two", "three", "if", "there", "here", "to", "with", "by"}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def unsupported_specifics(answer: str, context: str) -> list[str]:
    """Numbers and proper names in the answer that never appear in the
    retrieved context.

    Objective, deterministic complement to the LLM claim judge: if the
    answer states a year, a count, or names a person/place/band that the
    context does not contain, that detail came from somewhere else. It
    cannot see wrong *relationships* between supported facts, and it can
    miss a fabricated detail that happens to share a word with the context,
    so a clean result does not prove groundedness; a hit is a real red flag
    worth reading.

    Names are runs of capitalised words (joined by 'of', '&', 'the', 'and'
    only inside a run); a run counts as supported if the whole phrase, or
    every word of it, occurs in the context. Sentence-starting filler words
    are ignored.
    """
    haystack = " " + _normalize(context) + " "
    found = []

    for number in dict.fromkeys(re.findall(r"\b\d[\d,.]*\d\b|\b\d\b", answer)):
        if f" {_normalize(number)} " not in haystack:
            found.append(number)

    run = r"[A-Z][\w'’.-]*(?:\s+(?:of|the|and|&|de|van|von)\s+[A-Z][\w'’.-]*|\s+[A-Z][\w'’.-]*)*"
    for match in dict.fromkeys(m.rstrip(".,;:'-") for m in re.findall(run, answer)):
        words = [w for w in re.split(r"\s+", match) if _normalize(w) not in _NOT_NAMES]
        while words and _normalize(words[0]) in _NOT_NAMES:
            words = words[1:]
        phrase = _normalize(" ".join(words))
        if not phrase:
            continue
        if f" {phrase} " in haystack or all(f" {w} " in haystack for w in phrase.split()):
            continue
        found.append(match)
    return found
