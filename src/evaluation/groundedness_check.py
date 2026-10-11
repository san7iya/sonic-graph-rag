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
              "graph", "similar", "other", "one", "two", "three", "if", "there", "here", "to", "with", "by",
              "none", "note", "additionally", "additional", "because"}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_in_words(n: int) -> str:
    """English words for 0..999 ("41" -> "forty one"), hyphens as spaces so it
    matches normalised text. Larger numbers return "" (checked as digits only).
    """
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + number_in_words(n % 100) if n % 100 else "")
    return ""


def titles_not_in_context(answer: str, context: str, catalog_titles: list[str]) -> list[str]:
    """Catalogue titles the answer names that appear nowhere in the context.

    The context deliberately contains more than the top-k tracks (graph facts
    list the artist's other songs, genre examples and similar songs), so a
    song named from those lines is grounded. Only a title absent from the
    entire context text is a real leak of outside knowledge.
    """
    context_lower = context.lower()
    answer_lower = answer.lower()
    leaked = []
    for title in catalog_titles:
        t = title.lower()
        if t in context_lower:
            continue
        if " " not in t.strip():
            # A one-word title ("One", "Spring") is also an ordinary word, so only a
            # quoted mention counts as naming the song.
            pattern = r"[\"“‘']" + re.escape(t) + r"[\"”’']"
        else:
            pattern = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
        if re.search(pattern, answer_lower):
            leaked.append(title)
    return leaked


_BOLD_LABEL = re.compile(r"(?m)^[\s>*\-\u2022\d.)]*\*\*([^*\n\"\u201c]*)\*\*")
_NAME_RUN = (r"[A-Z][\w'\u2019.-]*(?:\s+(?:of|the|and|&|de|van|von)\s+[A-Z][\w'\u2019.-]*"
             r"|\s+[A-Z][\w'\u2019.-]*)*")


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

    Numbers match as digits or as English words ("41" is supported by
    "forty-one"). Names are runs of capitalised words (joined by 'of', '&',
    'the', 'and' only inside a run); a run counts as supported if the whole
    phrase, or every word of it, occurs in the context. To avoid flagging
    formatting, bold labels at the start of a bullet ("**Solo Innovation:**")
    are dropped, and a capitalised first word of a sentence is not treated
    as part of a name (a lone one is skipped; in a longer run it is dropped).
    """
    haystack = " " + _normalize(context) + " "
    found = []

    for number in dict.fromkeys(re.findall(r"\b\d[\d,.]*\d\b|\b\d\b", answer)):
        digits = number.replace(",", "")
        spelled = number_in_words(int(digits)) if digits.isdigit() else ""
        if f" {_normalize(number)} " in haystack or (spelled and f" {spelled} " in haystack):
            continue
        found.append(number)

    text = _BOLD_LABEL.sub(" ", answer).replace("*", " ").replace("_", " ")
    names = []
    for m in re.finditer(_NAME_RUN, text):
        run = m.group().rstrip(".,;:'-")
        prefix = text[:m.start()].rstrip(" \t>-\u2022")
        sentence_initial = prefix == "" or prefix[-1] in ".!?:\n"
        words = run.split()
        if sentence_initial:
            if len(words) == 1:
                continue
            words = words[1:]
        names.append(" ".join(words))

    for match in dict.fromkeys(names):
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
