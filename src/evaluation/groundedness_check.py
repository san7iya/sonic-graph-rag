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
