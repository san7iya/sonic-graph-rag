"""Rule-based query routing: pick modality weights from the wording.

Idea: "sounds like a slow ballad" is about audio, "songs about heartbreak"
is about text, "dark moody album cover" is about the image. A fixed
equal-weight fusion treats them all alike. The router looks for cue words
and, if exactly one modality is clearly favoured, up-weights it.

Deliberately simple (keyword lists, no model) so it can be explained in
one sentence, and opt-in: the project's rule is no weighting without a
measured reason, so scripts/run_routing_evaluation.py measures it against
the equal-weight baseline before it is ever used by default.
"""

from __future__ import annotations

import re

EQUAL = (1 / 3, 1 / 3, 1 / 3)       # (text, image, audio) -- the baseline
FAVOURED_WEIGHT, OTHER_WEIGHT = 0.6, 0.2

CUES = {
    "audio": ["sounds like", "sound", "sounds", "beat", "beats", "tempo", "rhythm", "rhythmic", "instrumental",
              "melody", "melodic", "loud", "quiet", "calm", "upbeat", "danceable", "fast", "slow", "vocals",
              "guitar", "saxophone", "trumpet", "piano", "orchestral", "groove", "energetic", "acoustic"],
    "text": ["lyrics", "about", "story", "meaning", "written", "wrote", "history", "released", "era", "century",
             "composer", "career", "biography", "born", "famous", "songs by", "songwriter"],
    "image": ["cover", "artwork", "album art", "looks", "looking", "visual", "colorful", "colourful",
              "photo", "picture", "illustration"],
}


def route_query(query: str) -> tuple[float, float, float]:
    """Return (w_text, w_image, w_audio), summing to 1.

    The modality with strictly the most cue-word hits is favoured; a tie
    or no hits leaves the equal-weight baseline, so the router only acts
    when the wording is unambiguous.
    """
    text = query.lower()
    hits = {m: sum(bool(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)) for c in cues)
            for m, cues in CUES.items()}
    best = max(hits.values())
    winners = [m for m, n in hits.items() if n == best]
    if best == 0 or len(winners) != 1:
        return EQUAL
    return tuple(FAVOURED_WEIGHT if m == winners[0] else OTHER_WEIGHT for m in ("text", "image", "audio"))
