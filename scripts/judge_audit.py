"""Audit the LLM claim judge against your own labels.

Two steps, run after scripts/run_groundedness.py has produced answers
and judge verdicts:

  python -m scripts.judge_audit export   # writes data/processed/judge_audit.csv
  (open the CSV, fill the `human_supported` column with yes/no for each row,
   judging each claim against the `context` column only)
  python -m scripts.judge_audit score    # agreement between you and the judge

Why: groundedness numbers from an LLM judge are only as trustworthy as the
judge. Labelling ~20 claims by hand once turns "the judge is probably
fine" into a measured agreement rate (and shows which way it errs).
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

PROGRESS_JSON = config.CACHE_DIR / "groundedness_progress.json"
AUDIT_CSV = config.PROCESSED_DIR / "judge_audit.csv"
SAMPLE_SIZE = 20
FIELDS = ["id", "query", "claim", "judge_supported", "human_supported", "context"]


def sample_claims(progress: dict, n: int = SAMPLE_SIZE, seed: int = 0) -> list[dict]:
    """Pick n judged claims, deliberately over-sampling the claims the judge
    marked unsupported (up to half), since those are rare and are where a
    judge that is too strict or too lenient shows up.
    """
    pool = [{"query": q, "claim": c["claim"], "judge_supported": c["supported"], "context": rec.get("context", "")}
            for q, rec in progress.items() for c in rec.get("claims", [])]
    rng = random.Random(seed)
    unsupported = [c for c in pool if not c["judge_supported"]]
    supported = [c for c in pool if c["judge_supported"]]
    rng.shuffle(unsupported)
    rng.shuffle(supported)
    chosen = unsupported[: n // 2]
    leftovers = supported + unsupported[n // 2:]      # fill the rest from everything not yet chosen
    chosen += leftovers[: n - len(chosen)]
    rng.shuffle(chosen)
    return [{"id": i, **c} for i, c in enumerate(chosen, start=1)]


def agreement(rows: list[dict]) -> dict:
    """Compare judge verdicts with human labels ('yes'/'no') on labelled rows.

    Returns counts plus the two error types separately: 'too lenient' =
    judge said supported but the human said not; 'too strict' = the reverse.
    """
    labelled = [r for r in rows if r["human_supported"].strip().lower() in ("yes", "no")]
    out = {"labelled": len(labelled), "agree": 0, "too_lenient": 0, "too_strict": 0}
    for r in labelled:
        human = r["human_supported"].strip().lower() == "yes"
        judge = str(r["judge_supported"]).lower() == "true"
        if human == judge:
            out["agree"] += 1
        elif judge and not human:
            out["too_lenient"] += 1
        else:
            out["too_strict"] += 1
    out["rate"] = out["agree"] / out["labelled"] if out["labelled"] else float("nan")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "export":
        if not PROGRESS_JSON.exists():
            sys.exit("No groundedness progress found: run scripts.run_groundedness first.")
        rows = sample_claims(json.loads(PROGRESS_JSON.read_text(encoding="utf-8")))
        with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({**r, "human_supported": ""} for r in rows)
        print(f"Wrote {len(rows)} claims to {AUDIT_CSV}. Fill human_supported with yes/no, then run `score`.")
    elif mode == "score":
        with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
            result = agreement(list(csv.DictReader(f)))
        print(f"{result['labelled']} labelled claims: judge agrees with you on {result['agree']} "
              f"({100 * result['rate']:.0f}%); too lenient {result['too_lenient']}, too strict {result['too_strict']}.")
    else:
        sys.exit("usage: python -m scripts.judge_audit export|score")
