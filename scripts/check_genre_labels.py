"""Cross-check curated genres against iTunes' tags and list disagreements
for review. Needs data/processed/catalog_extra.csv from
`python -m src.ingestion.enrich_catalog`. Writes results_labels.md.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation.label_check import compare_labels

OUT = config.ROOT_DIR / "results_labels.md"
EXTRA_CSV = config.PROCESSED_DIR / "catalog_extra.csv"

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with open(config.CATALOG_CSV, newline="", encoding="utf-8") as f:
        catalog = {r["track_id"]: r for r in csv.DictReader(f)}
    with open(EXTRA_CSV, newline="", encoding="utf-8") as f:
        extra = list(csv.DictReader(f))
    rows = [{**catalog[e["track_id"]], "itunes_genre": e["itunes_genre"]} for e in extra if e["track_id"] in catalog]
    result = compare_labels(rows)

    matched = sum(e["matched"] == "yes" for e in extra)
    lines = ["# Genre label cross-check (curated vs iTunes tags)", "",
             f"{len(rows)} catalogue tracks; {matched} matched to an iTunes result; {result['mapped']} had an iTunes tag "
             f"that maps to one of the six genres (others, e.g. Soundtrack or Folk, say nothing either way). "
             f"Agreement on those: **{100 * result['agreement']:.1f}%**. iTunes tags are per-release and noisy, so "
             "this is a review aid, not ground truth.", "",
             "| Curated genre | tracks compared | agree |", "|---|---|---|"]
    lines += [f"| {g} | {t} | {a} ({100 * a / t:.0f}%) |" for g, (t, a) in sorted(result["per_genre"].items())]
    lines += ["", f"## Disagreements to review ({len(result['disagreements'])})", "",
              "| track_id | curated | iTunes tag | maps to |", "|---|---|---|---|"]
    lines += [f"| {d['track_id']} | {d['curated']} | {d['itunes']} | {d['itunes_maps_to']} |"
              for d in result["disagreements"]]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"... full list in {OUT}")
