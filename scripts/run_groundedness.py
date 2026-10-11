"""Milestone 6 groundedness evaluation: for each eval query, retrieve
(with graph expansion), generate a Gemini answer, then check it with
(1) the deterministic leaked-title check and (2) the LLM claim judge.

Reuses the query vectors cached by scripts/run_evaluation.py, so run
that first; no encoder subprocesses are spawned here. Appends a
"Groundedness" section to results.md (replacing any previous one).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from scripts.encode_modality import eval_query_cache_path
from scripts.run_evaluation import EVAL_QUERIES_JSON, MODALITIES, RESULTS_MD, unit

TOP_K = 3
SECTION_MARK = "## Groundedness"
PAUSE_SECONDS = 4  # stay well under free-tier requests-per-minute

PROGRESS_JSON = config.CACHE_DIR / "groundedness_progress.json"


def generate_with_retry(prompt: str, attempts: int = 4, model: str | None = None) -> str:
    """Call Gemini, retrying transient failures (llm_client raises
    RuntimeError for those) so one dropped connection doesn't discard a
    multi-minute evaluation run.
    """
    from src.rag import llm_client
    for attempt in range(1, attempts + 1):
        try:
            return llm_client.generate_answer(prompt, model=model)
        except RuntimeError as e:
            if attempt == attempts:
                raise
            print(f"[retry {attempt}] {e}")
            time.sleep(10 * attempt)


if __name__ == "__main__":
    from scripts.run_pipeline import assemble_index, valid_tracks_with_bios

    from src.embeddings import fusion
    from src.evaluation.groundedness_check import judge_claims, unretrieved_titles_mentioned
    from src.graph.build_graph import build_graph
    from src.rag import prompt_templates, retriever

    eval_set = json.loads(EVAL_QUERIES_JSON.read_text(encoding="utf-8"))
    valid_rows, bios = valid_tracks_with_bios()
    index, track_metadata = assemble_index(valid_rows, bios)
    graph = build_graph(valid_rows)
    catalog_titles = [m["title"] for m in track_metadata.values()]
    qvecs = {m: dict(np.load(eval_query_cache_path(m))) for m in MODALITIES}

    from google.genai import errors as genai_errors

    # Per-query answers/verdicts are saved as they arrive, so a daily-quota
    # stop (HTTP 429) can be resumed by simply re-running the script.
    progress = json.loads(PROGRESS_JSON.read_text(encoding="utf-8")) if PROGRESS_JSON.exists() else {}
    quota_hit = False
    for i, entry in enumerate(eval_set):
        query = entry["query"]
        rec = progress.setdefault(query, {})
        qv = fusion.fuse_embeddings(*(unit(qvecs[m][f"q{i}"]) for m in MODALITIES))
        hits = retriever.retrieve(query, index, track_metadata, TOP_K, query_vec=qv, graph=graph)
        context = prompt_templates.build_context_block(hits)
        try:
            if "answer" not in rec:
                t0 = time.perf_counter()
                rec["answer"] = generate_with_retry(prompt_templates.build_rag_prompt(query, hits))
                rec["answer_seconds"] = time.perf_counter() - t0
                time.sleep(PAUSE_SECONDS)
            if "claims" not in rec and "judge_error" not in rec:
                try:
                    rec["claims"] = judge_claims(
                        rec["answer"], context,
                        lambda p: generate_with_retry(p, model=config.JUDGE_MODEL_NAME))
                except (ValueError, KeyError) as e:
                    rec["judge_error"] = type(e).__name__
                time.sleep(PAUSE_SECONDS)
        except genai_errors.ClientError as e:
            if e.code != 429:
                raise
            print(f"[quota] 429 on query {i + 1}; progress saved, re-run later to resume.")
            quota_hit = True
        PROGRESS_JSON.write_text(json.dumps(progress, indent=1), encoding="utf-8")
        if quota_hit:
            break
        print(f"[{i + 1}/{len(eval_set)}] {query!r}: done")

    if any("answer" not in progress.get(e["query"], {}) or
           ("claims" not in progress[e["query"]] and "judge_error" not in progress[e["query"]])
           for e in eval_set):
        print("Not all queries complete yet; results.md not updated.")
        sys.exit(1)

    rows_md, total_claims, total_supported, leak_count, judge_failures = [], 0, 0, 0, 0
    answer_seconds = []
    for i, entry in enumerate(eval_set):
        rec = progress[entry["query"]]
        qv = fusion.fuse_embeddings(*(unit(qvecs[m][f"q{i}"]) for m in MODALITIES))
        hits = retriever.retrieve(entry["query"], index, track_metadata, TOP_K, query_vec=qv, graph=graph)
        leaked = unretrieved_titles_mentioned(rec["answer"], [h["title"] for h in hits], catalog_titles)
        leak_count += bool(leaked)
        answer_seconds.append(rec["answer_seconds"])
        if "claims" in rec:
            supported = sum(c["supported"] for c in rec["claims"])
            total_claims += len(rec["claims"])
            total_supported += supported
            unsupported = [c["claim"] for c in rec["claims"] if not c["supported"]]
            verdict = f"{supported}/{len(rec['claims'])}"
        else:
            judge_failures += 1
            unsupported, verdict = [], f"judge failed ({rec['judge_error']})"
        rows_md.append(f"| {entry['query']} | {verdict} | {', '.join(leaked) or '-'} | "
                       f"{'; '.join(unsupported) or '-'} |")

    lines = [SECTION_MARK, "",
             f"{len(eval_set)} queries, top-{TOP_K} retrieval with graph expansion, answers from "
             f"`{config.LLM_MODEL_NAME}`. Claims judged by `{config.JUDGE_MODEL_NAME}` (see caveat).", "",
             f"- Claims supported by retrieved context: **{total_supported}/{total_claims}** "
             f"({100 * total_supported / max(total_claims, 1):.1f}%)",
             f"- Answers naming a catalogue song that was not retrieved: **{leak_count}/{len(eval_set)}**",
             f"- Judge replies that failed to parse: {judge_failures}",
             f"- Gemini answer latency: {np.mean(answer_seconds):.1f}s mean "
             f"(min {min(answer_seconds):.1f}s, max {max(answer_seconds):.1f}s, includes retries)", "",
             "Caveat: the claim judge is also an LLM from the same vendor, so its verdicts "
             "are an estimate; the leaked-title check is the objective one.", "",
             "| Query | supported claims | leaked titles | unsupported claims |", "|---|---|---|---|"] + rows_md
    existing = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    base = existing.split(SECTION_MARK)[0].rstrip() + "\n\n" if existing else ""
    RESULTS_MD.write_text(base + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
