"""CLI used by the Streamlit demo: given one query, run the full
retrieve -> generate pipeline in a single subprocess and print the
result as JSON after a marker line.

Why a whole-pipeline subprocess rather than importing this logic
directly into the Streamlit app: Streamlit keeps one process alive
across every query in a session, and this machine can't reliably embed
a *second* query once faiss/google-genai are already loaded in that
same process (reproducible crashes -- see retriever.py and
run_pipeline.py for the encode-per-subprocess pattern this project
already relies on for the same reason). Giving every query its own
fresh process sidesteps that: the Streamlit app itself never imports
torch/faiss/google-genai at all, so its own memory footprint never
grows across a session.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import load_index_and_graph, valid_tracks_with_bios

RESULT_MARKER = "###RESULT_JSON###"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    valid_rows, bios = valid_tracks_with_bios()

    # Embed the query before importing faiss/google-genai below -- same
    # ordering reason as run_pipeline.py.
    from src.rag import retriever
    query_vec = retriever.embed_query(args.query)

    index, graph, track_metadata = load_index_and_graph(valid_rows, bios)

    from src.rag import llm_client, prompt_templates
    hits = retriever.retrieve(args.query, index, track_metadata, args.top_k, query_vec=query_vec, graph=graph)
    prompt = prompt_templates.build_rag_prompt(args.query, hits)
    answer = llm_client.generate_answer(prompt)

    result = {
        "query": args.query,
        "fused_dim": int(query_vec.shape[0]),
        "hits": [
            {
                "track_id": h["track_id"], "title": h["title"], "artist": h["artist"],
                "genre": h["genre"], "similarity": float(h["similarity"]),
                "modality_scores": retriever.modality_scores(index, h["track_id"], query_vec),
                "graph": h["graph"],
            }
            for h in hits
        ],
        "answer": answer,
    }
    # Everything printed above this point (model-loading progress bars,
    # HF warnings, subprocess diagnostics) is noise the caller should
    # ignore; only the JSON after this marker is the real result.
    print(RESULT_MARKER)
    print(json.dumps(result))
