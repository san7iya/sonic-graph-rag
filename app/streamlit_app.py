"""Milestone 7 demo UI: a text box in, retrieved evidence + grounded
Gemini answer out.

Every query is answered by shelling out to `scripts/answer_query.py` as
a fresh subprocess (see that file's docstring): this Streamlit process
itself never imports torch/faiss/google-genai, so its memory footprint
stays flat no matter how many queries get asked in one session. That
matters on this machine specifically, where re-embedding a query in a
process that already has faiss/google-genai loaded was reproducibly
unreliable.
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import config
from scripts.answer_query import RESULT_MARKER

st.set_page_config(page_title="Music KG-RAG", page_icon="🎵", layout="wide")


def run_query(query: str, top_k: int) -> tuple[dict | None, str]:
    """Run one query in a fresh subprocess and return (result, raw_log).

    result is None if the subprocess failed or produced no parseable
    JSON; raw_log is the full stdout+stderr for debugging in that case.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.answer_query", query, "--top-k", str(top_k)],
        cwd=config.ROOT_DIR, capture_output=True, text=True,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    if RESULT_MARKER not in (proc.stdout or ""):
        return None, log
    json_str = proc.stdout.split(RESULT_MARKER, 1)[1].strip()
    try:
        return json.loads(json_str), log
    except json.JSONDecodeError:
        return None, log


st.title("🎵 Multi-Modal Music Knowledge Graph — RAG Demo")
st.caption(
    "Milestone 0 vertical slice · 18 tracks across 6 genres · text (MiniLM) + image (CLIP) + "
    "audio (CLAP) fused per song · FAISS retrieval · Gemini 3.6-flash generation, grounded in "
    "retrieved context only · no fine-tuning, baseline normalize-and-concatenate fusion"
)

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "1. **Chunk → tokenize → encode → pool → normalize** each track's bio (MiniLM, 384-dim)\n"
        "2. **Encode** cover art (CLIP, 512-dim) and a 30s audio preview (CLAP, 512-dim)\n"
        "3. **Fuse**: L2-normalize each modality vector, concatenate → 1408-dim\n"
        "4. **Index**: FAISS `IndexFlatIP` over all fused vectors\n"
        "5. **Retrieve**: embed the query through each model's own *text* tower, "
        "search top-k\n"
        "6. **Generate**: retrieved context → Gemini 3.6-flash, answer only from context"
    )
    st.divider()
    st.caption(
        "Every model load happens in its own subprocess — this machine's free RAM can't hold "
        "MiniLM, CLIP, and CLAP resident at once, so each modality (and each query) gets a "
        "fresh process. A query typically takes 5-15s for that reason."
    )

query = st.text_input(
    "Ask a question about the music collection",
    value="Which songs have an upbeat, danceable feel?",
)
top_k = st.slider("How many tracks to retrieve", min_value=3, max_value=10, value=5)
submitted = st.button("Ask", type="primary")

if submitted and query.strip():
    with st.spinner("Embedding query, retrieving, and asking Gemini — real inference, not cached..."):
        result, log = run_query(query.strip(), top_k)

    if result is None:
        st.error("The query pipeline failed. Raw output below (this can happen under this "
                  "machine's tight memory budget — retry usually works).")
        with st.expander("Raw subprocess output"):
            st.code(log or "(no output captured)")
    else:
        st.subheader(f"Top {len(result['hits'])} retrieved tracks")
        cols = st.columns(len(result["hits"]) or 1)
        for col, hit in zip(cols, result["hits"]):
            with col:
                art_path = config.RAW_IMAGE_DIR / f"{hit['track_id']}.jpg"
                if art_path.exists():
                    st.image(str(art_path), use_container_width=True)
                st.markdown(f"**{hit['title']}**")
                st.caption(f"{hit['artist']} · {hit['genre']}")
                st.metric("similarity", f"{hit['similarity']:.3f}")

        st.divider()
        st.subheader("Gemini answer (grounded in the context above)")
        st.info(result["answer"])

        with st.expander("Real vector info (for the 'how does this work' questions)"):
            st.markdown(
                f"- Fused vector dimensionality: **{result['fused_dim']}** "
                f"(384 text + 512 image + 512 audio)\n"
                "- Each modality vector is L2-normalized independently before concatenation, "
                "per the baseline fusion spec\n"
                "- Similarity score above is an inner product over the fused vectors — since "
                "each 384/512/512 block is separately unit-normalized rather than the full "
                "1408-dim vector, this is the *sum* of three per-modality cosine similarities "
                "(range roughly [-3, 3]), not a single bounded cosine similarity"
            )
elif submitted:
    st.warning("Type a query first.")
