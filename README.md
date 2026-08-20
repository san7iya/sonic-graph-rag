# Multi-Modal Music Knowledge Graph with LLM-Based Semantic Fusion

A Knowledge Graph where each song node's representation is fused from
**text** (bio), **image** (album art), and **audio** (30s preview), so a
query like *"find songs that sound like X but with happier lyrics"* can be
answered by retrieval across all three modalities together — not by any
single-modality system, and not from the LLM's own unverified prior
knowledge (Retrieval-Augmented Generation, grounded in retrieved context
only).

Full spec, milestone plan, and design rationale live in [CLAUDE.md](CLAUDE.md).
This README covers what's built and how to run it.

## Status: Milestone 0 complete (Review 1 vertical slice)

18 tracks across 6 genres (rock, pop, hip-hop, jazz, classical, reggae),
each with real bio + album art + audio preview, run through the full
pipeline: encode → fuse → index → retrieve → generate. No placeholders —
every vector, similarity score, and LLM answer comes from actually
running the pipeline.

**Explicit scope for this milestone** (see CLAUDE.md §2 for the full list):
no fine-tuning of any encoder or the LLM, zero-budget free-tier APIs only,
15–20 tracks (not the full dataset), baseline normalize-and-concatenate
fusion (no learned/weighted fusion).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows; source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
copy .env.example .env         # then fill in GEMINI_API_KEY from Google AI Studio
```

## Running it

```bash
python -m scripts.run_pipeline                        # full pipeline, one demo query, prints everything
python -m scripts.answer_query "your question" --top-k 5   # single query, JSON result
streamlit run app/streamlit_app.py                     # interactive demo UI
```

`scripts/run_pipeline.py` is the end-to-end script to demo: it prints
real vector shapes per track, real similarity scores at retrieval time,
and the real grounded Gemini answer.

## How it works

| Step | Model | Where |
|---|---|---|
| Text embedding | MiniLM (`all-MiniLM-L6-v2`), 384-dim | [src/embeddings/text_encoder.py](src/embeddings/text_encoder.py) |
| Image embedding | CLIP (`clip-vit-base-patch32`), 512-dim | [src/embeddings/image_encoder.py](src/embeddings/image_encoder.py) |
| Audio embedding | CLAP (`clap-htsat-unfused`), 512-dim | [src/embeddings/audio_encoder.py](src/embeddings/audio_encoder.py) |
| Fusion | L2-normalize each modality, concatenate → 1408-dim | [src/embeddings/fusion.py](src/embeddings/fusion.py) |
| Index | FAISS `IndexFlatIP` (cosine via inner product on unit vectors) | [src/indexing/faiss_index.py](src/indexing/faiss_index.py) |
| Retrieval | query → each model's own text tower → top-k | [src/rag/retriever.py](src/rag/retriever.py) |
| Generation | Gemini 3.6-flash, answer only from retrieved context | [src/rag/llm_client.py](src/rag/llm_client.py), [src/rag/prompt_templates.py](src/rag/prompt_templates.py) |

**Text pipeline in detail** (this is the one most likely to get asked
about in review): chunk (overlapping word windows) → tokenize (subword,
via MiniLM's own tokenizer) → encode (transformer self-attention →
contextual per-token vectors) → mean-pool → L2-normalize.

**Why this works — contrastive training, not just an API call**: MiniLM,
CLIP, and CLAP are each trained with a contrastive objective (similar
pairs pulled together, dissimilar pairs pushed apart), which is *why*
cosine similarity between their output vectors tracks semantic
similarity. CLIP and CLAP specifically align their non-text modality to a
shared *text* embedding space, which is what lets a plain-text query
retrieve against real image/audio vectors, not just against bios.

**Contextual, not static, embeddings**: these are contextual embeddings —
the same word gets a different vector depending on surrounding context —
unlike static Word2Vec/GloVe embeddings, which give one fixed vector per
word regardless of context. This app needs contextual embeddings because
a bio's meaning depends on how words combine (e.g. "heavy" in "heavy
metal" vs. "heavy heart"), not just which words are present.

## A note on this machine's constraints

This machine's free RAM can't hold MiniLM, CLIP, and CLAP resident in the
same process at once (confirmed by reproducible crashes). Every encoder
call — corpus encoding and query encoding alike — runs in its own
subprocess for exactly this reason (see
[scripts/encode_modality.py](scripts/encode_modality.py)), and the
Streamlit demo shells out to a fresh subprocess per query rather than
keeping models loaded across a session (see
[scripts/answer_query.py](scripts/answer_query.py)). Under tight memory
pressure a subprocess can still occasionally crash with a transient
allocation failure; there's a retry-with-backoff around every subprocess
call for this reason, and the demo UI surfaces a "retry usually works"
message if all retries are exhausted. This is a real, observed constraint
of the current dev machine — not a hypothetical.

## Testing

```bash
pytest
```

Covers the fusion math (correct shape, each modality block independently
unit-normalized), FAISS retrieval ordering, the retrieval→metadata
attachment logic, bio chunking, and one real MiniLM inference call.
Deliberately *not* included in the same test run: CLIP/CLAP encoder
tests, for the same reason described above — loading multiple heavy
models in one process is the exact crash this project's
subprocess-per-encoder pattern exists to avoid. `embed_query()`'s real
subprocess pipeline is exercised live by `scripts/run_pipeline.py` and
`scripts/answer_query.py` instead.
