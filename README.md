# Multi-Modal Music Knowledge Graph with LLM-Based Semantic Fusion

Music metadata is fragmented across modalities that are rarely reasoned
about together: lyrics/bio (text), album art (image), and the audio
itself (sound). A query like *"find songs that sound like X but with
happier lyrics"* can't be answered by any single-modality system.

This project builds a Knowledge Graph where each song node's
representation is fused from text, image, and audio embeddings, and uses
an LLM with Retrieval-Augmented Generation (RAG) to answer natural-language
queries grounded in that graph — not in the LLM's own unverified prior
knowledge.

Built entirely on pretrained, open-source, free-tier components — no
model training or fine-tuning, no paid APIs.

## Current status

Proof of concept complete on an 18-track sample spanning 6 genres (rock,
pop, hip-hop, jazz, classical, reggae): every track's bio, album art, and
a 30-second audio preview are encoded, fused into one vector, indexed,
and retrievable through a live RAG query. Nothing here is a placeholder —
every embedding, similarity score, and generated answer comes from
actually running the pipeline against real data.

Next up: scaling ingestion to 500–2000 tracks, building out the explicit
knowledge graph (artist/album/genre relations), and a formal
precision@k / groundedness evaluation.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows; source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
copy .env.example .env         # then fill in GEMINI_API_KEY from Google AI Studio
```

## Running it

```bash
python -m scripts.run_pipeline                             # full pipeline, one demo query, prints everything
python -m scripts.answer_query "your question" --top-k 5    # single query, JSON result
streamlit run app/streamlit_app.py                          # interactive demo UI
```

`scripts/run_pipeline.py` is the end-to-end script: it prints real vector
shapes per track, real similarity scores at retrieval time, and the real
grounded Gemini answer.

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

**Text pipeline:** chunk (overlapping word windows) → tokenize (subword,
via MiniLM's own tokenizer) → encode (transformer self-attention →
contextual per-token vectors) → mean-pool → L2-normalize.

**Why retrieval works:** MiniLM, CLIP, and CLAP are each trained with a
contrastive objective (similar pairs pulled together, dissimilar pairs
pushed apart), which is why cosine similarity between their output
vectors tracks semantic similarity. CLIP and CLAP specifically align
their non-text modality to a shared *text* embedding space, which is what
lets a plain-text query retrieve against real image/audio vectors, not
just against bios.

**Contextual, not static, embeddings:** the same word gets a different
vector depending on surrounding context (e.g. "heavy" in "heavy metal" vs.
"heavy heart"), unlike static Word2Vec/GloVe embeddings, which assign one
fixed vector per word regardless of context.

## Design note: subprocess-isolated encoding

Each encoder loads its own pretrained model (roughly 90MB–600MB). To keep
peak memory usage bounded rather than holding all three models resident
at once, encoding and query embedding each run in an isolated subprocess
([scripts/encode_modality.py](scripts/encode_modality.py)) that exits
(and releases its memory) as soon as it's done; the Streamlit demo shells
out to a fresh subprocess per query for the same reason
([scripts/answer_query.py](scripts/answer_query.py)). A short
retry-with-backoff wraps each subprocess call to absorb the occasional
transient allocation failure this pattern can hit under memory pressure.

## Testing

```bash
pytest
```

Covers the fusion math (correct shape, each modality block independently
unit-normalized), FAISS retrieval ordering, the retrieval→metadata
attachment logic, bio chunking, and one real MiniLM inference call. CLIP
and audio encoder tests are intentionally excluded from this test run, in
keeping with the subprocess-isolation design above — `embed_query()`'s
full subprocess pipeline is instead exercised live by
`scripts/run_pipeline.py` and `scripts/answer_query.py`.

## Scope

No fine-tuning of any encoder or the LLM — every model runs via
pretrained inference only. Free-tier APIs and open-source models
throughout, zero budget. Fusion is baseline normalize-and-concatenate; a
learned/weighted fusion variant is a possible future direction once
retrieval quality is measured against this baseline.
