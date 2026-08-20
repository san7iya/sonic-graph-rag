# Multi-Modal Knowledge Graph with LLM-Based Semantic Fusion

Project memory file for Claude Code. Place this at the repo root as `CLAUDE.md` — Claude Code loads it automatically at the start of every session in this directory.

## How to work with me on this

- I am building this solo (originally scoped as a 3-person project, now one person doing all three roles), for a Foundations of Data Science course. Review 1 is imminent — treat Milestone 0 below as the current priority until I say otherwise.
- I explicitly do not want vibecoded output. Every function should be something I can explain in a viva/review without hesitation. Prefer straightforward, readable code over clever abstractions.
- Add a short docstring to every function explaining *why*, not just what — especially in the embedding, fusion, and retrieval modules, since I'll be questioned on the mechanism, not just the output.
- Work in small, verifiable steps. After each milestone, show me a concrete way to check it worked (a printed value, a test passing, a plot) rather than just saying it's done.
- Don't silently swap a library, model, or design decision listed below — if something in this spec turns out to be a bad idea (wrong package name, deprecated API, etc.), flag it and ask before changing course.
- If a step needs an API key or credential from me, stop and ask rather than hardcoding or inventing one.

---

## 1. Problem Statement

Music metadata is fragmented across modalities that are rarely reasoned about together: lyrics/bio (text), album art (image), and the actual audio (sound). A query like "find songs that sound like X but with happier lyrics" can't be answered by any single-modality system. This project builds a Knowledge Graph where each node (song, artist, album) has a representation fused from text, image, and audio, and uses an LLM with Retrieval-Augmented Generation (RAG) to answer natural-language queries grounded in that graph — not in the LLM's own unverified prior knowledge.

## 2. Non-Negotiable Constraints

- **No model training or fine-tuning.** Every encoder and the LLM are used via pretrained inference only.
- **Zero budget.** Free-tier APIs and open-source models only.
- **Small subset, not the full dataset.** Target 15–20 tracks for Milestone 0, scaling to 500–2000 for later milestones — never assume the full dataset is in scope unless I say so.
- **Baseline fusion first.** Simple normalize-and-concatenate fusion is the Milestone 0–3 target. A learned/weighted fusion variant is a stretch goal, only after baseline retrieval quality is measured and a concrete weakness identified.
- Every design decision below should trace back to something I can defend in a review — if you add a new dependency or technique not listed here, tell me why in plain terms.

## 3. Tech Stack

| Layer | Library / Model | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Text embedding | `sentence-transformers`, model `all-MiniLM-L6-v2` | 384-dim, contextual, contrastively pretrained |
| Image embedding | `open_clip_torch` or `transformers` CLIP, model `openai/clip-vit-base-patch32` | 512-dim, aligned to text space |
| Audio embedding | `transformers` CLAP, model `laion/clap-htsat-unfused` | 512-dim, aligned to text space |
| Knowledge graph | `networkx` (default) | Zero-infra, in-process; export to Neo4j only if graph queries genuinely outgrow it — don't add Neo4j speculatively |
| Vector index | `faiss-cpu`, `IndexFlatIP` | Inputs must be L2-normalized so inner product = cosine similarity |
| LLM | Gemini API, package `google-genai` (NOT the older `google-generativeai` — that's being phased out), model `gemini-2.5-flash` or `gemini-2.0-flash` | Free tier; used only for generation, never for embeddings |
| Demo UI | `streamlit` | Later milestone only |
| Testing | `pytest` | One test file per non-trivial module |

Verify current package names/APIs with a quick doc check if anything here throws an import or deprecation error — don't just patch around it silently.

## 4. Repository Structure

```
music-kg-rag/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example              # GEMINI_API_KEY=
├── .gitignore                # data/raw, data/cache, .env
├── data/
│   ├── raw/                  # downloaded audio/images/text (gitignored)
│   ├── processed/            # cleaned data
│   └── cache/                # cached embeddings, FAISS index, graph export
├── src/
│   ├── config.py             # paths, constants, model names in ONE place
│   ├── ingestion/
│   │   ├── download_audio.py
│   │   ├── fetch_album_art.py       # Cover Art Archive API
│   │   └── fetch_text_metadata.py   # Wikipedia API
│   ├── preprocessing/
│   │   ├── clean_audio.py    # resample to 16kHz mono
│   │   ├── clean_text.py
│   │   └── clean_images.py
│   ├── graph/
│   │   ├── schema.py         # node/edge type definitions
│   │   ├── build_graph.py
│   │   └── graph_utils.py
│   ├── embeddings/
│   │   ├── text_encoder.py   # MiniLM wrapper: chunk -> tokenize -> encode -> pool -> normalize
│   │   ├── image_encoder.py  # CLIP wrapper
│   │   ├── audio_encoder.py  # CLAP wrapper
│   │   └── fusion.py         # normalize + concatenate
│   ├── indexing/
│   │   └── faiss_index.py
│   ├── rag/
│   │   ├── retriever.py      # query -> embed -> top-k -> graph expansion
│   │   ├── prompt_templates.py
│   │   └── llm_client.py     # Gemini wrapper
│   └── evaluation/
│       ├── precision_at_k.py
│       └── groundedness_check.py
├── scripts/
│   ├── run_pipeline.py       # end-to-end runner, this is what I demo at review
│   └── demo_query.py         # single query -> answer, for quick manual testing
├── tests/
│   ├── test_text_encoder.py
│   ├── test_fusion.py
│   ├── test_faiss_index.py
│   └── test_retriever.py
└── app/
    └── streamlit_app.py       # Milestone 6+ only
```

## 5. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in GEMINI_API_KEY
```

`requirements.txt` should pin at least: `sentence-transformers`, `transformers`, `torch`, `open_clip_torch`, `faiss-cpu`, `networkx`, `google-genai`, `streamlit`, `pytest`, `numpy`, `requests`, `python-dotenv`.

I'll get a Gemini API key myself from Google AI Studio — flag clearly if a step needs it and can't proceed without it, rather than stubbing around it invisibly.

## 6. Data Sources

**Milestone 0 note:** GTZAN's 1,000 clips (`blues.00000.au`, etc.) carry only a genre label — no real artist/song identity, and no bundled CSV mapping clips to actual songs (that line below was wrong; flagged and corrected 2026-08-20). For the 15–20 track vertical slice, per-track modality identity matters (art/bio/audio must all be the *same* song), so Milestone 0 sources audio from the iTunes Search API instead (free, no key, real 30s previews keyed by artist/title). GTZAN is deferred to Milestone 1+, where it's used for bulk genre-classification scale and per-song identity is less load-bearing.

| Modality | Source | Access | Notes |
|---|---|---|---|
| Audio (Milestone 0) | iTunes Search API | Free public API, no key | Real 30s preview per real song, keyed by artist/title |
| Audio (Milestone 1+) | GTZAN (1,000 clips, 10 genres, ~1.2GB) | Direct download | Anonymous per-clip identity — fine at bulk/genre scale, not for per-song demos. Fallback: FMA-small if more entities are needed for graph structure |
| Album art | Cover Art Archive | Free public API | Fallback: MusicBrainz cover art links |
| Lyrics / bio | Wikipedia API | Free public API | Light cleaning needed (strip markup) |
| Structured metadata | Curated manually for M0; GTZAN-bundled labels for M1+ | Local file | Artist, album, genre, track ID — use directly as graph fields, don't re-derive |

## 7. Build Order (Milestones)

Work through these in order. Each has a concrete "done" check — don't mark a milestone complete without running it.

### Milestone 0 — Review-1 Vertical Slice (CURRENT PRIORITY)
Thin end-to-end proof on 15–20 tracks spanning at least 3 genres.
1. Manually curate/download 15–20 tracks + their album art + bios.
2. Run each through its encoder (MiniLM/CLIP/CLAP); print raw vector shapes and a few real values.
3. Fuse (normalize + concatenate), store in a flat FAISS index.
4. Run one real query end-to-end: embed query → retrieve top-k → print actual similarity scores → pass retrieved context to Gemini → print the grounded answer.
5. **Done when:** running `scripts/run_pipeline.py` on the small sample prints real vectors, real similarity scores, and a real LLM answer — no placeholders, no mocked output.

### Milestone 1 — Full Data Collection & Cleaning
Scale ingestion to the full subset (500–2000 tracks). Resample audio, validate images, clean text.
**Done when:** `data/processed/` has a consistent record per track with all three modalities present or explicitly flagged missing.

### Milestone 2 — Knowledge Graph Construction
Build nodes (Song, Artist, Album, Genre) and explicit edges (PERFORMED_BY, BELONGS_TO_ALBUM, HAS_GENRE) in networkx from structured metadata.
**Done when:** graph can be queried for "all songs by artist X" and "all songs in genre Y" and returns correct results.

### Milestone 3 — Embeddings + Fusion at Scale
Run all three encoders across the full subset, cache to disk (never recompute live). Fuse per node.
**Done when:** every node has a cached fused vector; a quick nearest-neighbor check on 2–3 known-similar songs returns each other as top matches.

### Milestone 4 — Indexing + Retrieval
Load fused vectors into FAISS. Implement retriever: query → embed → top-k → 1-hop graph expansion for relational context.
**Done when:** a text query returns plausible top-k songs plus their graph neighborhood (artist, album, genre).

### Milestone 5 — RAG Generation
Prompt template that clearly delimits retrieved context from instructions, tells the LLM to answer only from context. Wire to Gemini.
**Done when:** 10–15 test queries produce answers that can be manually traced back to specific retrieved chunks.

### Milestone 6 — Evaluation
Precision@k on a held-out query set with manual relevance labels. Groundedness check (does every claim in the answer trace to retrieved evidence?). Latency measurement.
**Done when:** `results.md` exists with actual numbers, not estimates.

### Milestone 7 — Demo UI (time permitting)
Minimal Streamlit app: text box in, retrieved evidence + LLM answer out.
**Done when:** I can run `streamlit run app/streamlit_app.py` and query it live.

## 8. Key Technical Specs (keep consistent — these are what I'll be asked about)

- **Chunking:** 200–400 tokens per chunk, 15–20% overlap, for any long text (bios).
- **Text pipeline:** tokenize (subword) → transformer self-attention (contextual per-token vectors) → mean-pool (or [CLS]) → L2-normalize → store.
- **Why retrieval works:** MiniLM/CLIP/CLAP are all trained with a **contrastive objective** — similar pairs pulled together, dissimilar pushed apart — which is why cosine distance tracks semantic similarity. This is the answer to "how/why does this work," not just "we call an API."
- **Contextual vs. static embeddings:** we use contextual embeddings (same word/token gets a different vector depending on context) — not Word2Vec/GloVe-style static embeddings. Be able to say this distinction out loud.
- **Fusion:** L2-normalize each modality vector independently, then concatenate (384 + 512 + 512 = 1408-dim). Do not average or weight without a measured reason to.
- **Similarity metric:** cosine similarity via inner product on normalized vectors (`IndexFlatIP`).

## 9. Explicitly Out of Scope — Do Not Add These

- No fine-tuning of any encoder or the LLM.
- No Neo4j server setup unless networkx genuinely can't do what's needed.
- No learned/weighted fusion network until baseline fusion has been measured and found lacking.
- No scraping beyond the listed free APIs.
- No paid API tiers or credit-card-gated services.

## 10. Review 1 Deliverables Checklist

- [ ] LLM chosen and justified (Gemini 2.0/2.5 Flash, free tier, multimodal)
- [ ] Working demo of text → vector conversion with real numbers shown (chunking → tokenize → encode → pool → normalize)
- [ ] Clear articulation of contextual vs. static word embeddings, tied to why this app needs contextual
- [ ] Full RAG loop run live: retrieval → Gemini call → grounded answer, not just described
- [ ] Scope stated explicitly (subset size, no fine-tuning, baseline fusion) so the demo doesn't over-promise
