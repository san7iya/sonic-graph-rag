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

**Working end to end** on a 521-track corpus (rock 104, pop 95, hip-hop 82,
jazz 87, classical 76, reggae 77), built from curated artist lists via the
iTunes, Wikipedia and Cover Art Archive APIs. Every track's bio, album art
and 30-second audio preview is validated, encoded, fused into one vector,
indexed, linked into a knowledge graph and retrievable through a live RAG
query. Of 547 catalogued tracks, 26 were excluded by validation: missing
files, and 11 tracks identified as the wrong song (for example Lil Nas X
recordings matched to "Nas"). Nothing here is a placeholder — every
embedding, score and answer comes from running the pipeline on real data.

Headline results (95% bootstrap intervals; full tables in `results_*.md`):

- **Fusion helps.** A song's 5 nearest neighbours share its genre 91% of
  the time in the fused space, against 76–77% for any single modality
  (chance is 17%); the graph's SIMILAR_TO edges reach 95%.
- **Genre queries** (18 queries, 521 tracks): fused P@10 0.91
  [0.83, 0.97], and the top result always had the right genre. Image-only
  scores 0.89, text-only 0.81 and audio-only 0.62. Audio-only is notably
  weak on genres such as reggae, where the text and image modalities both
  find the right songs.
- **Name lookups** (127 artist and 88 album queries): fused is about as good
  as the best single modality (R-precision 0.84 and 0.77); audio cannot look
  up names at all, as expected.
- **Weighted fusion and query routing do not beat plain equal-weight
  concatenation** (weighting: +0.03, interval [−0.02, +0.08]; routing is
  worse on genre queries), so the baseline stays the default.
- **Query latency** about 7 seconds for embedding, plus the Gemini call.

**Groundedness** (14 test queries, answers from Gemini with the retrieved
context and graph facts): an LLM judge rated 75 of 75 claims in the answers
as supported by the context. Two deterministic checks found no answer that
names a catalogue song absent from the context and none that states a
number or proper name absent from it (numbers match spelled-out forms, so
"41" is supported by "forty-one"). The judge is the same vendor's model and
rated every claim supported, so a 20-claim sample was audited against the
retrieved context (`scripts/judge_audit.py`): the judge agreed on all 20,
with none too lenient. The sample labels were assigned by an LLM reading
each claim against its context, not by a human, and since the judge rated
no claim unsupported the audit can show leniency but not over-strictness.
Both deterministic checks were corrected after a first run
flagged many false alarms (graph facts list other songs that are
legitimately in the context; markdown labels looked like names), and the
checks have unit tests for both directions.

## What's in it

- **Knowledge graph** (networkx): Song, Artist, Album and Genre nodes with
  PERFORMED_BY, BELONGS_TO_ALBUM and HAS_GENRE edges from curated metadata,
  plus SIMILAR_TO edges linking each song to its nearest neighbours in the
  fused embedding space. A retrieved song's 1-hop neighbourhood (album,
  same-artist songs, genre, similar songs) is added to the LLM's context.
- **Query-by-example**: `scripts/similar_to.py` finds songs that sound like
  a given song (audio similarity), optionally steered by a text modifier
  matched against bios, e.g. *"like Three Little Birds, but about sadness"*.
- **Data validation** before encoding: unusable audio, images and bios are
  flagged and excluded rather than crashing a long run; a per-track record
  says which modalities passed and where each bio came from.
- **Evaluation**: hand-labelled retrieval queries, genre queries and
  artist/album lookups on the larger corpus (all with bootstrap confidence
  intervals), per-modality score breakdowns, neighbour agreement, and a
  groundedness check of generated answers.
- **Experiments, off by default**: weighted fusion and keyword-based query
  routing. Neither beat plain equal-weight concatenation on the small
  hand-labelled set, so the baseline stays the default; the experiments are
  re-run on the larger corpus before any change.
- **Demo UI** (Streamlit): retrieved tracks with per-modality scores, graph
  facts, a picture of the knowledge-graph neighbourhood, and the answer.

## Setup

Python 3.11 (the version the project was developed and tested on).
`requirements.txt` pins every direct dependency; `requirements-lock.txt`
is the full resolved environment for exact reproduction (Windows).

```bash
python -m venv venv
venv\Scripts\activate          # Windows; source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
copy .env.example .env         # then fill in GEMINI_API_KEY from Google AI Studio
```

## Running it

```bash
# data
python -m src.ingestion.build_catalog                      # curated artist list -> data/processed/catalog.csv
python -m src.ingestion.ingest_catalog                     # download audio, art, bios (resumable)
python -m scripts.preprocess                               # validate raw files -> data/processed/track_records.csv
python -m scripts.encode_modality text                     # also: image, audio (incremental, cached)

# using it
python -m scripts.run_pipeline                             # full pipeline, one demo query, prints everything
python -m scripts.answer_query "your question" --top-k 5    # single query, JSON result
python -m scripts.similar_to "three little birds" --modifier "songs about sadness"
streamlit run app/streamlit_app.py                          # interactive demo UI

# evaluation
python -m scripts.run_evaluation                           # hand-labelled queries: P@k, MRR, latency -> results.md
python -m scripts.run_scale_evaluation                     # genre queries on the larger corpus -> results_scale.md
python -m scripts.run_objective_evaluation                 # artist/album lookups -> results_objective.md
python -m scripts.run_neighbor_evaluation                  # do nearest neighbours share genre? -> results_neighbors.md
python -m scripts.run_groundedness                         # answer groundedness (resumable; needs Gemini quota)
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
| Graph | networkx: metadata edges + SIMILAR_TO edges | [src/graph/](src/graph/) |
| Retrieval | query → each model's own text tower → top-k → graph expansion | [src/rag/retriever.py](src/rag/retriever.py) |
| Generation | Gemini 3.6-flash, answer only from retrieved context | [src/rag/llm_client.py](src/rag/llm_client.py), [src/rag/prompt_templates.py](src/rag/prompt_templates.py) |

**Text pipeline:** clean the bio (strip citation markers and headings),
prepend a one-line "Title by Artist, from the album Album." header, then
chunk (overlapping word windows) → tokenize (subword, via MiniLM's own
tokenizer) → encode (transformer self-attention → contextual per-token
vectors) → mean-pool → L2-normalize. The header gives tracks that would
otherwise share an artist-level bio distinct text; it can be switched off
in `src/config.py`.

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

**Fusion and what a score means:** because each modality block is
separately unit-normalized, the fused inner product is the *sum* of three
cosine similarities (text + image + audio), not a single bounded cosine.
The demo splits each hit's score back into its three parts so you can see
which modality drove a match.

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

A query is embedded in a single such subprocess that loads the three
encoders one after another and frees each before loading the next, so
PyTorch is imported once rather than three times: about 7–9 s per query
(down from about 25 s with one process per encoder, and 35–60 s when
Hugging Face's online file checks were also retrying on a flaky
connection). Once all three models are cached locally, the hub library is
told to use the cache only. Peak memory stayed under 900 MB.

The FAISS index and the knowledge graph are saved to `data/cache/` after the
first build and reloaded on later queries, but only while the vector caches
and the validated track list they were built from are unchanged (a
fingerprint is checked, so a re-encode triggers a rebuild). The SIMILAR_TO
edges are computed with a FAISS nearest-neighbour search rather than a full
N×N similarity matrix: 5,000 songs took 0.4 s and about 13 MB of extra
memory (the old method took 1.1 s and 56 MB), and 20,000 songs took 4.5 s.
The search itself is still exact brute force, so at hundreds of thousands
of songs an approximate index would be the next step.

Encoded vectors are cached to disk and only missing tracks are encoded on
a rerun, with a checkpoint every 25 tracks. Text vectors are additionally
tied to a hash of the exact text they came from, so changing the cleaning
rules re-encodes the affected tracks instead of silently keeping stale
vectors.

## Testing

```bash
pytest
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the suite on
every push with only the light dependencies in `requirements-ci.txt`; the
one test that needs the MiniLM model weights is excluded there and runs
locally.

Covers the fusion math, FAISS retrieval ordering, graph queries and
SIMILAR_TO edges, retrieval-to-graph expansion, prompt assembly, the
Gemini client's error handling (with a fake client, so no key or quota is
needed), data validation, query-by-example scoring, bootstrap intervals,
the groundedness checks, label cross-checking, and one real MiniLM
inference call. CLIP and audio encoder tests are intentionally excluded
from the test run, in keeping with the subprocess-isolation design above —
`embed_query()`'s full subprocess pipeline is instead exercised live by
`scripts/run_pipeline.py` and `scripts/answer_query.py`.

## Limitations

- **Genre labels are curated by hand.** Each artist is assigned one genre
  in `data/processed/artist_seeds.json`; iTunes' own genre tags are too
  noisy to use directly. The genre-level evaluation therefore measures
  agreement with those labels, not with an independent ground truth, and
  it says nothing about finer attributes such as mood. Cross-checking the
  labels against iTunes' tags (`scripts/check_genre_labels.py`) gives 91%
  agreement on the tracks whose tag maps to one of the six genres, and
  most disagreements are benign (for example R&B/Soul tags on hip-hop
  artists); the list is in `results_labels.md`.
- **Some bios come from the artist's article, not the song's.** Many songs
  have no Wikipedia article of their own. A bio is only kept if the text
  mentions the song title or the artist, and the per-track record says
  which source each track used. The metadata header keeps those tracks'
  text distinct, but it also puts the artist and album names into every
  track's text, which makes artist and album lookups easier by
  construction; results for those queries should be read with that in mind.
  Measured on the 521-track corpus: the header raises text-only artist
  lookup from 0.78 to 0.83 and album lookup from 0.61 to 0.75, leaves genre
  retrieval unchanged, and does not reduce the shared-bio effect — tracks
  with an artist-article bio still have their own artist's songs as
  nearest neighbours more often (about 47% of the top 5 against 37% for
  tracks with their own song article), which also inflates their
  same-genre agreement. Text-only numbers for song-article tracks are the
  conservative figures.
- **The audio encoder is weak on some genre words.** CLAP finds reggae
  from a text query only about 10% of the time at P@10, where the text and
  image modalities find it reliably, so audio adds noise to some fused
  rankings; equal-weight fusion still matched or beat every single
  modality on genre queries, but a per-genre weighting is not justified by
  the data collected so far.
- **Catalogue identity relies on iTunes metadata.** Tracks are checked
  against the artist iTunes credits and, for classical, against iTunes'
  genre tag; a wrong recording with a plausible tag would not be caught,
  and genuine tracks can be excluded when credited under a band name (for
  example "The Jimi Hendrix Experience").
- **The groundedness judge is an LLM.** Claims in a generated answer are
  checked by a Gemini model, so its verdicts are an estimate and can share
  blind spots with the model that wrote the answer. Two checks are
  objective: whether the answer names a song that was never retrieved, and
  whether it states a number or proper name that is absent from the
  retrieved context. `scripts/judge_audit.py` measures the LLM judge's
  agreement with hand labels on a sample of claims.
- **Query embedding takes several seconds.** Each query loads the three
  encoders one after another in a fresh process to stay within a small
  memory budget, about 7–9 s per query, plus the Gemini call. Retrieval
  itself (the FAISS search) takes well under a millisecond.
- **Evaluation sets are small or name-based.** The hand-labelled set has 14
  queries over 18 tracks, and the genre set has 18 queries (three
  phrasings of each of six genres), so differences of a few hundredths in
  precision@k are within noise; reported tables carry bootstrap confidence
  intervals, and weights are validated leaving a whole genre out. The
  artist/album lookups have many queries but mostly test whether a name
  connects to its songs, not mood or sound.
- **Free-tier limits.** The Gemini free tier allows only about 20 requests
  per day per model, so groundedness runs resume across days.

## Data and licensing

The code is released under the MIT license (see `LICENSE`). **The data is
not covered by that license and is not included in the repository.**

- **Audio:** 30-second previews downloaded from the iTunes Search API. They
  are copyrighted recordings, used here for research and education only, and
  must not be redistributed or hosted publicly.
- **Album art:** from the Cover Art Archive and MusicBrainz, with iTunes
  artwork as a fallback. Cover images belong to their rights holders.
- **Bios:** introductions of Wikipedia articles, which are available under
  CC BY-SA 4.0. Showing them to other people requires attribution and a link
  to the source article, and derived text must carry the same license.
- **Services:** use of the iTunes, MusicBrainz, Cover Art Archive and Gemini
  APIs is subject to each provider's own terms, including their rate and
  quota limits.

`data/raw/` and the encoded caches are gitignored; only the artist list and
the catalogue (track ids, titles and source links) are tracked, so the corpus can be rebuilt
by running the ingestion scripts. A public or commercial deployment would
need licensed audio and artwork, or would have to show metadata only.

## Scope

No fine-tuning of any encoder or the LLM — every model runs via
pretrained inference only. Free-tier APIs and open-source models
throughout, zero budget. Fusion is baseline normalize-and-concatenate;
weighted fusion and query routing were tried as experiments and are kept
off unless they clearly beat the baseline on the larger corpus.
