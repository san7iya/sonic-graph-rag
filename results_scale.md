# Scaled-corpus evaluation (genre labels)

Corpus: 521 tracks. 18 genre queries (3 phrasings x 6 genres); a hit is relevant if the track's curated genre equals the query's genre. Metrics at k=10. Random-retrieval P@10 would be about 0.17.

| System | mean P@10 [95% CI] | MRR [95% CI] |
|---|---|---|
| fused (equal weights) | 0.906 [0.833, 0.967] | 1.000 [1.000, 1.000] |
| text only | 0.811 [0.694, 0.917] | 0.944 [0.861, 1.000] |
| image only | 0.894 [0.817, 0.956] | 0.900 [0.783, 1.000] |
| audio only | 0.622 [0.456, 0.778] | 0.756 [0.583, 0.914] |

## Per-genre P@10

| Genre | fused (equal weights) | text only | image only | audio only |
|---|---|---|---|---|
| rock | 0.900 | 0.533 | 0.933 | 0.867 |
| pop | 0.600 | 0.433 | 0.700 | 0.433 |
| hiphop | 1.000 | 0.900 | 0.867 | 0.567 |
| jazz | 0.967 | 1.000 | 0.900 | 0.867 |
| classical | 1.000 | 1.000 | 0.967 | 0.900 |
| reggae | 0.967 | 1.000 | 1.000 | 0.100 |

## Weighted fusion (weights text, image, audio)

- Best grid point in-sample: (0.1, 0.8, 0.1) -> 0.961 (optimistic)
- Leave-one-genre-out estimate: 0.933
- Leave-one-genre-out minus equal weights (paired): +0.028 [-0.017, +0.078] (interval includes 0: not a reliable gain)
- Equal-weight baseline: 0.906

## Per-modality cosine scale (mean / std)

| Modality | mean | std |
|---|---|---|
| text | 0.186 | 0.110 |
| image | 0.210 | 0.025 |
| audio | 0.307 | 0.137 |
