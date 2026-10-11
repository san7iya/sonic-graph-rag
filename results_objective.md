# Objective (metadata-derived) retrieval evaluation

521 tracks. Relevance = the track's artist (or artist+album) matches the query; metric = R-precision with 95% bootstrap intervals over queries. These are name-lookup queries: they show how well each modality connects a name to its songs, not mood or sound.

## Artist queries (n=127)

| System | R-precision [95% CI] | MRR [95% CI] |
|---|---|---|
| fused (equal weights) | 0.838 [0.805, 0.871] | 0.989 [0.972, 1.000] |
| text only | 0.834 [0.794, 0.872] | 0.983 [0.960, 1.000] |
| image only | 0.775 [0.728, 0.819] | 0.947 [0.911, 0.977] |
| audio only | 0.025 [0.012, 0.041] | 0.084 [0.056, 0.118] |

Fused minus text-only (paired): +0.004 [-0.029, +0.040]. Interval includes 0: no reliable difference.

## Album queries (n=88)

| System | R-precision [95% CI] | MRR [95% CI] |
|---|---|---|
| fused (equal weights) | 0.766 [0.705, 0.822] | 0.903 [0.856, 0.943] |
| text only | 0.746 [0.689, 0.802] | 0.889 [0.842, 0.934] |
| image only | 0.831 [0.759, 0.894] | 0.852 [0.790, 0.908] |
| audio only | 0.009 [0.000, 0.023] | 0.046 [0.021, 0.081] |

Fused minus text-only (paired): +0.020 [-0.036, +0.074]. Interval includes 0: no reliable difference.

