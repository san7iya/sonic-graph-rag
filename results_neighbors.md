# Neighbour agreement (same-genre@5)

521 tracks with all three modalities encoded. Genre counts: {'classical': 76, 'hiphop': 82, 'jazz': 87, 'pop': 95, 'reggae': 77, 'rock': 104}. For each song, the fraction of its 5 nearest other songs that share its curated genre. Random neighbours would score 0.167.

| Space | same-genre@5 |
|---|---|
| text | 0.767 |
| image | 0.767 |
| audio | 0.759 |
| fused (equal weights) | 0.912 |
| SIMILAR_TO graph edges (top-3, fused) | 0.955 |

## Bio source breakdown (text space and fused)

Tracks whose bio is the artist's article share text with that artist's other tracks, which could inflate same-artist neighbours. Same-genre and same-artist neighbour rates by the query song's bio source:

| Bio source | tracks | text same-genre@5 | text same-artist@5 | fused same-artist@5 |
|---|---|---|---|---|
| artist | 174 | 0.916 | 0.470 | 0.475 |
| seed | 18 | 0.767 | 0.333 | 0.367 |
| song | 329 | 0.689 | 0.366 | 0.431 |
