# Query routing vs equal weights

## A. Hand-labelled queries (18 seed tracks)

14 queries, P@3. Router changed the weights on 9.

| | mean P@3 |
|---|---|
| equal weights | 0.595 |
| routed | 0.595 |

| Query | weights (text, image, audio) | equal | routed |
|---|---|---|---|
| reggae music with a laid-back island rhythm | (0.2, 0.2, 0.6) | 1.00 | 0.67 |
| classical orchestral music | (0.2, 0.2, 0.6) | 1.00 | 1.00 |
| rap with lyrics about struggle and ambition | (0.6, 0.2, 0.2) | 1.00 | 1.00 |
| guitar-driven rock | (0.2, 0.2, 0.6) | 0.67 | 0.67 |
| an uplifting, optimistic song about hope | (0.6, 0.2, 0.2) | 0.33 | 0.67 |
| songs by Bob Marley | (0.6, 0.2, 0.2) | 0.67 | 0.67 |
| calm instrumental music with no vocals | (0.2, 0.2, 0.6) | 0.33 | 0.33 |
| music by a composer from the eighteenth or nineteenth century | (0.6, 0.2, 0.2) | 0.67 | 0.67 |
| rock song with operatic vocals and multiple sections | (0.2, 0.2, 0.6) | 0.33 | 0.33 |

## B. Genre queries (521 tracks)

18 queries, P@10. Router changed the weights on 9.

| | mean P@10 |
|---|---|
| equal weights | 0.906 |
| routed | 0.828 |

| Query | weights (text, image, audio) | equal | routed |
|---|---|---|---|
| loud guitar-driven rock song | (0.2, 0.2, 0.6) | 0.90 | 1.00 |
| danceable pop music | (0.2, 0.2, 0.6) | 0.50 | 0.30 |
| rap song with strong beats and rhymes | (0.2, 0.2, 0.6) | 1.00 | 0.70 |
| smooth jazz with saxophone or trumpet | (0.2, 0.2, 0.6) | 1.00 | 1.00 |
| classical orchestral music | (0.2, 0.2, 0.6) | 1.00 | 1.00 |
| a symphony or concerto by a classical composer | (0.6, 0.2, 0.2) | 1.00 | 1.00 |
| baroque or romantic era classical piece | (0.6, 0.2, 0.2) | 1.00 | 1.00 |
| reggae music with an offbeat rhythm | (0.2, 0.2, 0.6) | 1.00 | 0.40 |
| laid-back reggae groove | (0.2, 0.2, 0.6) | 0.90 | 0.50 |

Reading it: enable routing only if it is clearly higher on both sets. Differences of a few hundredths on set A are within noise.
