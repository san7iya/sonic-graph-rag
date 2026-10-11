"""Build the music knowledge graph from structured track metadata.

Structured fields (artist, album, genre) are used directly as graph
structure, not re-derived from text — per the data-source spec, they are
curated metadata, so there is nothing for an encoder to guess at.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.graph import schema


def build_graph(track_rows: list[dict]) -> nx.DiGraph:
    """Turn metadata rows (track_id, title, artist, genre, album) into a
    directed graph: Song -> Artist / Album / Genre.

    A DiGraph (not an undirected Graph) because the relations have a
    direction ("song PERFORMED_BY artist") and the edge-type label is
    what lets queries ask for one relation without the others. Artist,
    Album and Genre nodes are created once and shared via add_node's
    idempotence, which is what makes two songs by the same artist end up
    connected through one Artist node.
    """
    graph = nx.DiGraph()
    for row in track_rows:
        song = schema.node_id(schema.SONG, row["track_id"])
        artist = schema.node_id(schema.ARTIST, row["artist"])
        album = schema.album_node_id(row["album"], row["artist"])
        genre = schema.node_id(schema.GENRE, row["genre"])

        graph.add_node(song, type=schema.SONG, track_id=row["track_id"], title=row["title"])
        graph.add_node(artist, type=schema.ARTIST, name=row["artist"])
        graph.add_node(album, type=schema.ALBUM, name=row["album"])
        graph.add_node(genre, type=schema.GENRE, name=row["genre"])

        graph.add_edge(song, artist, relation=schema.PERFORMED_BY)
        graph.add_edge(song, album, relation=schema.BELONGS_TO_ALBUM)
        graph.add_edge(song, genre, relation=schema.HAS_GENRE)
    return graph


def add_similarity_edges(graph: nx.DiGraph, track_ids: list[str], fused_vectors: np.ndarray, k: int = 3) -> int:
    """Add Song -SIMILAR_TO-> Song edges to each song's k nearest
    neighbours in fused-embedding space; returns the number of edges added.

    Why: the metadata edges only say what a curator wrote down (artist,
    album, genre). SIMILAR_TO edges bring the learned multi-modal
    similarity into the graph itself, so a hop from a retrieved song also
    reaches songs that sound / read / look alike even when they share no
    metadata. The edge weight is the inner product of the fused vectors,
    so the evidence strength travels with the edge. Edges are directed
    (A's top-k need not include B), which matches how kNN behaves.
    """
    sims = fused_vectors @ fused_vectors.T
    np.fill_diagonal(sims, -np.inf)
    k = min(k, len(track_ids) - 1)
    added = 0
    for i, track_id in enumerate(track_ids):
        for j in np.argsort(-sims[i])[:k]:
            graph.add_edge(schema.node_id(schema.SONG, track_id),
                           schema.node_id(schema.SONG, track_ids[j]),
                           relation=schema.SIMILAR_TO, weight=float(sims[i, j]))
            added += 1
    return added


def build_graph_with_similarity(track_rows: list[dict], index, k: int = 3) -> nx.DiGraph:
    """Metadata graph plus SIMILAR_TO edges computed from a FusedVectorIndex.

    Reads the fused vectors back out of the index (so the graph and the
    retrieval index are guaranteed to describe the same vectors), and only
    for tracks that are in both the rows and the index.
    """
    graph = build_graph(track_rows)
    all_vectors = index.index.reconstruct_n(0, index.index.ntotal)
    wanted = {r["track_id"] for r in track_rows}
    keep = [i for i, t in enumerate(index.track_ids) if t in wanted]
    add_similarity_edges(graph, [index.track_ids[i] for i in keep], all_vectors[keep], k)
    return graph
