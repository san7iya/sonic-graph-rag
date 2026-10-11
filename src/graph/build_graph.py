"""Build the music knowledge graph from structured track metadata.

Structured fields (artist, album, genre) are used directly as graph
structure, not re-derived from text — per the data-source spec, they are
curated metadata, so there is nothing for an encoder to guess at.
"""

from __future__ import annotations

import networkx as nx

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
