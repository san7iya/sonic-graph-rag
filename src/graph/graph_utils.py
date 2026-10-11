"""Query helpers over the music knowledge graph."""

from __future__ import annotations

import networkx as nx

from src.graph import schema


def _songs_pointing_at(graph: nx.DiGraph, target: str) -> list[str]:
    """Return track_ids of every Song with an edge into `target`.

    Songs are the *sources* of all three relations, so reverse lookups
    ("who points at this artist/genre/album?") are just predecessors.
    Returns [] for an unknown node rather than raising, since "no such
    artist" and "artist with no songs" both mean "nothing to show".
    """
    if target not in graph:
        return []
    return sorted(graph.nodes[p]["track_id"] for p in graph.predecessors(target))


def songs_by_artist(graph: nx.DiGraph, artist: str) -> list[str]:
    """All track_ids performed by `artist` (Milestone 2 check #1)."""
    return _songs_pointing_at(graph, schema.node_id(schema.ARTIST, artist))


def songs_in_genre(graph: nx.DiGraph, genre: str) -> list[str]:
    """All track_ids in `genre` (Milestone 2 check #2)."""
    return _songs_pointing_at(graph, schema.node_id(schema.GENRE, genre))


def song_neighborhood(graph: nx.DiGraph, track_id: str) -> dict:
    """1-hop expansion of one song: its artist, album, genre, plus the
    other songs sharing its artist or genre, and its embedding-similar
    songs (SIMILAR_TO edges, most similar first, if the graph has them).

    Why this exists: vector retrieval only knows "similar embedding";
    the graph adds explicit relational facts (same artist, same album)
    that no embedding is guaranteed to encode. Handing these to the LLM
    as extra context lets it answer relational questions from evidence.
    """
    song = schema.node_id(schema.SONG, track_id)
    neighborhood = {"artist": None, "album": None, "genre": None,
                    "same_artist_tracks": [], "same_genre_tracks": [], "similar_tracks": []}
    if song not in graph:
        return neighborhood

    similar = []
    for _, target, data in graph.out_edges(song, data=True):
        if data["relation"] == schema.SIMILAR_TO:   # target is a Song node (has no "name")
            similar.append((data["weight"], graph.nodes[target]["track_id"]))
            continue
        name = graph.nodes[target]["name"]
        if data["relation"] == schema.PERFORMED_BY:
            neighborhood["artist"] = name
            neighborhood["same_artist_tracks"] = [
                t for t in _songs_pointing_at(graph, target) if t != track_id]
        elif data["relation"] == schema.BELONGS_TO_ALBUM:
            neighborhood["album"] = name
        elif data["relation"] == schema.HAS_GENRE:
            neighborhood["genre"] = name
            neighborhood["same_genre_tracks"] = [
                t for t in _songs_pointing_at(graph, target) if t != track_id]
    neighborhood["similar_tracks"] = [t for _, t in sorted(similar, reverse=True)]
    return neighborhood
