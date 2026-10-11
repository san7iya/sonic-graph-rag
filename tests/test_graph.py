"""Milestone 2 'done' check: the graph answers 'songs by artist X' and
'songs in genre Y' correctly, on a tiny hand-checkable dataset.
"""

from src.graph.build_graph import build_graph
from src.graph.graph_utils import song_neighborhood, songs_by_artist, songs_in_genre

ROWS = [
    {"track_id": "m1", "title": "No Woman No Cry", "artist": "Bob Marley", "genre": "reggae", "album": "Natty Dread"},
    {"track_id": "m2", "title": "Three Little Birds", "artist": "Bob Marley", "genre": "reggae", "album": "Exodus"},
    {"track_id": "t1", "title": "Pressure Drop", "artist": "Toots", "genre": "reggae", "album": "Pressure Drop"},
    {"track_id": "q1", "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "rock", "album": "Opera"},
]


def test_songs_by_artist():
    graph = build_graph(ROWS)
    assert songs_by_artist(graph, "Bob Marley") == ["m1", "m2"]
    assert songs_by_artist(graph, "Queen") == ["q1"]
    assert songs_by_artist(graph, "Nobody") == []


def test_songs_in_genre():
    graph = build_graph(ROWS)
    assert songs_in_genre(graph, "reggae") == ["m1", "m2", "t1"]
    assert songs_in_genre(graph, "rock") == ["q1"]
    assert songs_in_genre(graph, "jazz") == []


def test_song_neighborhood():
    graph = build_graph(ROWS)
    hood = song_neighborhood(graph, "m1")
    assert hood["artist"] == "Bob Marley"
    assert hood["album"] == "Natty Dread"
    assert hood["genre"] == "reggae"
    assert hood["same_artist_tracks"] == ["m2"]
    assert hood["same_genre_tracks"] == ["m2", "t1"]


def test_artist_and_album_with_same_name_stay_separate_nodes():
    rows = [{"track_id": "x", "title": "S", "artist": "Led Zeppelin", "genre": "rock", "album": "Led Zeppelin"}]
    graph = build_graph(rows)
    assert graph.number_of_nodes() == 4  # song, artist, album, genre


def test_context_block_includes_graph_facts():
    from src.rag.prompt_templates import build_context_block

    track = {"title": "T", "artist": "A", "genre": "g", "bio": "b", "similarity": 0.5,
             "graph": {"album": "Alb", "same_artist_tracks": ["Other"], "same_genre_tracks": []}}
    block = build_context_block([track])
    assert 'album "Alb"' in block and "Other" in block and "none" in block


def test_similarity_edges_link_nearest_songs_and_show_in_neighborhood():
    import numpy as np
    from src.graph.build_graph import add_similarity_edges
    from src.graph import schema

    graph = build_graph(ROWS)
    ids = ["m1", "m2", "t1", "q1"]
    vecs = np.array([[1, 0], [0.99, 0.1], [0.9, 0.3], [0, 1]], dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    added = add_similarity_edges(graph, ids, vecs, k=2)

    assert added == 8
    hood = song_neighborhood(graph, "m1")
    assert hood["similar_tracks"] == ["m2", "t1"]            # most similar first
    assert graph["Song:m1"]["Song:m2"]["relation"] == schema.SIMILAR_TO
    # metadata queries are unaffected by the extra Song->Song edges
    assert songs_by_artist(graph, "Bob Marley") == ["m1", "m2"]
    assert hood["artist"] == "Bob Marley"


def test_context_block_caps_genre_list_and_lists_similar():
    from src.rag.prompt_templates import build_context_block

    track = {"title": "T", "artist": "A", "genre": "g", "bio": "b", "similarity": 0.5,
             "graph": {"album": "Alb", "same_artist_tracks": [], "same_genre_tracks": ["x", "y"],
                       "same_genre_count": 12, "similar_tracks": ["S1", "S2"]}}
    block = build_context_block([track])
    assert "(+10 more)" in block and "S1, S2" in block
