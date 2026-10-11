"""Node and edge type names for the knowledge graph, defined once.

Kept as constants (rather than string literals scattered through
build_graph.py and graph_utils.py) so a typo in a type name fails loudly
as a NameError instead of silently creating a second, disconnected kind
of node.
"""

# Node types (stored in each node's "type" attribute)
SONG = "Song"
ARTIST = "Artist"
ALBUM = "Album"
GENRE = "Genre"

# Edge types (stored in each edge's "relation" attribute). All point
# *from the song to the thing it is attached to*, so "all songs by X" is
# "every predecessor of X along PERFORMED_BY".
PERFORMED_BY = "PERFORMED_BY"
BELONGS_TO_ALBUM = "BELONGS_TO_ALBUM"
HAS_GENRE = "HAS_GENRE"
# Song -> Song edge added from embedding similarity (not from metadata);
# carries a "weight" attribute: the fused inner-product score.
SIMILAR_TO = "SIMILAR_TO"


def node_id(node_type: str, name: str) -> str:
    """Build a graph node key like 'Artist:Queen'.

    The type prefix keeps namespaces apart: an artist and an album can
    share a name (e.g. the album 'Led Zeppelin' vs. the artist 'Led
    Zeppelin') and must still be two different nodes.
    """
    return f"{node_type}:{name}"


def album_node_id(album: str, artist: str) -> str:
    """Albums are keyed by (title, artist), not title alone, because
    different artists can release albums with the same title.
    """
    return node_id(ALBUM, f"{album} — {artist}")
