"""Prompt template for grounded RAG generation.

The template exists as its own module (rather than an inline f-string in
llm_client.py) so the exact wording — the part that enforces "answer only
from context" — is easy to point to and defend in review as one specific,
inspectable artifact.
"""

RAG_PROMPT_TEMPLATE = """You are answering a question about songs using ONLY the retrieved context below. \
Do not use any outside knowledge, even if you recognize the songs or artists. \
If the context does not contain enough information to answer, say so explicitly instead of guessing.

--- RETRIEVED CONTEXT ---
{context}
--- END RETRIEVED CONTEXT ---

Question: {query}

Answer using only the retrieved context above. When you state a fact, make it traceable to one of the retrieved tracks."""


def build_context_block(retrieved_tracks: list[dict]) -> str:
    """Format retrieved track records into the numbered context block the
    template above refers to, so each claim in the LLM's answer can be
    traced back to a specific numbered entry.
    """
    blocks = []
    for i, track in enumerate(retrieved_tracks, start=1):
        block = (
            f"[{i}] \"{track['title']}\" by {track['artist']} (genre: {track['genre']}, "
            f"similarity: {track['similarity']:.3f})\n"
            f"Bio: {track['bio']}"
        )
        graph = track.get("graph")
        if graph:
            block += (
                f"\nGraph facts: album \"{graph['album']}\"; "
                f"other songs by this artist: {', '.join(graph['same_artist_tracks']) or 'none'}; "
                f"other songs in this genre: {', '.join(graph['same_genre_tracks']) or 'none'}"
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_rag_prompt(query: str, retrieved_tracks: list[dict]) -> str:
    """Assemble the final prompt sent to the LLM for one query."""
    context = build_context_block(retrieved_tracks)
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)
