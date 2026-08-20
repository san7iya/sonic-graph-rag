"""Thin wrapper around the Gemini API (google-genai SDK) for generation.

Used only for the final answer generation step — never for embeddings.
Requires GEMINI_API_KEY in the environment (see .env.example); this file
does not hardcode or invent one.
"""

import os

from dotenv import load_dotenv
from google import genai

from src import config

load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "with a key from Google AI Studio."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(prompt: str) -> str:
    """Send one prompt to Gemini and return the plain-text response."""
    client = _get_client()
    response = client.models.generate_content(model=config.LLM_MODEL_NAME, contents=prompt)
    return response.text
