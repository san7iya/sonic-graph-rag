"""Thin wrapper around the Gemini API (google-genai SDK) for generation.

Used only for the final answer generation step — never for embeddings.
Requires GEMINI_API_KEY in the environment (see .env.example); this file
does not hardcode or invent one.
"""

import os

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src import config

load_dotenv()

_client: genai.Client | None = None

# The SDK's default retry policy (5 attempts, backoff up to 60s max delay)
# means a run of transient 503s can silently burn 1-3 minutes before it
# finally raises -- observed live against gemini-3.6-flash under "high
# demand" load. Trimmed so a real outage fails in ~15s instead, since this
# app surfaces the failure to a Streamlit spinner the user is watching.
_RETRY_OPTIONS = genai_types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=5.0)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
                "with a key from Google AI Studio."
            )
        http_options = genai_types.HttpOptions(retry_options=_RETRY_OPTIONS, timeout=20_000)
        _client = genai.Client(api_key=api_key, http_options=http_options)
    return _client


def generate_answer(prompt: str, model: str | None = None) -> str:
    """Send one prompt to Gemini and return the plain-text response.

    `model` overrides config.LLM_MODEL_NAME (used by the groundedness
    judge, which runs on a different model).

    Raises RuntimeError with a clear message on a transient server-side
    failure (e.g. 503 "high demand") instead of letting the SDK's raw
    ServerError propagate -- that's what the Streamlit caller's error box
    displays if this subprocess fails.
    """
    client = _get_client()
    try:
        response = client.models.generate_content(model=model or config.LLM_MODEL_NAME, contents=prompt)
    except genai_errors.ServerError as e:
        raise RuntimeError(
            f"Gemini ({config.LLM_MODEL_NAME}) returned a server error, likely transient "
            f"high-demand load on the free tier: {e}. Retry the query."
        ) from e
    except httpx.TransportError as e:
        raise RuntimeError(
            f"Network error talking to Gemini ({type(e).__name__}: {e}). Usually transient; retry."
        ) from e
    return response.text
