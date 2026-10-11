"""Thin wrapper around the Gemini API (google-genai SDK) for generation.

Used only for the final answer generation step — never for embeddings.
Reads GEMINI_API_KEY (and optionally GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...)
from the environment (see .env.example); this file does not hardcode or
invent one.

Several keys are rotated on quota exhaustion: the free tier's daily request
limit is counted per Google project and per model, so when one key's
project is out of requests for a model (HTTP 429), the next key is tried.
This only helps if the keys belong to different projects.
"""

import os

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src import config

load_dotenv()

_clients: dict[str, genai.Client] = {}
_key_index = 0
# (key position, model) pairs that returned 429 in this process; quota is per
# project *and* per model, so a key exhausted for one model may still serve another.
_exhausted: set[tuple[int, str]] = set()

# The SDK's default retry policy (5 attempts, backoff up to 60s max delay)
# means a run of transient 503s can silently burn 1-3 minutes before it
# finally raises -- observed live against gemini-3.6-flash under "high
# demand" load. Trimmed so a real outage fails in ~15s instead, since this
# app surfaces the failure to a Streamlit spinner the user is watching.
_RETRY_OPTIONS = genai_types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=5.0)


def _api_keys() -> list[str]:
    """All configured keys in order: GEMINI_API_KEY, GEMINI_API_KEY_2, _3, ..."""
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{n}" for n in range(2, 10)]
    keys = [os.environ[n] for n in names if os.environ.get(n)]
    if not keys:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "with a key from Google AI Studio."
        )
    return keys


def _make_client(api_key: str) -> genai.Client:
    http_options = genai_types.HttpOptions(retry_options=_RETRY_OPTIONS, timeout=20_000)
    return genai.Client(api_key=api_key, http_options=http_options)


def _get_client() -> genai.Client:
    """Client for the key currently in use (created once per key)."""
    keys = _api_keys()
    key = keys[min(_key_index, len(keys) - 1)]
    if key not in _clients:
        _clients[key] = _make_client(key)
    return _clients[key]


def _rotate_key(model: str) -> bool:
    """Mark the current key exhausted for `model` and switch to a key that is
    not. Returns False when every key is exhausted for that model.
    """
    global _key_index
    keys = _api_keys()
    _exhausted.add((_key_index, model))
    for i in range(len(keys)):
        if (i, model) not in _exhausted:
            _key_index = i
            return True
    return False


def generate_answer(prompt: str, model: str | None = None) -> str:
    """Send one prompt to Gemini and return the plain-text response.

    `model` overrides config.LLM_MODEL_NAME (used by the groundedness
    judge, which runs on a different model).

    On a 429 (daily quota) the next configured key is tried; the original
    429 is re-raised only when no key has quota left for this model. A
    transient server-side failure (e.g. 503 "high demand") is raised as
    RuntimeError with a clear message instead of the SDK's raw
    ServerError -- that's what the Streamlit caller's error box displays
    if this subprocess fails.
    """
    model_name = model or config.LLM_MODEL_NAME
    while True:
        # A key already known to be exhausted for this model (from an earlier
        # call in this process) is skipped up front instead of re-tried.
        if (_key_index, model_name) in _exhausted and not _rotate_key(model_name):
            raise genai_errors.ClientError(429, {"error": {"message": "All API keys are out of quota for "
                                                                      f"{model_name}."}})
        client = _get_client()
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except genai_errors.ClientError as e:
            if e.code == 429 and _rotate_key(model_name):
                print(f"[quota] key {_key_index + 1} now in use for {model_name}")
                continue
            raise
        except genai_errors.ServerError as e:
            raise RuntimeError(
                f"Gemini ({model_name}) returned a server error, likely transient "
                f"high-demand load on the free tier: {e}. Retry the query."
            ) from e
        except httpx.TransportError as e:
            raise RuntimeError(
                f"Network error talking to Gemini ({type(e).__name__}: {e}). Usually transient; retry."
            ) from e
