"""Gemini wrapper error handling and prompt assembly, tested with a fake
client so no API key, network access or quota is needed.
"""

import httpx
import pytest
from google.genai import errors as genai_errors

from src.rag import llm_client
from src.rag.prompt_templates import RAG_PROMPT_TEMPLATE, build_rag_prompt


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append((model, contents))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome):
        self.models = FakeModels(outcome)


class FakeResponse:
    text = "an answer"


@pytest.fixture
def use_client(monkeypatch):
    def _install(outcome):
        client = FakeClient(outcome)
        monkeypatch.setattr(llm_client, "_get_client", lambda: client)
        return client
    return _install


def test_generate_answer_returns_text_and_uses_default_model(use_client):
    client = use_client(FakeResponse())
    assert llm_client.generate_answer("hi") == "an answer"
    assert client.models.calls == [(llm_client.config.LLM_MODEL_NAME, "hi")]


def test_generate_answer_model_override(use_client):
    client = use_client(FakeResponse())
    llm_client.generate_answer("hi", model="other-model")
    assert client.models.calls[0][0] == "other-model"


def test_server_error_becomes_runtime_error(use_client):
    use_client(genai_errors.ServerError(503, {"error": {"message": "overloaded"}}))
    with pytest.raises(RuntimeError, match="server error"):
        llm_client.generate_answer("hi")


def test_network_error_becomes_runtime_error(use_client):
    use_client(httpx.RemoteProtocolError("Server disconnected"))
    with pytest.raises(RuntimeError, match="Network error"):
        llm_client.generate_answer("hi")


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm_client._get_client()


def test_build_rag_prompt_separates_context_from_instructions():
    hits = [{"title": "Song A", "artist": "Artist A", "genre": "rock", "bio": "A bio.", "similarity": 0.91}]
    prompt = build_rag_prompt("What is good?", hits)

    start = prompt.index("--- RETRIEVED CONTEXT ---")
    end = prompt.index("--- END RETRIEVED CONTEXT ---")
    context = prompt[start:end]
    assert '[1] "Song A" by Artist A' in context and "A bio." in context
    assert "0.910" in context
    assert "ONLY the retrieved context" in prompt[:start]       # instruction before the context
    assert "Question: What is good?" in prompt[end:]            # question after the context


def test_build_rag_prompt_numbers_every_hit_in_order():
    hits = [{"title": f"T{i}", "artist": "A", "genre": "g", "bio": "b", "similarity": 0.5} for i in range(3)]
    prompt = build_rag_prompt("q", hits)
    assert prompt.index('[1] "T0"') < prompt.index('[2] "T1"') < prompt.index('[3] "T2"')
    assert RAG_PROMPT_TEMPLATE.count("{context}") == 1
