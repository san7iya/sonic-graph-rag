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
    for name in ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{n}" for n in range(2, 10)]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm_client._get_client()


@pytest.fixture
def two_keys(monkeypatch):
    """Two configured keys, each mapped to its own fake client."""
    for name in [f"GEMINI_API_KEY_{n}" for n in range(3, 10)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-two")
    monkeypatch.setattr(llm_client, "_clients", {})
    monkeypatch.setattr(llm_client, "_exhausted", set())
    monkeypatch.setattr(llm_client, "_key_index", 0)
    fakes = {}

    def install(outcome_one, outcome_two):
        fakes["key-one"], fakes["key-two"] = FakeClient(outcome_one), FakeClient(outcome_two)
        monkeypatch.setattr(llm_client, "_make_client", lambda key: fakes[key])
        return fakes
    return install


def quota_error():
    return genai_errors.ClientError(429, {"error": {"message": "quota"}})


def test_rotates_to_second_key_on_quota_error(two_keys):
    fakes = two_keys(quota_error(), FakeResponse())
    assert llm_client.generate_answer("hi") == "an answer"
    assert len(fakes["key-one"].models.calls) == 1 and len(fakes["key-two"].models.calls) == 1
    # the exhausted key is not tried again on the next call
    llm_client.generate_answer("again")
    assert len(fakes["key-one"].models.calls) == 1 and len(fakes["key-two"].models.calls) == 2


def test_quota_error_is_raised_when_every_key_is_exhausted(two_keys):
    two_keys(quota_error(), quota_error())
    with pytest.raises(genai_errors.ClientError) as info:
        llm_client.generate_answer("hi")
    assert info.value.code == 429
    with pytest.raises(genai_errors.ClientError):          # still exhausted: fails fast without new calls
        llm_client.generate_answer("hi again")


def test_exhaustion_is_tracked_per_model(two_keys):
    fakes = two_keys(quota_error(), quota_error())
    with pytest.raises(genai_errors.ClientError):
        llm_client.generate_answer("hi", model="model-a")
    fakes["key-one"].models.outcome = fakes["key-two"].models.outcome = FakeResponse()
    assert llm_client.generate_answer("hi", model="model-b") == "an answer"   # other model still has quota


def test_non_quota_client_error_does_not_rotate(two_keys):
    fakes = two_keys(genai_errors.ClientError(400, {"error": {"message": "bad request"}}), FakeResponse())
    with pytest.raises(genai_errors.ClientError):
        llm_client.generate_answer("hi")
    assert len(fakes["key-two"].models.calls) == 0


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
