from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from model_gateway.main import app
import model_gateway.routes as routes_module


def test_chat_route_resolves_role_and_calls_deepinfra_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {"input": 3, "output": 2}, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.json() == {"content": "the answer", "reasoning": None}
    fake_adapter.chat.assert_awaited_once_with("big-model", [{"role": "user", "content": "hi"}], None, None)


def test_chat_route_forwards_response_format_to_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("{}", {}, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "small-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    client.post("/v1/chat", json={
        "role": "slm", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })

    fake_adapter.chat.assert_awaited_once_with(
        "small-model", [{"role": "user", "content": "hi"}], {"type": "json_object"}, None,
    )


def test_chat_route_surfaces_reasoning_when_present(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, "thinking it through...")
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.json() == {"content": "the answer", "reasoning": "thinking it through..."}


def test_chat_route_rejects_unknown_role(monkeypatch):
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})
    client = TestClient(app)

    response = client.post("/v1/chat", json={"role": "nonexistent", "messages": []})

    assert response.status_code == 400


def test_embed_route_resolves_query_embed_to_voyage_provider(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.embed.return_value = ([0.1, 0.2], {"input": 1})
    captured_providers = []

    def fake_get_adapter(provider):
        captured_providers.append(provider)
        return fake_adapter

    monkeypatch.setattr(routes_module, "get_adapter", fake_get_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"query_embed": "voyage-4-large"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"query_embed": "voyage"})

    client = TestClient(app)
    response = client.post("/v1/embed", json={"role": "query_embed", "text": "hello"})

    assert response.json() == {"embedding": [0.1, 0.2]}
    assert captured_providers == ["voyage"]  # embed route must resolve to the voyage provider, not deepinfra


def test_rerank_route(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"reranker": "rerank-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"reranker": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/rerank", json={"role": "reranker", "query": "q", "documents": ["a", "b"]})

    assert response.json() == {"scores": [0.9, 0.1]}


def test_rerank_route_returns_empty_scores_without_calling_adapter_when_documents_empty(monkeypatch):
    """DeepInfra's rerank endpoint 422s on an empty documents list ("the number of
    queries and documents must be the same"), which used to surface as an unhandled 500.
    Short-circuit before the adapter is ever called."""
    fake_adapter = AsyncMock()
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"reranker": "rerank-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"reranker": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/rerank", json={"role": "reranker", "query": "q", "documents": []})

    assert response.json() == {"scores": []}
    fake_adapter.rerank.assert_not_called()


def test_get_model_route_returns_model_for_known_role(monkeypatch):
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "meta-llama/Meta-Llama-3.1-8B-Instruct"})
    client = TestClient(app)

    response = client.get("/v1/models/slm")

    assert response.status_code == 200
    assert response.json() == {"role": "slm", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"}


def test_get_model_route_rejects_unknown_role(monkeypatch):
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "some-model"})
    client = TestClient(app)

    response = client.get("/v1/models/nonexistent")

    assert response.status_code == 404


def test_chat_route_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "default-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    response = client.post(
        "/v1/chat",
        json={"role": "slm", "messages": [{"role": "user", "content": "hi"}], "model": "candidate-model"},
    )

    assert response.status_code == 200
    fake_adapter.chat.assert_awaited_once_with("candidate-model", [{"role": "user", "content": "hi"}], None, None)


def test_chat_route_falls_back_to_role_default_when_model_omitted(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "default-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    client.post("/v1/chat", json={"role": "slm", "messages": [{"role": "user", "content": "hi"}]})

    fake_adapter.chat.assert_awaited_once_with("default-model", [{"role": "user", "content": "hi"}], None, None)


def test_rerank_route_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"reranker": "default-reranker"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"reranker": "deepinfra"})

    client = TestClient(app)
    client.post(
        "/v1/rerank",
        json={"role": "reranker", "query": "q", "documents": ["a", "b"], "model": "candidate-reranker"},
    )

    fake_adapter.rerank.assert_awaited_once_with("candidate-reranker", "q", ["a", "b"])
