from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from model_gateway.main import app
import model_gateway.routes as routes_module


def test_chat_route_resolves_role_and_calls_deepinfra_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {"input": 3, "output": 2})
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.json() == {"content": "the answer"}
    fake_adapter.chat.assert_awaited_once_with("big-model", [{"role": "user", "content": "hi"}])


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
