from unittest.mock import AsyncMock

import pytest

import model_gateway.client as client_module
from model_gateway.adapters.deepinfra import DeepInfraAdapter
from model_gateway.adapters.local import LocalAdapter
from model_gateway.adapters.local_rerank import LocalRerankAdapter
from model_gateway.adapters.voyage import VoyageAdapter
from model_gateway.client import GatewayClient, UnknownRoleError


def _client(monkeypatch, fake_adapter, model_map, provider_map, reasoning_map=None, trace_enabled=False):
    monkeypatch.setattr(client_module, "_get_adapter", lambda provider, settings: fake_adapter)
    gateway = GatewayClient(trace_enabled=trace_enabled)
    gateway._role_model_map = model_map
    gateway._role_provider_map = provider_map
    gateway._role_reasoning_map = reasoning_map or {}
    return gateway


def test_get_adapter_resolves_provider_to_matching_adapter_type():
    settings = client_module.get_gateway_settings()

    assert isinstance(client_module._get_adapter("local_rerank", settings), LocalRerankAdapter)
    assert isinstance(client_module._get_adapter("local", settings), LocalAdapter)
    assert isinstance(client_module._get_adapter("voyage", settings), VoyageAdapter)
    assert isinstance(client_module._get_adapter("deepinfra", settings), DeepInfraAdapter)


@pytest.mark.asyncio
async def test_chat_resolves_role_and_calls_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {"input": 3, "output": 2}, None)
    gateway = _client(monkeypatch, fake_adapter, {"synthesis": "big-model"}, {"synthesis": "deepinfra"})

    result = await gateway.chat(role="synthesis", messages=[{"role": "user", "content": "hi"}])

    assert result == "the answer"
    fake_adapter.chat.assert_awaited_once_with(
        "big-model", [{"role": "user", "content": "hi"}], None, None, role="synthesis", reasoning_enabled=True,
    )


@pytest.mark.asyncio
async def test_chat_forwards_response_format_to_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("{}", {}, None)
    gateway = _client(monkeypatch, fake_adapter, {"slm": "small-model"}, {"slm": "deepinfra"})

    await gateway.chat(
        role="slm", messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    fake_adapter.chat.assert_awaited_once_with(
        "small-model", [{"role": "user", "content": "hi"}], {"type": "json_object"}, None,
        role="slm", reasoning_enabled=True,
    )


@pytest.mark.asyncio
async def test_chat_with_reasoning_surfaces_reasoning_when_present(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, "thinking it through...")
    gateway = _client(monkeypatch, fake_adapter, {"synthesis": "big-model"}, {"synthesis": "deepinfra"})

    content, reasoning = await gateway.chat_with_reasoning(
        role="synthesis", messages=[{"role": "user", "content": "hi"}],
    )

    assert content == "the answer"
    assert reasoning == "thinking it through..."


@pytest.mark.asyncio
async def test_chat_rejects_unknown_role(monkeypatch):
    gateway = _client(monkeypatch, AsyncMock(), {"synthesis": "big-model"}, {"synthesis": "deepinfra"})

    with pytest.raises(UnknownRoleError):
        await gateway.chat(role="nonexistent", messages=[])


@pytest.mark.asyncio
async def test_embed_resolves_query_embed_to_voyage_provider(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.embed.return_value = ([0.1, 0.2], {"input": 1})
    gateway = _client(monkeypatch, fake_adapter, {"query_embed": "voyage-4-large"}, {"query_embed": "voyage"})

    result = await gateway.embed(role="query_embed", text="hello")

    assert result == [0.1, 0.2]
    fake_adapter.embed.assert_awaited_once_with("voyage-4-large", "hello")


@pytest.mark.asyncio
async def test_rerank_returns_scores(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    gateway = _client(monkeypatch, fake_adapter, {"reranker": "rerank-model"}, {"reranker": "deepinfra"})

    result = await gateway.rerank(role="reranker", query="q", documents=["a", "b"])

    assert result == [0.9, 0.1]


@pytest.mark.asyncio
async def test_rerank_forwards_instruction_to_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    gateway = _client(monkeypatch, fake_adapter, {"reranker": "rerank-model"}, {"reranker": "deepinfra"})

    await gateway.rerank(role="reranker", query="q", documents=["a", "b"], instruction="rank by X")

    fake_adapter.rerank.assert_awaited_once_with("rerank-model", "q", ["a", "b"], instruction="rank by X")


@pytest.mark.asyncio
async def test_rerank_returns_empty_scores_without_calling_adapter_when_documents_empty(monkeypatch):
    """DeepInfra's rerank endpoint 422s on an empty documents list ("the number of
    queries and documents must be the same"). Short-circuit before the adapter is
    ever called."""
    fake_adapter = AsyncMock()
    gateway = _client(monkeypatch, fake_adapter, {"reranker": "rerank-model"}, {"reranker": "deepinfra"})

    result = await gateway.rerank(role="reranker", query="q", documents=[])

    assert result == []
    fake_adapter.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_get_model_returns_model_for_known_role(monkeypatch):
    gateway = _client(monkeypatch, AsyncMock(), {"slm": "meta-llama/Meta-Llama-3.1-8B-Instruct"}, {"slm": "deepinfra"})

    result = await gateway.get_model(role="slm")

    assert result == "meta-llama/Meta-Llama-3.1-8B-Instruct"


@pytest.mark.asyncio
async def test_get_model_rejects_unknown_role(monkeypatch):
    gateway = _client(monkeypatch, AsyncMock(), {"slm": "some-model"}, {"slm": "deepinfra"})

    with pytest.raises(UnknownRoleError):
        await gateway.get_model(role="nonexistent")


@pytest.mark.asyncio
async def test_chat_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None)
    gateway = _client(monkeypatch, fake_adapter, {"slm": "default-model"}, {"slm": "deepinfra"})

    await gateway.chat(
        role="slm", messages=[{"role": "user", "content": "hi"}], model="candidate-model",
    )

    fake_adapter.chat.assert_awaited_once_with(
        "candidate-model", [{"role": "user", "content": "hi"}], None, None, role="slm", reasoning_enabled=True,
    )


@pytest.mark.asyncio
async def test_chat_falls_back_to_role_default_when_model_omitted(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None)
    gateway = _client(monkeypatch, fake_adapter, {"slm": "default-model"}, {"slm": "deepinfra"})

    await gateway.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    fake_adapter.chat.assert_awaited_once_with(
        "default-model", [{"role": "user", "content": "hi"}], None, None, role="slm", reasoning_enabled=True,
    )


@pytest.mark.asyncio
async def test_rerank_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    gateway = _client(monkeypatch, fake_adapter, {"reranker": "default-reranker"}, {"reranker": "deepinfra"})

    await gateway.rerank(role="reranker", query="q", documents=["a", "b"], model="candidate-reranker")

    fake_adapter.rerank.assert_awaited_once_with("candidate-reranker", "q", ["a", "b"], instruction=None)


@pytest.mark.asyncio
async def test_chat_respects_role_reasoning_map(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None)
    gateway = _client(
        monkeypatch, fake_adapter, {"slm": "default-model"}, {"slm": "deepinfra"},
        reasoning_map={"slm": False},
    )

    await gateway.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    fake_adapter.chat.assert_awaited_once_with(
        "default-model", [{"role": "user", "content": "hi"}], None, None, role="slm", reasoning_enabled=False,
    )
