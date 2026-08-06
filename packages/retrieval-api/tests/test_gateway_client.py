import httpx
import pytest
import respx

from retrieval_api.gateway_client import GatewayClient


def test_trace_headers_can_be_disabled(monkeypatch):
    import retrieval_api.gateway_client as module

    monkeypatch.setattr(module, "_trace_headers", lambda: (_ for _ in ()).throw(AssertionError("called")))
    assert GatewayClient("http://gateway", trace_enabled=False)._headers() == {}


@pytest.mark.asyncio
@respx.mock
async def test_chat_calls_gateway_and_unwraps_content():
    respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    assert result == "hi there"


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_reasoning_returns_content_and_reasoning():
    respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there", "reasoning": "thinking..."})
    )
    client = GatewayClient(base_url="http://gateway")

    content, reasoning = await client.chat_with_reasoning(role="synthesis", messages=[{"role": "user", "content": "hi"}])

    assert content == "hi there"
    assert reasoning == "thinking..."


@pytest.mark.asyncio
@respx.mock
async def test_embed_unwraps_embedding():
    respx.post("http://gateway/v1/embed").mock(
        return_value=httpx.Response(200, json={"embedding": [1.0, 2.0]})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.embed(role="query_embed", text="hello")

    assert result == [1.0, 2.0]


@pytest.mark.asyncio
@respx.mock
async def test_rerank_unwraps_scores():
    respx.post("http://gateway/v1/rerank").mock(
        return_value=httpx.Response(200, json={"scores": [0.5]})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.rerank(role="reranker", query="q", documents=["a"])

    assert result == [0.5]
