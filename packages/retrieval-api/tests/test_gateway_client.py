import json

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


@pytest.mark.asyncio
@respx.mock
async def test_get_model_returns_model_for_role():
    respx.get("http://gateway/v1/models/slm").mock(
        return_value=httpx.Response(200, json={"role": "slm", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.get_model(role="slm")

    assert result == "meta-llama/Meta-Llama-3.1-8B-Instruct"


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_tools_posts_tools_and_returns_tool_calls():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={
            "content": None,
            "reasoning": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
        })
    )
    client = GatewayClient(base_url="http://gateway")
    tools = [{"type": "function", "function": {"name": "search_es", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    result = await client.chat_with_tools("agent_chat", [{"role": "user", "content": "hi"}], tools, tool_choice="auto")

    assert result == {
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
        "reasoning": None,
    }
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"role": "agent_chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools, "tool_choice": "auto"}


@pytest.mark.asyncio
@respx.mock
async def test_trace_enabled_false_sends_no_trace_headers():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway", trace_enabled=False)

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    sent_headers = route.calls.last.request.headers
    assert "x-langfuse-trace-id" not in sent_headers
    assert "x-langfuse-parent-observation-id" not in sent_headers


@pytest.mark.asyncio
@respx.mock
async def test_trace_enabled_default_sends_trace_headers_when_trace_active(monkeypatch):
    import retrieval_api.gateway_client as gateway_client_module

    monkeypatch.setattr(
        gateway_client_module,
        "_trace_headers",
        lambda: {
            "x-langfuse-trace-id": "trace-123",
            "x-langfuse-parent-observation-id": "obs-456",
        },
    )
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    sent_headers = route.calls.last.request.headers
    assert sent_headers.get("x-langfuse-trace-id") == "trace-123"
    assert sent_headers.get("x-langfuse-parent-observation-id") == "obs-456"


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_model_override_when_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}], model="candidate-model")

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-model"


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_model_key_when_not_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "model" not in sent


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_reasoning_sends_model_override():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there", "reasoning": None})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat_with_reasoning(
        role="synthesis", messages=[{"role": "user", "content": "hi"}], model="candidate-synth-model",
    )

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-synth-model"


@pytest.mark.asyncio
@respx.mock
async def test_rerank_sends_model_override():
    route = respx.post("http://gateway/v1/rerank").mock(
        return_value=httpx.Response(200, json={"scores": [0.5]})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.rerank(role="reranker", query="q", documents=["a"], model="candidate-reranker")

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-reranker"
