import json

import httpx
import pytest
import respx

from model_gateway.adapters.deepinfra import DeepInfraAdapter


@pytest.mark.asyncio
@respx.mock
async def test_chat_posts_openai_shape_and_returns_content():
    respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    )
    adapter = DeepInfraAdapter(api_key="k")

    content, usage, reasoning, tool_calls = await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    assert content == "hello"
    assert usage == {"input": 10, "output": 5}
    assert reasoning is None
    assert tool_calls is None


@pytest.mark.asyncio
@respx.mock
async def test_chat_extracts_reasoning_content_when_present():
    respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "the answer", "reasoning_content": "thinking it through..."}}],
        })
    )
    adapter = DeepInfraAdapter(api_key="k")

    _content, _usage, reasoning, _tool_calls = await adapter.chat("reasoning-model", [{"role": "user", "content": "hi"}])

    assert reasoning == "thinking it through..."


@pytest.mark.asyncio
@respx.mock
async def test_embed_returns_vector():
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(200, json={
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 4},
        })
    )
    adapter = DeepInfraAdapter(api_key="k")

    embedding, usage = await adapter.embed("embed-model", "some text")

    assert embedding == [0.1, 0.2, 0.3]
    assert usage == {"input": 4}


@pytest.mark.asyncio
@respx.mock
async def test_rerank_returns_scores_in_input_order():
    route = respx.post("https://api.deepinfra.com/v1/inference/rerank-model").mock(
        return_value=httpx.Response(200, json={"scores": [0.9, 0.2]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.rerank("rerank-model", "query", ["doc a", "doc b"])

    assert result == [0.9, 0.2]
    # DeepInfra's reranker models (e.g. Qwen3-Reranker) require "queries" as a
    # list, not a bare "query" string - the old bge-reranker-large shape 404s.
    assert json.loads(route.calls.last.request.content) == {
        "queries": ["query"], "documents": ["doc a", "doc b"],
    }


@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_tools_and_returns_tool_calls():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
            }}],
        })
    )
    adapter = DeepInfraAdapter(api_key="k")
    tools = [{"type": "function", "function": {"name": "search_es", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    content, _usage, _reasoning, tool_calls = await adapter.chat(
        "some-model", [{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto",
    )

    assert content is None
    assert tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}]
    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_tools_key_when_not_given():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "tools" not in sent
    assert "tool_choice" not in sent
