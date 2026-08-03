import httpx
import pytest
import respx

from model_gateway.adapters.deepinfra import DeepInfraAdapter


@pytest.mark.asyncio
@respx.mock
async def test_chat_posts_openai_shape_and_returns_content():
    respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    assert result == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_embed_returns_vector():
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.embed("embed-model", "some text")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@respx.mock
async def test_rerank_returns_scores_in_input_order():
    respx.post("https://api.deepinfra.com/v1/inference/rerank-model").mock(
        return_value=httpx.Response(200, json={"scores": [0.9, 0.2]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.rerank("rerank-model", "query", ["doc a", "doc b"])

    assert result == [0.9, 0.2]
