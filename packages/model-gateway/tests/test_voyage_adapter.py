import httpx
import pytest
import respx

from model_gateway.adapters.voyage import VoyageAdapter


@pytest.mark.asyncio
@respx.mock
async def test_embed_sends_query_input_type_and_returns_vector():
    route = respx.post("https://api.voyageai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.4, 0.5, 0.6]}]})
    )
    adapter = VoyageAdapter(api_key="k")

    result = await adapter.embed("voyage-4-large", "some query text")

    assert result == [0.4, 0.5, 0.6]
    sent_body = route.calls.last.request.content
    import json
    payload = json.loads(sent_body)
    assert payload == {"input": ["some query text"], "model": "voyage-4-large", "input_type": "query"}


@pytest.mark.asyncio
async def test_chat_raises_not_implemented():
    adapter = VoyageAdapter(api_key="k")

    with pytest.raises(NotImplementedError):
        await adapter.chat("model", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_rerank_raises_not_implemented():
    adapter = VoyageAdapter(api_key="k")

    with pytest.raises(NotImplementedError):
        await adapter.rerank("model", "query", ["a", "b"])
