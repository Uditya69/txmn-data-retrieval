import json

import httpx
import pytest
import respx

from model_gateway.adapters.local_rerank import LocalRerankAdapter


@pytest.mark.asyncio
@respx.mock
async def test_rerank_reorders_results_by_index_back_to_input_order():
    # The self-hosted server returns results sorted by relevance_score, not by input
    # order - index 2 (doc c) scored highest here, but rerank() must hand back scores
    # positionally aligned to the documents list it was given: [doc a, doc b, doc c].
    respx.post("http://localhost:8001/v1/rerank").mock(
        return_value=httpx.Response(200, json={
            "results": [
                {"index": 2, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.87},
                {"index": 1, "relevance_score": 0.01},
            ],
        })
    )
    adapter = LocalRerankAdapter(base_url="http://localhost:8001/v1", api_key="k")

    scores = await adapter.rerank("qwen3-reranker", "query", ["doc a", "doc b", "doc c"])

    assert scores == [0.87, 0.01, 0.99]


@pytest.mark.asyncio
@respx.mock
async def test_rerank_sends_model_query_documents_and_bearer_auth():
    route = respx.post("http://localhost:8001/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})
    )
    adapter = LocalRerankAdapter(base_url="http://localhost:8001/v1", api_key="unsecure-api-key-taxmann")

    await adapter.rerank("qwen3-reranker", "query text", ["doc a"])

    sent = json.loads(route.calls.last.request.content)
    assert sent == {"model": "qwen3-reranker", "query": "query text", "documents": ["doc a"]}
    assert route.calls.last.request.headers["authorization"] == "Bearer unsecure-api-key-taxmann"


@pytest.mark.asyncio
async def test_chat_and_embed_are_not_supported():
    adapter = LocalRerankAdapter(base_url="http://localhost:8001/v1", api_key="k")

    with pytest.raises(NotImplementedError):
        await adapter.chat("model", [{"role": "user", "content": "hi"}])
    with pytest.raises(NotImplementedError):
        await adapter.embed("model", "text")
