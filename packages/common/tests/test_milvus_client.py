import asyncio
from unittest.mock import MagicMock
import pytest
from common.milvus_client import hybrid_search


class FakeMilvusClient:
    def __init__(self, hits_by_collection):
        self.hits_by_collection = hits_by_collection
        self.calls = []

    def search(self, collection_name, data, anns_field, limit, filter=None, output_fields=None, **kwargs):
        self.calls.append((collection_name, anns_field, filter))
        return [self.hits_by_collection.get(collection_name, [])]


def _hit(chunk_id, doc_id, text, score):
    return {"id": chunk_id, "distance": score, "entity": {"doc_id": doc_id, "text": text}}


@pytest.mark.asyncio
async def test_hybrid_search_runs_all_collections_concurrently_and_shapes_rows():
    client = FakeMilvusClient({
        "ruling": [_hit("d1::ruling::0", "d1", "ruling text", 0.9)],
        "facts": [_hit("d2::facts::0", "d2", "facts text", 0.8)],
    })

    result = await hybrid_search(
        client, collections=["ruling", "facts"],
        dense_vector=[0.1, 0.2], sparse_query_text="income tax",
    )

    assert result["ruling"] == [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "ruling text", "score": 0.9}]
    assert result["facts"] == [{"chunk_id": "d2::facts::0", "doc_id": "d2", "text": "facts text", "score": 0.8}]
    assert {c for c, _, _ in client.calls} == {"ruling", "facts"}


@pytest.mark.asyncio
async def test_hybrid_search_applies_doc_id_allowlist_filter():
    client = FakeMilvusClient({"ruling": []})

    await hybrid_search(
        client, collections=["ruling"],
        dense_vector=[0.1], sparse_query_text="q",
        doc_id_allowlist=["d1", "d2"],
    )

    _, _, filter_expr = client.calls[0]
    assert filter_expr == 'doc_id in ["d1", "d2"]'
