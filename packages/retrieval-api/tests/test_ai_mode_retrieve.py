from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.retrieve import rrf_merge, retrieve


def test_rrf_merge_combines_and_ranks_by_reciprocal_rank():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"  # appears rank-1 sparse + rank-2 dense: highest combined score
    assert set(ids) == {"a", "b", "c"}
    assert merged[0]["rrf_score"] > merged[-1]["rrf_score"]


def test_rrf_merge_dedupes_by_chunk_id():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "a", "text": "A"}]

    merged = rrf_merge(dense, sparse)

    assert len(merged) == 1


@pytest.mark.asyncio
async def test_retrieve_embeds_rewritten_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await retrieve(gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=["d1"])

    gateway.embed.assert_awaited_once_with(role="query_embed", text="q")
    assert result[0]["chunk_id"] == "a"


def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = collection_trace({"ruling": rows})

    assert trace == {
        "collections": [{
            "name": "ruling",
            "hit_count": 7,
            "top_hits": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "score": float(i), "text_preview": "x" * 200}
                for i in range(5)
            ],
        }]
    }


@pytest.mark.asyncio
async def test_retrieve_emits_dense_sparse_and_rrf_merge_steps(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await module.retrieve(gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, on_step=on_step)

    assert steps == ["milvus_dense", "milvus_sparse", "rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}
