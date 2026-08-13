# packages/retrieval-api/tests/test_instant_search.py
from unittest.mock import AsyncMock
import pytest
from retrieval_api.instant.search import run_instant


@pytest.mark.asyncio
async def test_run_instant_returns_both_branches_on_success(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            assert dense_vector == [0.1, 0.2]  # Instant embeds the raw query for true dense ANN
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="tax exemption")

    assert result["es"] == [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]
    assert result["es_error"] is None
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}
    assert result["milvus_sparse"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}
    assert result["milvus_error"] is None
    gateway.embed.assert_awaited_once_with(role="query_embed", text="tax exemption")


@pytest.mark.asyncio
async def test_run_instant_applies_elbow_cutoff_to_es_and_milvus_results(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        # steep drop after the first hit - only the first should survive
        return [
            {"doc_id": "d1", "score": 10.0},
            {"doc_id": "d2", "score": 1.0},
            {"doc_id": "d3", "score": 0.1},
        ]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {
            "ruling": [
                {"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 10.0},
                {"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 1.0},
            ],
        }

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] == [{"doc_id": "d1", "score": 10.0}]
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 10.0}]}
    assert result["milvus_sparse"] == {
        "ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 10.0}],
    }


@pytest.mark.asyncio
async def test_run_instant_keeps_flat_score_distribution_uncapped(monkeypatch):
    import retrieval_api.instant.search as search_module

    flat_scores = [{"doc_id": f"d{i}", "score": 5.0} for i in range(12)]

    async def fake_raw_search(client, query, limit=20):
        return flat_scores

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    # AI Mode's rerank caps at 5 regardless of flatness; Instant has no such
    # ceiling since it's a UI preview list, not an LLM prompt.
    assert result["es"] == flat_scores


@pytest.mark.asyncio
async def test_run_instant_returns_partial_result_when_es_fails(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def failing_raw_search(client, query, limit=20):
        raise RuntimeError("ES down")

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", failing_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] is None
    assert result["es_error"] == "ES down"
    assert result["milvus"] == {"ruling": []}
    assert result["milvus_sparse"] == {"ruling": []}
    assert result["milvus_error"] is None


@pytest.mark.asyncio
async def test_run_instant_returns_partial_result_when_gateway_embed_fails(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)

    gateway = AsyncMock()
    gateway.embed.side_effect = RuntimeError("gateway down")

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] == [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]
    assert result["es_error"] is None
    assert result["milvus"] is None
    assert result["milvus_sparse"] is None
    assert result["milvus_error"] == "gateway down"


@pytest.mark.asyncio
async def test_run_instant_returns_reranked_list_when_rerank_flag_set(monkeypatch):
    """d1 is an ES-only hit (no corresponding Milvus row) - reranking only ever
    fuses/reranks Milvus-sourced doc_ids, so d1 is correctly dropped from the
    reranked view rather than sent to the reranker as a full ES document."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2, "heading": "h1", "subheading": "s1"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    gateway.rerank.return_value = [0.9]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", rerank=True)

    assert "es" not in result
    assert "milvus" not in result
    assert result["reranked_error"] is None
    assert {row["doc_id"] for row in result["reranked"]} == {"d2"}


@pytest.mark.asyncio
async def test_run_instant_skips_rerank_when_es_branch_failed(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def failing_raw_search(client, query, limit=20):
        raise RuntimeError("ES down")

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", failing_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", rerank=True)

    assert result["reranked"] == []
    assert result["reranked_error"] == "ES down"


@pytest.mark.asyncio
async def test_run_instant_emits_es_and_milvus_trace_steps(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        row = {"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}
        return {"ruling": [row] if dense_vector is not None else [row]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = []

    async def on_step(step, data):
        steps.append(step)

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", on_step=on_step)

    # es_search runs on an independent branch (asyncio.gather with _run_milvus)
    # so its relative order vs. the milvus steps isn't guaranteed - only that
    # dense precedes sparse within the milvus branch, and query_analysis (emitted
    # synchronously before the gather) comes first.
    assert set(steps) == {"query_analysis", "es_search", "milvus_dense", "milvus_sparse"}
    assert steps[0] == "query_analysis"
    assert steps.index("milvus_dense") < steps.index("milvus_sparse")
