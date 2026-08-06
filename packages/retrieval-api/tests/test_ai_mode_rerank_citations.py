from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.rerank import rerank_top_chunks
from retrieval_api.ai_mode.citations import prefetch_citations, rerank_and_prefetch


@pytest.mark.asyncio
async def test_rerank_top_chunks_sorts_by_score_and_truncates():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.2, 0.9, 0.5]
    candidates = [
        {"chunk_id": "a", "text": "A", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "B", "rrf_score": 0.02},
        {"chunk_id": "c", "text": "C", "rrf_score": 0.01},
    ]

    result = await rerank_top_chunks(gateway, "query", candidates, top_n=2)

    assert [row["chunk_id"] for row in result] == ["b", "c"]
    assert result[0]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_rerank_top_chunks_defaults_to_single_chunk_when_one_dominates():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9, 0.1, 0.05]
    candidates = [
        {"chunk_id": "a", "text": "A", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "B", "rrf_score": 0.02},
        {"chunk_id": "c", "text": "C", "rrf_score": 0.01},
    ]

    result = await rerank_top_chunks(gateway, "query", candidates)

    assert [row["chunk_id"] for row in result] == ["a"]


@pytest.mark.asyncio
async def test_rerank_top_chunks_defaults_extend_while_scores_stay_close():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9, 0.85, 0.3]
    candidates = [
        {"chunk_id": "a", "text": "A", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "B", "rrf_score": 0.02},
        {"chunk_id": "c", "text": "C", "rrf_score": 0.01},
    ]

    result = await rerank_top_chunks(gateway, "query", candidates)

    assert [row["chunk_id"] for row in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_top_chunks_defaults_cap_at_five_even_if_all_close():
    gateway = AsyncMock()
    scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
    gateway.rerank.return_value = scores
    candidates = [
        {"chunk_id": str(i), "text": str(i), "rrf_score": 0.01} for i in range(len(scores))
    ]

    result = await rerank_top_chunks(gateway, "query", candidates)

    assert len(result) == 5
    assert [row["chunk_id"] for row in result] == ["0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_prefetch_citations_dedupes_doc_ids_and_caps_at_top_n(monkeypatch):
    import retrieval_api.ai_mode.citations as module

    async def fake_fetch_citations(client, doc_ids):
        assert doc_ids == ["d1", "d2"]
        return {"d1": {"masterinfo": {"court": "SC"}}}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)
    candidates = [
        {"chunk_id": "x", "doc_id": "d1", "rrf_score": 0.9},
        {"chunk_id": "y", "doc_id": "d1", "rrf_score": 0.8},
        {"chunk_id": "z", "doc_id": "d2", "rrf_score": 0.7},
    ]

    result = await prefetch_citations(es_client=object(), candidates=candidates, top_n_docs=2)

    assert result == {"d1": {"masterinfo": {"court": "SC"}}}


@pytest.mark.asyncio
async def test_rerank_and_prefetch_runs_both_concurrently(monkeypatch):
    import retrieval_api.ai_mode.citations as citations_module
    import retrieval_api.ai_mode.rerank as rerank_module

    async def fake_prefetch(es_client, candidates, top_n_docs=20):
        return {"d1": {"masterinfo": {}}}

    async def fake_rerank_top(gateway, query, candidates, top_n=3):
        return [{"chunk_id": "a"}]

    monkeypatch.setattr(citations_module, "prefetch_citations", fake_prefetch)
    monkeypatch.setattr(rerank_module, "rerank_top_chunks", fake_rerank_top)

    top_chunks, citations = await rerank_and_prefetch(
        gateway=object(), es_client=object(), query="q", candidates=[{"chunk_id": "a", "doc_id": "d1"}],
    )

    assert top_chunks == [{"chunk_id": "a"}]
    assert citations == {"d1": {"masterinfo": {}}}


@pytest.mark.asyncio
async def test_rerank_and_prefetch_emits_rerank_step(monkeypatch):
    import retrieval_api.ai_mode.citations as citations_module
    import retrieval_api.ai_mode.rerank as rerank_module

    async def fake_prefetch(es_client, candidates, top_n_docs=20):
        return {"d1": {"masterinfo": {}}}

    async def fake_rerank_top(gateway, query, candidates, top_n=3):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "chunk text", "rerank_score": 0.95}]

    monkeypatch.setattr(citations_module, "prefetch_citations", fake_prefetch)
    monkeypatch.setattr(rerank_module, "rerank_top_chunks", fake_rerank_top)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await rerank_and_prefetch(
        gateway=object(), es_client=object(), query="q",
        candidates=[{"chunk_id": "a", "doc_id": "d1"}], on_step=on_step,
    )

    assert steps == [("rerank", {
        "considered_count": 1,
        "top_chunks": [{"chunk_id": "a", "doc_id": "d1", "rerank_score": 0.95, "text": "chunk text"}],
    })]
