from unittest.mock import AsyncMock

import pytest

from retrieval_api.instant.rerank import _collapse_to_doc_id, rerank_instant_results, rrf_merge_by_doc_id


def test_collapse_to_doc_id_keeps_first_occurrence_per_doc_id():
    rows = [
        {"doc_id": "d1", "score": 9.0},
        {"doc_id": "d2", "score": 5.0},
        {"doc_id": "d1", "score": 1.0},  # later duplicate, dropped
    ]
    assert _collapse_to_doc_id(rows) == [{"doc_id": "d1", "score": 9.0}, {"doc_id": "d2", "score": 5.0}]


def test_rrf_merge_by_doc_id_upweights_source_with_higher_weight():
    sources = {
        "es": [{"doc_id": "d1", "score": 5.0}],
        "milvus_dense": [{"doc_id": "d2", "score": 5.0}],
        "milvus_sparse": [],
    }
    # es weighted higher than milvus_dense -> d1 (rank 1 in es) outranks d2 (rank 1 in milvus_dense)
    fused = rrf_merge_by_doc_id(sources, {"es": 1.5, "milvus_dense": 0.5, "milvus_sparse": 1.5})
    assert [row["doc_id"] for row in fused] == ["d1", "d2"]


def test_rrf_merge_by_doc_id_combines_scores_across_sources_for_same_doc():
    sources = {
        "es": [{"doc_id": "d1", "score": 5.0}, {"doc_id": "d2", "score": 4.0}],
        "milvus_dense": [{"doc_id": "d2", "score": 9.0, "chunk_id": "c1", "text": "t"}],
        "milvus_sparse": [],
    }
    fused = rrf_merge_by_doc_id(sources, {"es": 1.0, "milvus_dense": 1.0, "milvus_sparse": 1.0})
    # d2 appears in two sources (rank 1 in each) -> higher combined rrf_score than d1 (rank 1 in only one)
    assert fused[0]["doc_id"] == "d2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


@pytest.mark.asyncio
async def test_rerank_instant_results_fetches_fulltext_and_sorts_by_rerank_score(monkeypatch):
    import retrieval_api.instant.rerank as rerank_module

    async def fake_fetch_fulltext_batch(client, doc_ids):
        return {doc_id: f"fulltext for {doc_id}" for doc_id in doc_ids}

    monkeypatch.setattr(rerank_module, "fetch_fulltext_batch", fake_fetch_fulltext_batch)

    gateway = AsyncMock()
    # shape="plain" weights milvus_dense (1.5) over es (1.0), so d2 is the
    # higher-ranked fused candidate and comes first in the documents list.
    gateway.rerank.return_value = [0.9, 0.7]  # both stay within elbow_cutoff's 0.6 ratio

    es_result = [{"doc_id": "d1", "score": 10.0, "heading": "h1", "subheading": "s1"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", shape="plain",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
    )

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    gateway.rerank.assert_awaited_once_with(role="reranker", query="q", documents=["fulltext for d2", "fulltext for d1"])


@pytest.mark.asyncio
async def test_rerank_instant_results_drops_candidates_with_no_fulltext(monkeypatch):
    import retrieval_api.instant.rerank as rerank_module

    async def fake_fetch_fulltext_batch(client, doc_ids):
        return {}  # nothing found in ES

    monkeypatch.setattr(rerank_module, "fetch_fulltext_batch", fake_fetch_fulltext_batch)

    gateway = AsyncMock()

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", shape="plain",
        es_result=[{"doc_id": "d1", "score": 1.0}], milvus_dense={}, milvus_sparse={},
    )

    assert result == []
    gateway.rerank.assert_not_awaited()
