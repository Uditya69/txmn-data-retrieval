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
        "milvus_dense": [{"doc_id": "d1", "score": 5.0}],
        "milvus_sparse": [{"doc_id": "d2", "score": 5.0}],
    }
    fused = rrf_merge_by_doc_id(sources, {"milvus_dense": 1.5, "milvus_sparse": 0.5})
    assert [row["doc_id"] for row in fused] == ["d1", "d2"]


def test_rrf_merge_by_doc_id_combines_scores_across_sources_for_same_doc():
    sources = {
        "milvus_dense": [{"doc_id": "d2", "score": 9.0, "chunk_id": "c1", "text": "t"}],
        "milvus_sparse": [{"doc_id": "d1", "score": 5.0}, {"doc_id": "d2", "score": 4.0}],
    }
    fused = rrf_merge_by_doc_id(sources, {"milvus_dense": 1.0, "milvus_sparse": 1.0})
    # d2 appears in two sources (rank 1 in each) -> higher combined rrf_score than d1 (rank 1 in only one)
    assert fused[0]["doc_id"] == "d2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


@pytest.mark.asyncio
async def test_rerank_instant_results_reranks_using_existing_milvus_chunk_text():
    gateway = AsyncMock()
    # shape="plain" weights milvus_dense (1.5) over milvus_sparse (0.5), so d2 is
    # the higher-ranked fused candidate and comes first in the documents list.
    gateway.rerank.return_value = [0.9, 0.7]  # both stay within elbow_cutoff's 0.6 ratio

    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "chunk text d2"}]}
    milvus_sparse = {"ruling": [{"doc_id": "d3", "score": 4.0, "chunk_id": "c2", "text": "chunk text d3"}]}

    result = await rerank_instant_results(
        gateway, query="q", shape="plain", es_result=[], milvus_dense=milvus_dense, milvus_sparse=milvus_sparse,
    )

    assert [row["doc_id"] for row in result] == ["d2", "d3"]
    gateway.rerank.assert_awaited_once_with(
        role="reranker", query="q", documents=["chunk text d2", "chunk text d3"],
    )


@pytest.mark.asyncio
async def test_rerank_instant_results_drops_es_only_doc_ids_from_the_reranked_view():
    """ES hits that have no corresponding Milvus row never reach the reranker as a
    full document - they can only boost a doc_id Milvus already surfaced."""
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9]

    milvus_dense = {"ruling": [{"doc_id": "d1", "score": 5.0, "chunk_id": "c1", "text": "chunk text"}]}
    es_result = [{"doc_id": "es-only-doc", "score": 10.0, "heading": "h", "subheading": "s"}]

    result = await rerank_instant_results(
        gateway, query="q", shape="plain", es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
    )

    assert [row["doc_id"] for row in result] == ["d1"]


@pytest.mark.asyncio
async def test_rerank_instant_results_boosts_a_doc_id_es_and_milvus_both_confirm():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9, 0.85]

    # shape="citation" weights milvus_sparse (1.5) over milvus_dense (0.5), so
    # milvus_sparse's rank-1 (d2) would normally outrank milvus_dense's rank-1
    # (d1) - but d1 is also ES's rank-1 hit, and citation shape weights the ES
    # boost at 1.5 too, so the boost should push d1 back to the top.
    milvus_dense = {"ruling": [{"doc_id": "d1", "score": 1.0, "chunk_id": "c1", "text": "text d1"}]}
    milvus_sparse = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c2", "text": "text d2"}]}
    es_result = [{"doc_id": "d1", "score": 9.0, "heading": "h", "subheading": "s"}]

    await rerank_instant_results(
        gateway, query="q", shape="citation", es_result=es_result, milvus_dense=milvus_dense, milvus_sparse=milvus_sparse,
    )

    sent_texts = gateway.rerank.await_args.kwargs["documents"]
    assert sent_texts[0] == "text d1"


@pytest.mark.asyncio
async def test_rerank_instant_results_returns_empty_when_no_milvus_candidates():
    gateway = AsyncMock()

    result = await rerank_instant_results(
        gateway, query="q", shape="plain",
        es_result=[{"doc_id": "d1", "score": 1.0}], milvus_dense={}, milvus_sparse={},
    )

    assert result == []
    gateway.rerank.assert_not_awaited()
