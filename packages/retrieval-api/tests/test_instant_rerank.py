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
    # label="INTENT" weights milvus_dense (1.5) over es (1.0), so d2 is the
    # higher-ranked fused candidate and comes first in the documents list.
    gateway.rerank.return_value = [0.9, 0.7]  # both stay within elbow_cutoff's 0.6 ratio

    es_result = [{"doc_id": "d1", "score": 10.0, "heading": "h1", "subheading": "s1"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
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
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=[{"doc_id": "d1", "score": 1.0}], milvus_dense={}, milvus_sparse={},
    )

    assert result == []
    gateway.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_instant_results_returns_top_candidates_as_is_when_rerank_false():
    """rerank=False: rows keep whichever field selected them (rrf_score here,
    since rrf=True) rather than being relabeled rerank_score - that field only
    appears once the cross-encoder actually ran."""
    gateway = AsyncMock()

    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, rerank=False,
    )

    assert {row["doc_id"] for row in result} == {"d1", "d2"}
    for row in result:
        assert "rerank_score" not in row
        assert "rrf_score" in row
    gateway.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_instant_results_plain_es_candidates_keep_score_field_when_rrf_and_rerank_both_off():
    gateway = AsyncMock()

    es_result = [{"doc_id": "d1", "score": 10.0}]

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=es_result, milvus_dense={}, milvus_sparse={},
        rrf=False, rerank=False,
    )

    assert result == [{"doc_id": "d1", "score": 10.0}]
    assert "rerank_score" not in result[0]
    assert "rrf_score" not in result[0]
    gateway.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rrf_merge_step_with_candidates(monkeypatch):
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    gateway = AsyncMock()
    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, rerank=False, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]
    rrf_step_data = steps[0][1]
    assert rrf_step_data["candidate_count"] == len(result)
    assert rrf_step_data["top_candidates"] == result


@pytest.mark.asyncio
async def test_rerank_instant_results_skips_rrf_merge_step_when_rrf_false():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    gateway = AsyncMock()

    await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=[{"doc_id": "d1", "score": 10.0}], milvus_dense={}, milvus_sparse={},
        rrf=False, rerank=False, on_step=on_step,
    )

    assert steps == []


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rerank_step_when_cross_encoder_runs(monkeypatch):
    import retrieval_api.instant.rerank as rerank_module

    async def fake_fetch_fulltext_batch(client, doc_ids):
        return {doc_id: f"fulltext for {doc_id}" for doc_id in doc_ids}

    monkeypatch.setattr(rerank_module, "fetch_fulltext_batch", fake_fetch_fulltext_batch)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9, 0.7]

    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway, es_client=object(), query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, rerank=True, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge", "rerank"]
    rerank_step_data = steps[1][1]
    assert rerank_step_data["total_candidates"] == 2
    assert rerank_step_data["considered_count"] == 2
    assert rerank_step_data["top_chunks"] == result
    for row in result:
        assert "rerank_score" in row
