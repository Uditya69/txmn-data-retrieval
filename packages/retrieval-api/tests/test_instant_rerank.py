from unittest.mock import AsyncMock

import pytest

from retrieval_api.instant.rerank import _collapse_to_doc_id, _union_by_doc_id, rerank_instant_results, rrf_merge_by_doc_id


def test_collapse_to_doc_id_keeps_first_occurrence_per_doc_id():
    rows = [
        {"doc_id": "d1", "score": 9.0},
        {"doc_id": "d2", "score": 5.0},
        {"doc_id": "d1", "score": 1.0},  # later duplicate, dropped
    ]
    assert _collapse_to_doc_id(rows) == [{"doc_id": "d1", "score": 9.0}, {"doc_id": "d2", "score": 5.0}]


def test_union_by_doc_id_dedups_across_sources_with_no_scoring():
    es_result = [{"doc_id": "d1", "score": 9.0}, {"doc_id": "d2", "score": 1.0}]
    milvus_dense = [{"doc_id": "d2", "score": 99.0}, {"doc_id": "d3", "score": 5.0}]

    union = _union_by_doc_id(es_result, milvus_dense)

    # d2 kept once, from the source it appeared in first (no rank/score math involved).
    assert [row["doc_id"] for row in union] == ["d1", "d2", "d3"]
    assert union[1]["score"] == 1.0


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
async def test_rerank_instant_results_rrf_merges_es_and_milvus():
    es_result = [{"doc_id": "d1", "score": 10.0, "heading": "h1", "subheading": "s1"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    # label="INTENT" weights milvus_dense (1.5) over es (1.0), so d2 outranks d1.
    result = await rerank_instant_results(
        gateway=None, es_client=None, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={}, rrf=True,
    )

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    for row in result:
        assert "rrf_score" in row


@pytest.mark.asyncio
async def test_rerank_instant_results_plain_es_candidates_keep_score_field_when_rrf_and_rerank_off():
    es_result = [{"doc_id": "d1", "score": 10.0}]

    result = await rerank_instant_results(
        gateway=None, es_client=None, query="q", label="INTENT",
        es_result=es_result, milvus_dense={}, milvus_sparse={}, rrf=False, rerank=False,
    )

    assert result == [{"doc_id": "d1", "score": 10.0}]
    assert "rrf_score" not in result[0]
    assert "rerank_score" not in result[0]


@pytest.mark.asyncio
async def test_rerank_instant_results_falls_back_to_milvus_when_plan_skips_es_and_rrf_off():
    """A routing plan that skipped ES entirely (e.g. the INTENT classifier label) must not
    fall back to an empty es_result - Milvus dense's own hits are the only source that ran.
    milvus_sparse carries no weight (see _fallback_fused/_LABEL_RRF_WEIGHTS) - it's always
    empty in practice since common.config.Settings.milvus_sparse_enabled defaults off
    app-wide - so a populated milvus_sparse here must NOT surface in the result."""
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}
    milvus_sparse = {"ruling": [{"doc_id": "d3", "score": 3.0, "chunk_id": "c2", "text": "t3"}]}

    result = await rerank_instant_results(
        gateway=None, es_client=None, query="q", label="INTENT",
        es_result=[], milvus_dense=milvus_dense, milvus_sparse=milvus_sparse,
        rrf=False, rerank=False, plan={"es": False, "milvus": True, "fuse": False},
    )

    assert {row["doc_id"] for row in result} == {"d2"}


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rrf_merge_step_with_candidates():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        gateway=None, es_client=None, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]
    rrf_step_data = steps[0][1]
    assert rrf_step_data["candidate_count"] == len(result)
    assert rrf_step_data["top_candidates"] == result


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rrf_merge_step_when_rrf_and_rerank_false():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await rerank_instant_results(
        gateway=None, es_client=None, query="q", label="INTENT",
        es_result=[{"doc_id": "d1", "score": 10.0}], milvus_dense={}, milvus_sparse={},
        rrf=False, rerank=False, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]


@pytest.mark.asyncio
async def test_rerank_instant_results_reranks_union_of_es_and_milvus_via_gateway():
    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 1.0, "chunk_id": "c1", "text": "t2"}]}

    gateway = AsyncMock()
    # Scores close enough that elbow_cutoff (ratio 0.6) keeps both, so ordering can be
    # asserted without the cutoff trimming d1 away.
    gateway.rerank.return_value = [0.7, 0.9]  # d1 scores lower, d2 scores higher

    es_client = object()
    fulltext = {"d1": "full text one", "d2": "full text two"}

    async def fake_fetch_fulltext_batch(client, doc_ids):
        assert client is es_client
        return {doc_id: fulltext[doc_id] for doc_id in doc_ids}

    import retrieval_api.instant.rerank as rerank_module
    original_fetch = rerank_module.fetch_fulltext_batch
    rerank_module.fetch_fulltext_batch = fake_fetch_fulltext_batch
    try:
        result = await rerank_instant_results(
            gateway=gateway, es_client=es_client, query="q", label="INTENT",
            es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
            rrf=True, rerank=True,
        )
    finally:
        rerank_module.fetch_fulltext_batch = original_fetch

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.7
    # rrf=True is irrelevant once rerank=True - the reranker call happened, so no
    # rrf_score should leak into the final rows (candidate gathering was a plain union).
    assert "rrf_score" not in result[0]
    gateway.rerank.assert_awaited_once_with(role="reranker", query="q", documents=["full text one", "full text two"])


@pytest.mark.asyncio
async def test_rerank_instant_results_reranker_covers_plan_skip_es_case_without_special_casing():
    """Unlike the rrf=False/rerank=False fallback (which needs _fallback_fused's explicit
    plan-skips-es branch), the reranker's plain union naturally handles this: an empty
    es_result just contributes nothing."""
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    gateway = AsyncMock()
    gateway.rerank.return_value = [0.5]

    es_client = object()

    import retrieval_api.instant.rerank as rerank_module
    original_fetch = rerank_module.fetch_fulltext_batch
    rerank_module.fetch_fulltext_batch = AsyncMock(return_value={"d2": "full text"})
    try:
        result = await rerank_instant_results(
            gateway=gateway, es_client=es_client, query="q", label="INTENT",
            es_result=[], milvus_dense=milvus_dense, milvus_sparse={},
            rrf=False, rerank=True, plan={"es": False, "milvus": True, "fuse": False},
        )
    finally:
        rerank_module.fetch_fulltext_batch = original_fetch

    assert {row["doc_id"] for row in result} == {"d2"}


@pytest.mark.asyncio
async def test_rerank_instant_results_returns_empty_when_no_candidates_and_rerank_on():
    result = await rerank_instant_results(
        gateway=AsyncMock(), es_client=object(), query="q", label="INTENT",
        es_result=[], milvus_dense={}, milvus_sparse={}, rrf=True, rerank=True,
    )

    assert result == []
