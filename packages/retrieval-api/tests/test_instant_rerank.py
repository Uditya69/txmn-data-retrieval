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
async def test_rerank_instant_results_rrf_merges_es_and_milvus():
    es_result = [{"doc_id": "d1", "score": 10.0, "heading": "h1", "subheading": "s1"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    # label="INTENT" weights milvus_dense (1.5) over es (1.0), so d2 outranks d1.
    result = await rerank_instant_results(
        label="INTENT", es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
    )

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    for row in result:
        assert "rrf_score" in row


@pytest.mark.asyncio
async def test_rerank_instant_results_plain_es_candidates_keep_score_field_when_rrf_off():
    es_result = [{"doc_id": "d1", "score": 10.0}]

    result = await rerank_instant_results(
        label="INTENT", es_result=es_result, milvus_dense={}, milvus_sparse={}, rrf=False,
    )

    assert result == [{"doc_id": "d1", "score": 10.0}]
    assert "rrf_score" not in result[0]


@pytest.mark.asyncio
async def test_rerank_instant_results_falls_back_to_milvus_when_plan_skips_es():
    """A routing plan that skipped ES entirely (e.g. the INTENT classifier label) must not
    fall back to an empty es_result - Milvus's own hits are the only source that ran."""
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}
    milvus_sparse = {"ruling": [{"doc_id": "d3", "score": 3.0, "chunk_id": "c2", "text": "t3"}]}

    result = await rerank_instant_results(
        label="INTENT", es_result=[], milvus_dense=milvus_dense, milvus_sparse=milvus_sparse,
        rrf=False, plan={"es": False, "milvus": True, "fuse": False},
    )

    assert {row["doc_id"] for row in result} == {"d2", "d3"}


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rrf_merge_step_with_candidates():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    es_result = [{"doc_id": "d1", "score": 10.0}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    result = await rerank_instant_results(
        label="INTENT", es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]
    rrf_step_data = steps[0][1]
    assert rrf_step_data["candidate_count"] == len(result)
    assert rrf_step_data["top_candidates"] == result


@pytest.mark.asyncio
async def test_rerank_instant_results_emits_rrf_merge_step_when_rrf_false_too():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await rerank_instant_results(
        label="INTENT", es_result=[{"doc_id": "d1", "score": 10.0}], milvus_dense={}, milvus_sparse={},
        rrf=False, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]
