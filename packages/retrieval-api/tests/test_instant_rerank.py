from unittest.mock import AsyncMock

import pytest

from retrieval_api.instant.rerank import (
    _RERANK_INSTRUCTION,
    _collapse_to_doc_id,
    _enrich_es_only_candidates_with_milvus_text,
    _union_by_doc_id,
    rerank_instant_results,
    rrf_merge_by_doc_id,
)


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
        gateway=None, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={}, rrf=True,
    )

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    for row in result:
        assert "rrf_score" in row


@pytest.mark.asyncio
async def test_rerank_instant_results_plain_es_candidates_keep_score_field_when_rrf_and_rerank_off():
    es_result = [{"doc_id": "d1", "score": 10.0}]

    result = await rerank_instant_results(
        gateway=None, query="q", label="INTENT",
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
        gateway=None, query="q", label="INTENT",
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
        gateway=None, query="q", label="INTENT",
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
        gateway=None, query="q", label="INTENT",
        es_result=[{"doc_id": "d1", "score": 10.0}], milvus_dense={}, milvus_sparse={},
        rrf=False, rerank=False, on_step=on_step,
    )

    assert [s for s, _ in steps] == ["rrf_merge"]


@pytest.mark.asyncio
async def test_rerank_instant_results_reranks_union_of_es_and_milvus_via_gateway():
    es_result = [{"doc_id": "d1", "score": 10.0, "text": "full text one"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 1.0, "chunk_id": "c1", "text": "full text two"}]}

    gateway = AsyncMock()
    # Scores close enough that elbow_cutoff (ratio 0.6) keeps both, so ordering can be
    # asserted without the cutoff trimming d1 away.
    gateway.rerank.return_value = [0.7, 0.9]  # d1 scores lower, d2 scores higher

    result = await rerank_instant_results(
        gateway=gateway, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=False, rerank=True,
    )

    assert [row["doc_id"] for row in result] == ["d2", "d1"]
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.7
    # Plain union candidate gathering (rrf=False) - no rrf_score should leak into the
    # final rows.
    assert "rrf_score" not in result[0]
    gateway.rerank.assert_awaited_once_with(role="reranker", query="q", documents=["full text one", "full text two"], model=None, instruction=_RERANK_INSTRUCTION)


@pytest.mark.asyncio
async def test_rerank_instant_results_rrf_and_rerank_together_rrf_selects_candidates_then_reranks():
    """rrf=True, rerank=True: candidate selection uses RRF fusion (not a plain union), but
    the final ordering still comes from the cross-encoder, not rrf_score - RRF only decides
    which 20 candidates reach the reranker, and its own ranking is discarded once picked."""
    # label="INTENT" weights milvus_dense (1.5) over es (1.0) - d2 outranks d1 in RRF fusion,
    # but the reranker scores d1 higher, so the final order should follow the reranker.
    es_result = [{"doc_id": "d1", "score": 10.0, "text": "full text one"}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "full text two"}]}

    gateway = AsyncMock()
    # RRF (label=INTENT) orders d2 before d1, so documents=[text_for_d2, text_for_d1] -
    # scores assign d2=0.7, d1=0.9, so the reranker's own order (d1 first) overrides RRF's.
    gateway.rerank.return_value = [0.7, 0.9]

    result = await rerank_instant_results(
        gateway=gateway, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, rerank=True,
    )

    # Both candidates reach the reranker (RRF fusion, not the union, selected them - but
    # with only one candidate per source here, membership is the same either way; what
    # matters is the final order follows rerank_score, not RRF's own fused ranking).
    assert {row["doc_id"] for row in result} == {"d1", "d2"}
    assert [row["doc_id"] for row in result] == ["d1", "d2"]
    # rrf_score survives as a leftover field from RRF-based candidate selection, but the
    # final order follows rerank_score, not it - d1 has the lower rrf_score (es-only) yet
    # ranks first here because the reranker scored it higher.
    assert "rrf_score" in result[0]
    assert result[0]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_rerank_instant_results_drops_candidates_with_no_text_before_reranking():
    """An ES row with nothing highlightable for this query (raw_search found no fragment)
    has no text to send the reranker - it must be dropped from the pool rather than sent
    as an empty string, same as a fetch failure used to be handled pre-highlight-refactor."""
    es_result = [{"doc_id": "d1", "score": 10.0, "text": ""}]
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 1.0, "chunk_id": "c1", "text": "full text two"}]}

    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9]

    result = await rerank_instant_results(
        gateway=gateway, query="q", label="INTENT",
        es_result=es_result, milvus_dense=milvus_dense, milvus_sparse={},
        rrf=True, rerank=True,
    )

    assert [row["doc_id"] for row in result] == ["d2"]
    gateway.rerank.assert_awaited_once_with(role="reranker", query="q", documents=["full text two"], model=None, instruction=_RERANK_INSTRUCTION)


@pytest.mark.asyncio
async def test_rerank_instant_results_reranker_covers_plan_skip_es_case_without_special_casing():
    """Unlike the rrf=False/rerank=False fallback (which needs _fallback_fused's explicit
    plan-skips-es branch), the reranker's plain union naturally handles this: an empty
    es_result just contributes nothing."""
    milvus_dense = {"ruling": [{"doc_id": "d2", "score": 5.0, "chunk_id": "c1", "text": "t2"}]}

    gateway = AsyncMock()
    gateway.rerank.return_value = [0.5]

    result = await rerank_instant_results(
        gateway=gateway, query="q", label="INTENT",
        es_result=[], milvus_dense=milvus_dense, milvus_sparse={},
        rrf=False, rerank=True, plan={"es": False, "milvus": True, "fuse": False},
    )

    assert {row["doc_id"] for row in result} == {"d2"}


@pytest.mark.asyncio
async def test_rerank_instant_results_returns_empty_when_no_candidates_and_rerank_on():
    result = await rerank_instant_results(
        gateway=AsyncMock(), query="q", label="INTENT",
        es_result=[], milvus_dense={}, milvus_sparse={}, rrf=True, rerank=True,
    )

    assert result == []


@pytest.mark.asyncio
async def test_enrich_es_only_candidates_skips_milvus_lookup_when_no_dense_vector():
    candidates = [{"doc_id": "d1", "score": 10.0, "text": "es snippet"}]

    result = await _enrich_es_only_candidates_with_milvus_text(candidates, "q", milvus_client=object(), dense_vector=None)

    assert result == candidates


@pytest.mark.asyncio
async def test_enrich_es_only_candidates_leaves_milvus_origin_rows_untouched(monkeypatch):
    """A candidate that already has a chunk_id came from Milvus, not ES - it must not be
    looked up again (and the fake hybrid_search below asserts it's never even called)."""
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("hybrid_search should not be called - no ES-only candidates")

    import retrieval_api.instant.rerank as rerank_module
    monkeypatch.setattr(rerank_module, "hybrid_search", fail_if_called)

    candidates = [{"doc_id": "d1", "chunk_id": "c1", "score": 10.0, "text": "milvus chunk"}]

    result = await _enrich_es_only_candidates_with_milvus_text(candidates, "q", milvus_client=object(), dense_vector=[0.1])

    assert result == candidates


@pytest.mark.asyncio
async def test_enrich_es_only_candidates_replaces_text_with_best_scoring_milvus_chunk(monkeypatch):
    """d1 has chunks in two collections - the higher-scoring one (held, 9.0) wins over
    ruling (5.0), and the doc_id_allowlist/dense_vector passed through unchanged."""
    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert doc_id_allowlist == ["d1"]
        assert dense_vector == [0.1, 0.2]
        assert limit == 1
        return {
            "ruling": [{"doc_id": "d1", "chunk_id": "c1", "text": "ruling chunk", "score": 5.0}],
            "held": [{"doc_id": "d1", "chunk_id": "c2", "text": "held chunk", "score": 9.0}],
        }

    import retrieval_api.instant.rerank as rerank_module
    monkeypatch.setattr(rerank_module, "hybrid_search", fake_hybrid_search)

    candidates = [{"doc_id": "d1", "score": 10.0, "text": "es highlight snippet"}]

    result = await _enrich_es_only_candidates_with_milvus_text(
        candidates, "q", milvus_client=object(), dense_vector=[0.1, 0.2],
    )

    assert result == [{"doc_id": "d1", "score": 10.0, "text": "held chunk"}]


@pytest.mark.asyncio
async def test_enrich_es_only_candidates_keeps_existing_text_when_milvus_has_nothing(monkeypatch):
    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    import retrieval_api.instant.rerank as rerank_module
    monkeypatch.setattr(rerank_module, "hybrid_search", fake_hybrid_search)

    candidates = [{"doc_id": "d1", "score": 10.0, "text": "es highlight snippet"}]

    result = await _enrich_es_only_candidates_with_milvus_text(
        candidates, "q", milvus_client=object(), dense_vector=[0.1, 0.2],
    )

    assert result == candidates


@pytest.mark.asyncio
async def test_rerank_instant_results_enriches_es_only_candidate_via_milvus_client(monkeypatch):
    """End-to-end through rerank_instant_results: an ES-only candidate's text is replaced
    by a real Milvus chunk fetched via the milvus_client/dense_vector params, and the
    reranker sees that chunk instead of ES's own highlight snippet."""
    es_result = [{"doc_id": "d1", "score": 10.0, "text": "es highlight snippet"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"doc_id": "d1", "chunk_id": "c1", "text": "real milvus chunk", "score": 5.0}]}

    import retrieval_api.instant.rerank as rerank_module
    monkeypatch.setattr(rerank_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9]

    result = await rerank_instant_results(
        gateway=gateway, query="q", label="INTENT",
        es_result=es_result, milvus_dense={}, milvus_sparse={},
        rrf=False, rerank=True, milvus_client=object(), dense_vector=[0.1, 0.2],
    )

    assert result[0]["doc_id"] == "d1"
    gateway.rerank.assert_awaited_once_with(role="reranker", query="q", documents=["real milvus chunk"], model=None, instruction=_RERANK_INSTRUCTION)
