# packages/retrieval-api/tests/test_instant_search.py
from unittest.mock import AsyncMock
import pytest
from retrieval_api.instant.search import run_instant


@pytest.mark.asyncio
async def test_run_instant_returns_both_branches_on_success(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            assert dense_vector == [0.1, 0.2]  # Instant embeds the raw query for true dense ANN
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="tax exemption",
        milvus_sparse_enabled=True,
    )

    assert result["es"] == [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]
    assert result["es_error"] is None
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}
    assert result["milvus_sparse"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}
    assert result["milvus_error"] is None
    gateway.embed.assert_awaited_once_with(role="query_embed", text="tax exemption")


@pytest.mark.asyncio
async def test_run_instant_applies_elbow_cutoff_to_es_and_milvus_results(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
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

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", milvus_sparse_enabled=True,
    )

    assert result["es"] == [{"doc_id": "d1", "score": 10.0}]
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 10.0}]}
    assert result["milvus_sparse"] == {
        "ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 10.0}],
    }


@pytest.mark.asyncio
async def test_run_instant_keeps_flat_score_distribution_uncapped(monkeypatch):
    import retrieval_api.instant.search as search_module

    flat_scores = [{"doc_id": f"d{i}", "score": 5.0} for i in range(12)]

    async def fake_raw_search(client, query, limit=20, boost=False):
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

    async def failing_raw_search(client, query, limit=20, boost=False):
        raise RuntimeError("ES down")

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", failing_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", milvus_sparse_enabled=True,
    )

    assert result["es"] is None
    assert result["es_error"] == "ES down"
    assert result["milvus"] == {"ruling": []}
    assert result["milvus_sparse"] == {"ruling": []}
    assert result["milvus_error"] is None


@pytest.mark.asyncio
async def test_run_instant_returns_partial_result_when_gateway_embed_fails(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
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
async def test_run_instant_forwards_boost_flag_to_raw_search(monkeypatch):
    import retrieval_api.instant.search as search_module

    seen_boost = []

    async def fake_raw_search(client, query, limit=20, boost=False):
        seen_boost.append(boost)
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", boost=True)

    assert seen_boost == [True]


@pytest.mark.asyncio
async def test_run_instant_defaults_boost_to_false(monkeypatch):
    import retrieval_api.instant.search as search_module

    seen_boost = []

    async def fake_raw_search(client, query, limit=20, boost=False):
        seen_boost.append(boost)
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert seen_boost == [False]


@pytest.mark.asyncio
async def test_run_instant_returns_reranked_list_when_rrf_flag_set(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2, "heading": "h1", "subheading": "s1"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", rrf=True)

    # es/milvus keys must survive into the fuse branch too - ws.py reads
    # es_error/milvus_error unconditionally to build instant_ok.
    assert result["es_error"] is None
    assert result["milvus_error"] is None
    assert result["reranked_error"] is None
    # rrf=True pulls in Milvus (d2) via fusion, not just ES's own d1.
    assert {row["doc_id"] for row in result["reranked"]} == {"d1", "d2"}


@pytest.mark.asyncio
async def test_run_instant_without_rrf_skips_fusion(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2, "heading": "h1", "subheading": "s1"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", rrf=False)

    assert result["reranked_error"] is None
    # rrf=False: candidates are ES's own top ranking only, Milvus (d2) isn't consulted.
    assert {row["doc_id"] for row in result["reranked"]} == {"d1"}


@pytest.mark.asyncio
async def test_run_instant_skips_fusion_when_es_branch_failed(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def failing_raw_search(client, query, limit=20, boost=False):
        raise RuntimeError("ES down")

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", failing_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["reranked"] == []
    assert result["reranked_error"] == "ES down"


@pytest.mark.asyncio
async def test_run_instant_emits_es_and_milvus_trace_steps(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
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

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", on_step=on_step,
        milvus_sparse_enabled=True,
    )

    # es_search runs on an independent branch (asyncio.gather with _run_milvus)
    # so its relative order vs. the milvus steps isn't guaranteed - only that
    # dense precedes sparse within the milvus branch, and query_analysis/classifier
    # (emitted synchronously before the gather) come first, in that order. Instant mode
    # always fuses (rrf/plan fallback) after that, emitting rrf_merge + instant_reranked.
    assert set(steps) == {
        "query_correction", "query_analysis", "classifier", "es_search", "milvus_dense", "milvus_sparse",
        "rrf_merge", "instant_reranked",
    }
    assert steps[0] == "query_correction"
    assert steps[1] == "query_analysis"
    assert steps[2] == "classifier"
    assert steps.index("milvus_dense") < steps.index("milvus_sparse")


@pytest.mark.asyncio
async def test_run_instant_sends_cleaned_text_to_milvus_but_raw_text_to_es(monkeypatch):
    """The exact bug this fixes: "section 55" and "what is section 55" must send Milvus
    dense/sparse identical, cleaned text - ES already handles this itself via chunk_query's
    own stopword-stripping inside _build_field_query's phrase-boost clauses, so ES keeps
    getting the raw sentence (its own pipeline needs the full text for the loose multi_match
    recall clause too)."""
    import retrieval_api.instant.search as search_module

    es_queries = []
    milvus_queries = []

    async def fake_raw_search(client, query, limit=20, boost=False):
        es_queries.append(query)
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        milvus_queries.append(sparse_query_text)
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="what is section 55",
        milvus_sparse_enabled=True,
    )

    assert es_queries == ["what is section 55"]
    assert milvus_queries == ["section 55", "section 55"]  # dense pass + sparse pass
    gateway.embed.assert_awaited_once_with(role="query_embed", text="section 55")


@pytest.mark.asyncio
async def test_run_instant_corrects_misspelled_court_before_search(monkeypatch):
    import retrieval_api.instant.search as search_module

    seen_queries = []

    async def fake_raw_search(client, query, limit=20, boost=False):
        seen_queries.append(query)
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_queries.append(sparse_query_text)
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="case from AHMDABAD tribunal",
    )

    assert "AHMEDABAD" in seen_queries[0].split()
    # Milvus gets the cleaned (stopword-stripped) text, not the raw corrected sentence -
    # "from" is a stopword and chunk_query drops it (see build_dense_sparse_query).
    gateway.embed.assert_awaited_once_with(role="query_embed", text="case AHMEDABAD tribunal")
    assert result["query_correction"]["original"] == "case from AHMDABAD tribunal"
    assert result["query_correction"]["corrections"] == [
        {"original": "AHMDABAD", "corrected": "AHMEDABAD", "score": result["query_correction"]["corrections"][0]["score"]},
    ]


@pytest.mark.asyncio
async def test_run_instant_emits_query_correction_trace_step_even_with_no_corrections(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = {}

    async def on_step(step, data):
        steps[step] = data

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", on_step=on_step)

    assert steps["query_correction"] == {"original": "q", "corrected": "q", "corrections": []}


@pytest.mark.asyncio
async def test_run_instant_emits_classifier_trace_step_with_label_confidence_and_plan(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("KEYWORD", 0.987))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = {}

    async def on_step(step, data):
        steps[step] = data

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="Section 52",
        auto_route=True, on_step=on_step,
    )

    assert steps["classifier"] == {
        "label": "KEYWORD", "confidence": 0.987, "auto_route": True,
        "plan": {"es": True, "milvus": False, "fuse": False},
    }


@pytest.mark.asyncio
async def test_run_instant_forwards_on_step_into_fusion_for_rrf_merge_step(monkeypatch):
    """run_instant passes on_step into rerank_instant_results, so with rrf=True
    the trace also picks up the rrf_merge step rerank.py emits."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2, "heading": "h1", "subheading": "s1"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = []

    async def on_step(step, data):
        steps.append(step)

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q",
        rrf=True, on_step=on_step,
    )

    assert "rrf_merge" in steps
    assert steps.index("rrf_merge") < steps.index("instant_reranked")


@pytest.mark.asyncio
async def test_run_instant_auto_route_keyword_skips_milvus(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2}]

    milvus_called = False

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        nonlocal milvus_called
        milvus_called = True
        return {}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("KEYWORD", 0.99))

    gateway = AsyncMock()
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="Section 52", auto_route=True,
    )

    assert result["es"] == [{"doc_id": "d1", "score": 4.2}]
    assert result["milvus"] is None
    assert not milvus_called
    gateway.embed.assert_not_called()


@pytest.mark.asyncio
async def test_run_instant_auto_route_intent_skips_es(monkeypatch):
    import retrieval_api.instant.search as search_module

    es_called = False

    async def fake_raw_search(client, query, limit=20, boost=False):
        nonlocal es_called
        es_called = True
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("INTENT", 0.95))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="how do I evade tax", auto_route=True,
    )

    assert result["es"] is None
    assert not es_called
    assert result["milvus"] is not None


@pytest.mark.asyncio
async def test_run_instant_auto_route_hybrid_forces_rrf_fusion(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("HYBRID", 0.97))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="where is section 52 applicable",
        auto_route=True, rrf=False,  # auto_route overrides the manual rrf=False when it's on
    )

    assert "reranked" in result
    assert any(row["doc_id"] == "d2" for row in result["reranked"])  # fused in from Milvus


@pytest.mark.asyncio
async def test_run_instant_auto_route_false_preserves_today_behavior(monkeypatch):
    """auto_route and rrf both default to False - both backends still run, and Instant
    mode always exposes a `reranked` list (an ES-only ranking here, since rrf=False falls
    back to the single source that ran) with no AI/cross-encoder call involved."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] is not None
    assert result["milvus"] is not None
    assert {row["doc_id"] for row in result["reranked"]} == {"d1"}


@pytest.mark.asyncio
async def test_run_instant_skips_native_milvus_sparse_pass_by_default(monkeypatch):
    """milvus_sparse_enabled defaults to False - Instant mode must not call hybrid_search
    for the native sparse (dense_vector=None) pass at all unless explicitly opted in.
    Instant has no ES sparse-fallback (that's AI-Mode-only), so disabling this leaves
    nothing else feeding the sparse side at all."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return []

    sparse_pass_called = False

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        nonlocal sparse_pass_called
        if dense_vector is None:
            sparse_pass_called = True
            return {}
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert not sparse_pass_called
    assert result["milvus_sparse"] == {}
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}


@pytest.mark.asyncio
async def test_run_instant_omits_milvus_sparse_trace_step_when_disabled(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False):
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = []

    async def on_step(step, data):
        steps.append(step)

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q", on_step=on_step)

    assert "milvus_dense" in steps
    assert "milvus_sparse" not in steps
