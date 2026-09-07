# packages/retrieval-api/tests/test_instant_search.py
from unittest.mock import AsyncMock
import pytest
from retrieval_api.instant.rerank import _RERANK_INSTRUCTION
from retrieval_api.instant.search import run_instant


@pytest.mark.asyncio
async def test_run_instant_returns_both_branches_on_success(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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
    # KEYWORD skips the ES elbow (see test_run_instant_keyword_label_skips_elbow_cutoff_on_es_results) -
    # pin a non-KEYWORD label here so this test keeps exercising the elbow mechanism itself.
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("HYBRID", 0.9))

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
async def test_run_instant_keyword_label_skips_elbow_cutoff_on_es_results(monkeypatch):
    """KEYWORD-shape queries are precise anchor lookups whose ES results span steep
    boost-tier gaps (heading:100000 vs fullcontent:1, see _PHRASE_BOOSTS in es_client.py) -
    the elbow's ratio test misreads a legit lower-tier match as a cliff and prunes it.
    KEYWORD skips the elbow entirely and returns ES's own top-_ES_LIMIT ranking as-is."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        # steep drop after the first hit - same shape the elbow would normally prune to 1,
        # but all three are genuine tiered-boost matches that should survive for KEYWORD.
        return [
            {"doc_id": "d1", "score": 100000.0},
            {"doc_id": "d2", "score": 50000.0},
            {"doc_id": "d3", "score": 1.0},
        ]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("KEYWORD", 0.99))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="Rule 6")

    assert result["es"] == [
        {"doc_id": "d1", "score": 100000.0},
        {"doc_id": "d2", "score": 50000.0},
        {"doc_id": "d3", "score": 1.0},
    ]


@pytest.mark.asyncio
async def test_run_instant_non_keyword_label_still_applies_elbow_cutoff(monkeypatch):
    """HYBRID/INTENT queries lean on dense fusion, not a raw ES ranking shown as-is -
    the elbow protection this fix removes for KEYWORD must stay intact for them."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [
            {"doc_id": "d1", "score": 10.0},
            {"doc_id": "d2", "score": 1.0},
            {"doc_id": "d3", "score": 0.1},
        ]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("HYBRID", 0.9))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["es"] == [{"doc_id": "d1", "score": 10.0}]


@pytest.mark.asyncio
async def test_run_instant_keeps_flat_score_distribution_uncapped(monkeypatch):
    import retrieval_api.instant.search as search_module

    flat_scores = [{"doc_id": f"d{i}", "score": 5.0} for i in range(12)]

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def failing_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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
async def test_run_instant_defaults_boost_to_true_and_repotaxmannapi(monkeypatch):
    """2026-09-02: explicit user override (see raw_search's own docstring) - a caller of
    run_instant that omits boost/boost_source now gets the byte-exact ported .NET
    multiply-mode formula on by default, matching raw_search's own new default."""
    import retrieval_api.instant.search as search_module

    seen_boost = []
    seen_boost_source = []

    async def fake_raw_search(client, query, limit=20, boost=True, boost_source="repotaxmannapi", page=1, page_size=None):
        seen_boost.append(boost)
        seen_boost_source.append(boost_source)
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert seen_boost == [True]
    assert seen_boost_source == ["repotaxmannapi"]


@pytest.mark.asyncio
async def test_run_instant_returns_reranked_list_when_rrf_flag_set(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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
async def test_run_instant_rerank_true_calls_cross_encoder_and_unions_es_and_milvus(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [{"doc_id": "d1", "score": 4.2, "heading": "h1", "subheading": "s1", "text": "full text for d1"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d2::ruling::0", "doc_id": "d2", "text": "full text for d2", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    # d1 (ES-only, no chunk_id) triggers rerank.py's own Milvus-chunk-enrichment lookup -
    # a separately-bound import (see CLAUDE.md's monkeypatch+direct-import gotcha), so it
    # needs its own patch. Returns nothing found, so d1 keeps its original ES text below.
    import retrieval_api.instant.rerank as rerank_module
    monkeypatch.setattr(rerank_module, "hybrid_search", AsyncMock(return_value={}))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    gateway.rerank.return_value = [0.8, 0.9]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", rerank=True,
    )

    assert result["reranked_error"] is None
    # rerank=True: candidate pool is a plain union of ES + Milvus dense, no rrf/plan-based
    # fusion involved - both d1 and d2 reach the cross-encoder.
    assert {row["doc_id"] for row in result["reranked"]} == {"d1", "d2"}
    assert all("rerank_score" in row for row in result["reranked"])
    gateway.rerank.assert_awaited_once_with(
        role="reranker", query="q", documents=["full text for d1", "full text for d2"], model=None,
        instruction=_RERANK_INSTRUCTION,
    )


@pytest.mark.asyncio
async def test_run_instant_defaults_rerank_to_false(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="q")

    assert result["reranked_error"] is None
    gateway.rerank.assert_not_called()
    assert {row["doc_id"] for row in result["reranked"]} == {"d1"}


@pytest.mark.asyncio
async def test_run_instant_skips_fusion_when_es_branch_failed(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def failing_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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
async def test_run_instant_exact_phrase_query_skips_classifier_and_milvus(monkeypatch):
    """A search-bar query that is ENTIRELY one double-quoted phrase must never reach the ML
    shape classifier or Milvus - regression guard for a real production bug: a long quoted
    headnote-text phrase got mislabeled HYBRID (confidence ~0.50, below threshold), routed to
    both ES and Milvus with fuse=True, and since ES's own exact-phrase query structure
    correctly requires a genuine phrase match (returning zero hits when there isn't one),
    Milvus's semantic search - which has no such requirement - silently filled the entire
    result list with topically-similar but non-matching cases, defeating the whole point of
    quoting an exact phrase. `effective_label_with_confidence` monkeypatched to raise proves
    it's never called at all for this query shape, not just that its result gets overridden -
    this must be a structural bypass, not routing logic downstream of a (possibly wrong)
    classifier call."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [{"doc_id": "d1", "score": 4.2}]

    def classifier_should_not_be_called(query):
        raise AssertionError("classifier must not run for a whole-query exact phrase")

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "effective_label_with_confidence", classifier_should_not_be_called)
    monkeypatch.setattr(search_module, "routing_plan", classifier_should_not_be_called)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    gateway = AsyncMock()
    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(),
        query='"record and correct application of principle in law"', on_step=on_step,
    )

    assert result["milvus"] is None
    gateway.embed.assert_not_called()
    classifier_step = next(data for step, data in steps if step == "classifier")
    assert classifier_step["plan"] == {"es": True, "milvus": False, "fuse": False}
    assert classifier_step["confidence"] == 1.0


@pytest.mark.asyncio
async def test_run_instant_auto_route_intent_skips_es(monkeypatch):
    import retrieval_api.instant.search as search_module

    es_called = False

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
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


@pytest.mark.asyncio
async def test_run_instant_omits_grouped_es_when_boost_source_is_sum(monkeypatch):
    """Default path (boost_source="sum") must never call raw_search_grouped at all -
    grouping is exclusively part of the repotaxmannapi replica, no sum-mode equivalent."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fail_raw_search_grouped(*args, **kwargs):
        raise AssertionError("raw_search_grouped must not be called under boost_source='sum'")

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "raw_search_grouped", fail_raw_search_grouped)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", boost=True, boost_source="sum",
    )

    assert result["grouped_es"] is None
    assert result["grouped_es_error"] is None


@pytest.mark.asyncio
async def test_run_instant_returns_grouped_es_when_boost_source_is_repotaxmannapi(monkeypatch):
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fake_raw_search_grouped(client, query, limit_per_group=5):
        return {"ACT": [{"doc_id": "a1", "score": 9.0, "heading": "H", "subheading": "S"}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "raw_search_grouped", fake_raw_search_grouped)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="SECTION 52",
        boost=True, boost_source="repotaxmannapi", on_step=on_step,
    )

    assert result["grouped_es"] == {"ACT": [{"doc_id": "a1", "score": 9.0, "heading": "H", "subheading": "S"}]}
    assert result["grouped_es_error"] is None
    assert "es_grouped" in steps


@pytest.mark.asyncio
async def test_run_instant_grouped_es_failure_degrades_gracefully(monkeypatch):
    """A grouped-query ES failure must not crash the whole search or block the flat
    es/milvus/reranked results - same fail-open pattern as _run_es/_run_milvus."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def failing_raw_search_grouped(client, query, limit_per_group=5):
        raise RuntimeError("es down")

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(search_module, "raw_search_grouped", failing_raw_search_grouped)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q",
        boost=True, boost_source="repotaxmannapi",
    )

    assert result["grouped_es"] is None
    assert result["grouped_es_error"] == "es down"
    assert result["es"] == [{"doc_id": "d1", "score": 4.2}]
    assert result["es_error"] is None


@pytest.mark.asyncio
async def test_run_instant_passes_page_and_page_size_through_to_raw_search(monkeypatch):
    import retrieval_api.instant.search as search_module

    captured = {}

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        captured["page"] = page
        captured["page_size"] = page_size
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="section 80HH",
        page=2, page_size=10,
    )

    assert captured == {"page": 2, "page_size": 10}


@pytest.mark.asyncio
async def test_run_instant_defaults_page_to_1_and_page_size_to_none(monkeypatch):
    import retrieval_api.instant.search as search_module

    captured = {}

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        captured["page"] = page
        captured["page_size"] = page_size
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="section 80HH")

    assert captured == {"page": 1, "page_size": None}


@pytest.mark.asyncio
async def test_run_instant_skips_elbow_cutoff_on_es_results_when_page_size_is_set(monkeypatch):
    """The elbow's ratio test assumes a flat top-N window starting at rank 1 - on a
    server-paged (page_size is not None) request it would evaluate over an arbitrary
    mid-corpus score window and prune non-deterministically w.r.t. page size. Skip it
    entirely for any paged request, same as the existing KEYWORD-label skip_cutoff path."""
    import retrieval_api.instant.search as search_module

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        # steep drop after the first hit - would normally get pruned to just d1 by the
        # elbow, but must survive untouched here because page_size is set.
        return [
            {"doc_id": "d1", "score": 10.0},
            {"doc_id": "d2", "score": 1.0},
            {"doc_id": "d3", "score": 0.1},
        ]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)
    # Non-KEYWORD label, so skip_cutoff itself is False - page_size alone must trigger the skip.
    monkeypatch.setattr(search_module, "effective_label_with_confidence", lambda query: ("HYBRID", 0.9))

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    result = await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="q", page=2, page_size=10,
    )

    assert result["es"] == [
        {"doc_id": "d1", "score": 10.0},
        {"doc_id": "d2", "score": 1.0},
        {"doc_id": "d3", "score": 0.1},
    ]
