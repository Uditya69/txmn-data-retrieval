from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.retrieve import rrf_merge, retrieve


async def _no_es_hits(*args, **kwargs):
    return []


def test_rrf_merge_combines_and_ranks_by_reciprocal_rank():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"  # appears rank-1 sparse + rank-2 dense: highest combined score
    assert set(ids) == {"a", "b", "c"}
    assert merged[0]["rrf_score"] > merged[-1]["rrf_score"]


def test_rrf_merge_dedupes_by_chunk_id():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "a", "text": "A"}]

    merged = rrf_merge(dense, sparse)

    assert len(merged) == 1


def test_rrf_merge_default_weights_match_prior_unweighted_behavior():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_upweights_dense_list_over_sparse():
    # "a" is dense-rank-1-only; "c" is sparse-rank-1-only. Equal weight would
    # tie them (both contribute 1/(60+1)); upweighting dense must break the
    # tie in "a"'s favor.
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=1.5, sparse_weight=0.5)

    assert merged[0]["chunk_id"] == "a"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


def test_rrf_merge_upweights_sparse_list_over_dense():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=0.5, sparse_weight=1.5)

    assert merged[0]["chunk_id"] == "c"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


@pytest.mark.asyncio
async def test_retrieve_embeds_rewritten_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)

    result = await retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q", doc_id_allowlist=["d1"],
    )

    gateway.embed.assert_awaited_once_with(role="query_embed", text="q")
    assert result[0]["chunk_id"] == "a"


def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = collection_trace({"ruling": rows})

    assert trace == {
        "collections": [{
            "name": "ruling",
            "hit_count": 7,
            "top_hits": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "score": float(i), "text_preview": "x" * 200}
                for i in range(5)
            ],
        }]
    }


@pytest.mark.asyncio
async def test_retrieve_emits_dense_sparse_and_rrf_merge_steps(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, on_step=on_step,
    )

    assert steps == ["milvus_dense", "milvus_sparse", "es_boost", "rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}


@pytest.mark.asyncio
async def test_retrieve_resolves_conceptual_intent_to_dense_weighted_rrf(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, intent="conceptual",
    )

    # conceptual -> dense_weight=1.5, sparse_weight=0.5: the dense-only chunk
    # must outrank the sparse-only chunk despite both being rank-1 in their list.
    assert result[0]["chunk_id"] == "a"


@pytest.mark.asyncio
async def test_retrieve_resolves_citation_lookup_intent_to_sparse_weighted_rrf(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, intent="citation_lookup",
    )

    assert result[0]["chunk_id"] == "c"


@pytest.mark.asyncio
async def test_retrieve_defaults_to_neutral_weighting_for_unrecognized_intent(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)

    # Neither "unknown" (a real intent label) nor a totally unrecognized
    # string should raise or behave differently from each other - both must
    # resolve to neutral (1.0, 1.0) weighting.
    result_unknown = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, intent="unknown",
    )
    result_unrecognized = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, intent="not_a_real_label",
    )

    assert result_unknown[0]["rrf_score"] == result_unrecognized[0]["rrf_score"]


@pytest.mark.asyncio
async def test_retrieve_defaults_intent_param_to_unknown_when_omitted(monkeypatch):
    """Backward-compatibility: existing callers that don't pass `intent` at
    all (e.g. the eval harness, if any direct caller exists) must keep
    getting today's neutral weighting."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q", doc_id_allowlist=None,
    )

    assert result[0]["chunk_id"] == "a"


@pytest.mark.asyncio
async def test_retrieve_kill_switch_forces_neutral_weighting_regardless_of_intent(monkeypatch):
    """When intent_rrf_weighting_enabled is False, even an intent that would
    normally skew the merge (e.g. citation_lookup -> dense_weight=0.5,
    sparse_weight=1.5) must resolve to neutral (1.0, 1.0)."""
    import retrieval_api.ai_mode.retrieve as module

    class FakeSettings:
        intent_rrf_weighting_enabled = False
        ai_mode_es_boost_enabled = False

    monkeypatch.setattr(module, "get_settings", lambda: FakeSettings())

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q", doc_id_allowlist=None,
        intent="citation_lookup", on_step=on_step,
    )

    rrf_step = next(data for step, data in steps if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 1.0
    assert rrf_step["sparse_weight"] == 1.0


@pytest.mark.asyncio
async def test_retrieve_includes_resolved_weights_in_rrf_merge_trace_step(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", _no_es_hits)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q", doc_id_allowlist=None,
        intent="provision_lookup", on_step=on_step,
    )

    rrf_step = next(data for step, data in steps if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 0.5
    assert rrf_step["sparse_weight"] == 1.5




@pytest.mark.asyncio
async def test_retrieve_boosts_doc_ids_confirmed_by_es_top_hits(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    async def fake_raw_search(client, query, limit=10):
        return [{"doc_id": "d1", "score": 9.0}]

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q",
        doc_id_allowlist=None, on_step=on_step,
    )

    assert result[0]["doc_id"] == "d1"
    es_boost_step = next(data for step, data in steps if step == "es_boost")
    assert es_boost_step["es_doc_ids"] == ["d1"]
    assert es_boost_step["boosted_doc_ids"] == ["d1"]


@pytest.mark.asyncio
async def test_retrieve_skips_es_search_when_kill_switch_disabled(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    class FakeSettings:
        intent_rrf_weighting_enabled = True
        ai_mode_es_boost_enabled = False

    monkeypatch.setattr(module, "get_settings", lambda: FakeSettings())

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("raw_search should not be called when the kill switch is off")

    monkeypatch.setattr(module, "raw_search", fail_if_called)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), rewritten_query="q", doc_id_allowlist=None,
    )

    assert result[0]["chunk_id"] == "a"
