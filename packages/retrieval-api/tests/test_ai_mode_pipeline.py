from unittest.mock import AsyncMock

import pytest

from retrieval_api.ai_mode.pipeline import run_ai_mode


@pytest.mark.asyncio
async def test_run_ai_mode_success_path(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None):
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="original query")

    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}}


@pytest.mark.asyncio
async def test_run_ai_mode_returns_error_on_any_stage_failure(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def failing_extract_intent(gateway, query, on_step=None):
        raise ValueError("SLM did not return valid JSON")

    monkeypatch.setattr(module, "extract_intent", failing_extract_intent)

    result = await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": False, "error": "SLM did not return valid JSON"}


@pytest.mark.asyncio
async def test_run_ai_mode_succeeds_with_party_only_filter(monkeypatch):
    """Regression test: a party-only filter dict from the SLM must not raise
    ValueError inside resolve_doc_id_allowlist and abort the whole AI Mode run.
    """
    import retrieval_api.ai_mode.pipeline as module

    class FakeESClient:
        index = "test_index"

        async def search(self, index, query, size):
            assert query == {
                "bool": {"must": [{"match": {"otherinfo.partyname.name": "Reliance Industries"}}]}
            }
            return {"hits": {"hits": [{"_source": {"id": "d1"}}]}}

    async def fake_extract_intent(gateway, query, on_step=None):
        return {
            "rewritten_query": "rewritten",
            "intent": "x",
            "filters": {"party": "Reliance Industries"},
        }

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        assert doc_id_allowlist == ["d1"]
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(
        gateway=object(), es_client=FakeESClient(), milvus_client=object(), query="original query"
    )

    assert result["ok"] is True
    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}}


@pytest.mark.asyncio
async def test_run_ai_mode_forwards_on_step_to_every_stage(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    received_on_steps = []

    async def fake_extract_intent(gateway, query, on_step=None):
        received_on_steps.append(("extract_intent", on_step))
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        received_on_steps.append(("resolve_allowlist", on_step))
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        received_on_steps.append(("retrieve", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        received_on_steps.append(("rerank_and_prefetch", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        received_on_steps.append(("synthesize", on_step))
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    async def on_step(step, data):
        pass

    await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="q", on_step=on_step)

    assert received_on_steps == [
        ("extract_intent", on_step),
        ("resolve_allowlist", on_step),
        ("retrieve", on_step),
        ("rerank_and_prefetch", on_step),
        ("synthesize", on_step),
    ]


@pytest.mark.asyncio
async def test_run_ai_mode_emits_all_seven_trace_steps_in_order_end_to_end(monkeypatch):
    """Genuine integration test: only the true I/O boundaries (gateway calls,
    hybrid_search, resolve_doc_id_allowlist, fetch_citations) are faked. All 5
    real stage functions and the real pipeline orchestration run, so this is
    the cheapest test that would catch a dropped emit, a renamed step, or a
    reordering - none of which the mocked-stage tests above can catch."""
    import retrieval_api.ai_mode.intent as intent_module
    import retrieval_api.ai_mode.filter_resolve as filter_resolve_module
    import retrieval_api.ai_mode.retrieve as retrieve_module
    import retrieval_api.ai_mode.citations as citations_module
    import retrieval_api.ai_mode.synthesize as synthesize_module

    async def fake_resolve_doc_id_allowlist(client, filters):
        return None

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    async def fake_fetch_citations(client, doc_ids):
        return {doc_id: {"title": doc_id} for doc_id in doc_ids}

    monkeypatch.setattr(filter_resolve_module, "resolve_doc_id_allowlist", fake_resolve_doc_id_allowlist)
    monkeypatch.setattr(retrieve_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(citations_module, "fetch_citations", fake_fetch_citations)
    monkeypatch.setattr(synthesize_module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()

    async def fake_chat(role, messages):
        if role == "slm":
            return '{"rewritten_query": "rewritten query", "intent": "case_lookup", "filters": {}}'
        if role == "synthesis":
            return "final synthesized answer"
        raise AssertionError(f"unexpected chat role: {role}")

    gateway.chat.side_effect = fake_chat
    gateway.embed.return_value = [0.1, 0.2]
    gateway.rerank.return_value = [0.5, 0.9]

    collected: list[tuple[str, dict]] = []

    async def collector(step, data):
        collected.append((step, data))

    result = await run_ai_mode(
        gateway=gateway, es_client=object(), milvus_client=object(), query="original query", on_step=collector
    )

    assert result["ok"] is True
    assert [step for step, _ in collected] == [
        "intent",
        "filters_resolved",
        "milvus_dense",
        "milvus_sparse",
        "rrf_merge",
        "rerank",
        "synthesis_prompt",
    ]
