from unittest.mock import AsyncMock

import pytest

from retrieval_api.ai_mode.pipeline import run_ai_mode


@pytest.mark.asyncio
async def test_run_ai_mode_success_path(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        return {"original_query": query, "search_query": "rewritten", "intent": ["caselaws"], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    received_intent = {}

    async def fake_retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent, on_step=None):
        received_intent["value"] = intent
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="original query")

    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}, "intent": ["caselaws"]}
    assert received_intent["value"] == ["caselaws"]


@pytest.mark.asyncio
async def test_run_ai_mode_passes_persona_context_to_synthesize_and_returns_intent(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        return {"original_query": query, "search_query": "rewritten", "intent": ["acts"], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    received_persona_context = {}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        received_persona_context["value"] = persona_context
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(
        gateway=object(), es_client=object(), milvus_client=object(), query="q",
        persona_context="This user frequently asks about acts.",
    )

    assert received_persona_context["value"] == "This user frequently asks about acts."
    assert result["intent"] == ["acts"]


@pytest.mark.asyncio
async def test_run_ai_mode_returns_error_on_any_stage_failure(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def failing_extract_intent(gateway, query, on_step=None, persona_context=""):
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
                "bool": {"must": [{"match": {"otherinfo.partyname.name": {"query": "Reliance Industries", "operator": "and"}}}]}
            }
            return {"hits": {"hits": [{"_source": {"id": "d1"}}]}}

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        return {
            "original_query": query,
            "search_query": "rewritten",
            "intent": ["caselaws"],
            "filters": {"party": "Reliance Industries"},
        }

    async def fake_retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent, on_step=None):
        assert doc_id_allowlist == ["d1"]
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(
        gateway=object(), es_client=FakeESClient(), milvus_client=object(), query="original query"
    )

    assert result["ok"] is True
    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}, "intent": ["caselaws"]}


@pytest.mark.asyncio
async def test_run_ai_mode_forwards_on_step_to_every_stage(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    received_on_steps = []

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        received_on_steps.append(("extract_intent", on_step))
        return {"original_query": query, "search_query": "rewritten", "intent": ["caselaws"], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        received_on_steps.append(("resolve_allowlist", on_step))
        return None

    async def fake_retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent, on_step=None):
        received_on_steps.append(("retrieve", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        received_on_steps.append(("rerank_and_prefetch", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
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
async def test_run_ai_mode_forwards_persona_context_to_extract_intent_and_synthesize(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    received = {}

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        received["intent_persona"] = persona_context
        return {"original_query": query, "search_query": "rewritten", "intent": [], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent, on_step=None):
        return []

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        return [], {}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        received["synth_persona"] = persona_context
        return {"answer": "a", "citations": {}}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    await run_ai_mode(
        gateway=object(), es_client=object(), milvus_client=object(), query="q",
        persona_context="persona-hint",
    )

    assert received["intent_persona"] == "persona-hint"
    assert received["synth_persona"] == "persona-hint"


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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {}

    monkeypatch.setattr(filter_resolve_module, "resolve_doc_id_allowlist", fake_resolve_doc_id_allowlist)
    monkeypatch.setattr(retrieve_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieve_module, "sparse_fallback_search", fake_sparse_fallback_search)
    monkeypatch.setattr(citations_module, "fetch_citations", fake_fetch_citations)
    monkeypatch.setattr(synthesize_module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    async def fake_chat(role, messages, model=None, response_format=None, temperature=None):
        if role == "slm":
            return '{"search_query": "rewritten query", "intent": ["caselaws"], "filters": {}}'
        raise AssertionError(f"unexpected chat role: {role}")

    gateway.chat.side_effect = fake_chat
    gateway.chat_with_reasoning.return_value = ("final synthesized answer", None)
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

    rrf_step = next(data for step, data in collected if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 1.0
    assert rrf_step["sparse_weight"] == 1.0
