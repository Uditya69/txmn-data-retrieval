import pytest

from retrieval_api.ai_mode.pipeline import run_ai_mode


@pytest.mark.asyncio
async def test_run_ai_mode_success_path(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query):
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters):
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations):
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

    async def failing_extract_intent(gateway, query):
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
        async def search(self, index, query, size):
            assert query == {
                "bool": {"must": [{"match": {"masterinfo.partyname": "Reliance Industries"}}]}
            }
            return {"hits": {"hits": [{"_source": {"doc_id": "d1"}}]}}

    async def fake_extract_intent(gateway, query):
        return {
            "rewritten_query": "rewritten",
            "intent": "x",
            "filters": {"party": "Reliance Industries"},
        }

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist):
        assert doc_id_allowlist == ["d1"]
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations):
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
