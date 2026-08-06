from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.synthesize import synthesize


@pytest.mark.asyncio
async def test_synthesize_uses_prefetched_citations_without_extra_lookup(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    fetch_calls = []

    async def fake_fetch_citations(client, doc_ids):
        fetch_calls.append(doc_ids)
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Final answer with citation.", None)

    result = await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {"masterinfo": {"court": "SC"}}},
    )

    assert fetch_calls == []  # already prefetched, no fallback lookup needed
    assert result == {
        "answer": "Final answer with citation.",
        "citations": {"d1": {"masterinfo": {"court": "SC"}}},
        "reasoning": None,
    }


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_on_demand_lookup_for_missing_doc(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        assert doc_ids == ["d2"]
        return {"d2": {"masterinfo": {"court": "HC"}}}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", None)

    result = await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "b", "doc_id": "d2", "text": "chunk text"}],
        citations={},
    )

    assert result["citations"] == {"d2": {"masterinfo": {"court": "HC"}}}


@pytest.mark.asyncio
async def test_synthesize_surfaces_reasoning_when_present(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    monkeypatch.setattr(module, "fetch_citations", AsyncMock(return_value={}))

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", "step by step reasoning...")

    result = await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}},
    )

    assert result["reasoning"] == "step by step reasoning..."


@pytest.mark.asyncio
async def test_synthesize_emits_synthesis_prompt_step(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Final answer.", None)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await synthesize(
        gateway, es_client=object(), query="what is cgst",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}}, on_step=on_step,
    )

    assert len(steps) == 1
    step, data = steps[0]
    assert step == "synthesis_prompt"
    assert "what is cgst" in data["prompt"]
    assert "[d1] chunk text" in data["prompt"]
