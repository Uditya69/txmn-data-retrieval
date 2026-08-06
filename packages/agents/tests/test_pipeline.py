import pytest

from agents.pipeline import MAX_CITATION_RETRIES, run_agentic_search


@pytest.mark.asyncio
async def test_pipeline_returns_ok_when_first_answer_is_fully_cited(monkeypatch):
    import agents.pipeline as pipeline_module

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        return {"answer": "See [d1].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)

    result = await run_agentic_search(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}


@pytest.mark.asyncio
async def test_pipeline_retries_on_invalid_citation_then_succeeds(monkeypatch):
    import agents.pipeline as pipeline_module

    calls = {"n": 0}

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"answer": "See [d999].", "seen_doc_ids": {"d1"}, "messages": messages}
        return {"answer": "See [d1].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await run_agentic_search(
        gateway=object(), es_client=object(), milvus_client=object(), query="q", on_step=on_step,
    )

    assert result == {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}
    assert calls["n"] == 2
    assert steps[0] == ("agent_citation_rejected", {"invalid_doc_ids": ["d999"], "attempt": 1})
    assert steps[1][0] == "agent_answer"


@pytest.mark.asyncio
async def test_pipeline_returns_unverifiable_after_max_retries(monkeypatch):
    import agents.pipeline as pipeline_module

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        return {"answer": "See [d999].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)
    calls = {"n": 0}
    original = pipeline_module.run_agent_loop

    async def counting_run_agent_loop(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_agent_loop", counting_run_agent_loop)

    result = await run_agentic_search(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["d999"]}
    assert calls["n"] == MAX_CITATION_RETRIES
