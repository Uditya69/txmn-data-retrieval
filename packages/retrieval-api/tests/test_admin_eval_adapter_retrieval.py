from unittest.mock import AsyncMock, Mock

import pytest

import retrieval_api.admin_eval.adapters.retrieval as adapter


def _cases():
    return [
        {"id": "Q01", "class": "direct", "query": "capital gains exemption section 54F", "gold_doc_ids": ["d1"], "expected_collections": [], "pass_at": 5},
        {"id": "Q02", "class": "direct", "query": "GST refund on export", "gold_doc_ids": ["d2"], "expected_collections": [], "pass_at": 5},
    ]


@pytest.mark.asyncio
async def test_run_uses_reranker_rank_for_pass_fail(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())
    monkeypatch.setattr(adapter, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())
    fake_es = AsyncMock()
    fake_milvus = Mock()
    monkeypatch.setattr(adapter, "get_es_client", lambda settings: fake_es)
    monkeypatch.setattr(adapter, "get_milvus_client", lambda settings: fake_milvus)

    async def fake_evaluate_case(case, gateway, es_client, milvus_client, **kwargs):
        rank = 2 if case["id"] == "Q01" else None  # Q02 misses (rank None)
        return {"ranks": {"reranker": rank}, "errors": {}, "timings_ms": {}}

    monkeypatch.setattr(adapter, "evaluate_case", fake_evaluate_case)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]

    assert case_events[0]["status"] == "pass"   # rank 2 <= pass_at 5
    assert case_events[1]["status"] == "fail"   # rank None
    assert events[-1] == {"type": "done", "summary": {"total": 2, "passed": 1}}
    fake_es.close.assert_awaited_once()
    fake_milvus.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_closes_clients_even_on_case_error(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases()[:1])
    monkeypatch.setattr(adapter, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())
    fake_es = AsyncMock()
    fake_milvus = Mock()
    monkeypatch.setattr(adapter, "get_es_client", lambda settings: fake_es)
    monkeypatch.setattr(adapter, "get_milvus_client", lambda settings: fake_milvus)

    async def failing_evaluate_case(case, gateway, es_client, milvus_client, **kwargs):
        raise RuntimeError("es unreachable")

    monkeypatch.setattr(adapter, "evaluate_case", failing_evaluate_case)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "error"
    fake_es.close.assert_awaited_once()
    fake_milvus.close.assert_called_once()
