# packages/retrieval-api/tests/test_ws_integration.py
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module


def test_ws_search_sends_instant_then_ai_mode_events(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "final answer", "citations": {"d1": {}}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "tax exemption"})

        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "instant_result"
    assert first["es"] == [{"doc_id": "d1"}]
    assert second == {"type": "ai_mode_done", "answer": "final answer", "citations": {"d1": {}}}


def test_ws_search_streams_ai_mode_trace_steps_before_final_answer(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        await on_step("intent", {"query": query, "rewritten_query": "r", "intent": "x", "filters": {}})
        await on_step("filters_resolved", {"filters": {}, "doc_id_count": 0, "doc_id_sample": []})
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "trace": True})
        messages = []
        while True:
            msg = websocket.receive_json()
            messages.append(msg)
            if msg["type"] in ("ai_mode_done", "ai_mode_error"):
                break

    trace_1 = {
        "type": "ai_mode_trace", "step": "intent",
        "data": {"query": "q", "rewritten_query": "r", "intent": "x", "filters": {}},
    }
    trace_2 = {
        "type": "ai_mode_trace", "step": "filters_resolved",
        "data": {"filters": {}, "doc_id_count": 0, "doc_id_sample": []},
    }
    final = {"type": "ai_mode_done", "answer": "final answer", "citations": {}}

    # instant_result arrives at some point (no cross-stream ordering guarantee
    # with ai_mode's trace steps - both paths run concurrently).
    assert any(m["type"] == "instant_result" for m in messages)

    # trace steps preserve pipeline stage order, and both land before the
    # final answer.
    types_and_payloads = [m for m in messages if m["type"] in ("ai_mode_trace", "ai_mode_done")]
    assert types_and_payloads == [trace_1, trace_2, final]


@pytest.mark.asyncio
async def test_emit_trace_step_swallows_send_errors():
    from retrieval_api.ws import _emit_trace_step

    async def failing_send(payload):
        raise RuntimeError("connection closed")

    await _emit_trace_step(failing_send, "intent", {"foo": "bar"})  # must not raise


def test_ws_search_does_not_pass_on_step_when_trace_flag_is_absent(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    captured_on_step = "unset"

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        nonlocal captured_on_step
        captured_on_step = on_step
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})  # no "trace" field
        websocket.receive_json()
        websocket.receive_json()

    assert captured_on_step is None


def test_ws_search_passes_on_step_when_trace_flag_is_true(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    captured_on_step = "unset"

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        nonlocal captured_on_step
        captured_on_step = on_step
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "trace": True})
        websocket.receive_json()
        websocket.receive_json()

    assert callable(captured_on_step)


def test_ws_search_instant_mode_does_not_emit_trace_steps(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        raise AssertionError("ai_mode should not run in instant-only mode")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "instant"})
        only = websocket.receive_json()

    assert only["type"] == "instant_result"


def test_ws_search_sends_ai_mode_error_event_on_failure(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "gateway unreachable"}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})
        websocket.receive_json()  # instant_result
        second = websocket.receive_json()

    assert second == {"type": "ai_mode_error", "error": "gateway unreachable"}


def test_ws_search_still_answers_when_milvus_client_construction_fails(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        assert milvus_client is None
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": None, "milvus_error": "connection refused"}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        assert milvus_client is None
        return {"ok": False, "error": "connection refused"}

    def raise_milvus_unavailable(*_):
        raise ConnectionError("Milvus unavailable")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", raise_milvus_unavailable)
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "cgst"})
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first == {
        "type": "instant_result", "es": [{"doc_id": "d1"}], "es_error": None,
        "milvus": None, "milvus_error": "connection refused",
    }
    assert second == {"type": "ai_mode_error", "error": "connection refused"}


def test_ws_search_instant_mode_skips_ai_mode(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        raise AssertionError("ai_mode should not run in instant-only mode")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "instant"})
        only = websocket.receive_json()

    assert only["type"] == "instant_result"


def test_ws_search_ai_mode_only_skips_instant(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        raise AssertionError("instant should not run in ai_mode-only mode")

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "ai_mode"})
        only = websocket.receive_json()

    assert only == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
