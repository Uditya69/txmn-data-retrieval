# packages/retrieval-api/tests/test_ws_integration.py
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module


def test_ws_search_sends_instant_then_ai_mode_events(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
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


def test_ws_agent_sends_trace_then_done(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        if on_step:
            import asyncio
            asyncio.get_event_loop()
        return {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        done = websocket.receive_json()

    assert done == {"type": "agent_done", "answer": "See [d1].", "doc_ids": ["d1"]}


def test_ws_agent_sends_unverifiable_when_citations_fail(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["d999"]}

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        message = websocket.receive_json()

    assert message == {"type": "agent_unverifiable", "invalid_doc_ids": ["d999"]}


def test_ws_agent_runs_real_pipeline_and_streams_tool_trace(monkeypatch):
    """Exercises the REAL run_agentic_search -> run_agent_loop -> dispatch_tool_call
    chain end to end (only dispatch_tool_call's lowest layer is mocked), so a
    mismatch between the real implementations of these pieces would be caught
    here - unlike other tests in this file, which mock run_agentic_search
    itself and therefore never call the real on_step trace callback."""
    import json

    import agents.loop as loop_module

    call_count = {"n": 0}

    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "content": None,
                    "reasoning": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_es", "arguments": json.dumps({"query": "gst rate"})},
                        }
                    ],
                }
            return {"content": "The rate is confirmed. [d1]", "reasoning": None, "tool_calls": None}

    async def fake_dispatch_tool_call(name, arguments, *, gateway, es_client, milvus_client):
        assert name == "search_es"
        return {"rows": [{"doc_id": "d1", "score": 1.0, "text": "some case text"}]}

    monkeypatch.setattr(loop_module, "dispatch_tool_call", fake_dispatch_tool_call)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: FakeGateway())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})

        messages = []
        while True:
            message = websocket.receive_json()
            messages.append(message)
            if message["type"] in ("agent_done", "agent_unverifiable", "agent_error"):
                break

    trace_steps = [m for m in messages if m["type"] == "ai_mode_trace"]
    tool_call_steps = [m for m in trace_steps if m["step"] == "agent_tool_call"]
    tool_result_steps = [m for m in trace_steps if m["step"] == "agent_tool_result"]
    assert len(tool_call_steps) == 1
    assert tool_call_steps[0]["data"] == {"name": "search_es", "arguments": {"query": "gst rate"}}
    assert len(tool_result_steps) == 1
    assert tool_result_steps[0]["data"]["result"]["rows"][0]["doc_id"] == "d1"

    done = messages[-1]
    assert done == {"type": "agent_done", "answer": "The rate is confirmed. [d1]", "doc_ids": ["d1"]}


def test_ws_agent_sends_error_on_pipeline_exception(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        message = websocket.receive_json()

    assert message == {"type": "agent_error", "error": "RuntimeError: gateway down"}
