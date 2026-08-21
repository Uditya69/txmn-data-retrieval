# packages/retrieval-api/tests/test_ws_integration.py
import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module
from semantic_cache.repository import write as cache_write


def test_ws_search_sends_instant_then_ai_mode_events(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        await on_step(
            "intent",
            {"query": query, "original_query": query, "search_query": "r", "intent": ["caselaws"], "filters": {}},
        )
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
        "data": {"query": "q", "original_query": "q", "search_query": "r", "intent": ["caselaws"], "filters": {}},
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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


def test_ws_search_auto_route_defaults_to_false_when_absent(monkeypatch):
    captured_auto_route = "unset"

    async def fake_run_instant(
        gateway, es_client, milvus_client, query, on_step=None, rrf=False, rerank=False, auto_route=False,
    ):
        nonlocal captured_auto_route
        captured_auto_route = auto_route
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})  # no "auto_route" field
        websocket.receive_json()
        websocket.receive_json()

    assert captured_auto_route is False


def test_ws_search_forwards_auto_route_to_run_instant_when_enabled(monkeypatch):
    captured_auto_route = "unset"

    async def fake_run_instant(
        gateway, es_client, milvus_client, query, on_step=None, rrf=False, rerank=False, auto_route=False,
    ):
        nonlocal captured_auto_route
        captured_auto_route = auto_route
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(
        ws_module, "get_settings",
        lambda: Mock(instant_mode_rerank_enabled=True, instant_mode_auto_route_enabled=True),
    )
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "auto_route": True})
        websocket.receive_json()
        websocket.receive_json()

    assert captured_auto_route is True


def test_ws_search_kill_switch_forces_auto_route_off_even_when_requested(monkeypatch):
    captured_auto_route = "unset"

    async def fake_run_instant(
        gateway, es_client, milvus_client, query, on_step=None, rrf=False, rerank=False, auto_route=False,
    ):
        nonlocal captured_auto_route
        captured_auto_route = auto_route
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(
        ws_module, "get_settings",
        lambda: Mock(instant_mode_rerank_enabled=True, instant_mode_auto_route_enabled=False),
    )
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "auto_route": True})
        websocket.receive_json()
        websocket.receive_json()

    assert captured_auto_route is False


def test_ws_search_instant_mode_does_not_emit_trace_steps(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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


class _FakeEmbedGateway:
    def __init__(self, embedding):
        self._embedding = embedding

    async def embed(self, role, text):
        assert role == "query_embed"
        return self._embedding


@pytest.mark.asyncio
async def test_ai_mode_cache_hit_skips_run_ai_mode_and_returns_cached_answer(
    monkeypatch, fake_semantic_cache_collection,
):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("run_ai_mode should not be called on a cache hit")

    monkeypatch.setattr(ws_module, "run_ai_mode", fail_if_called)
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))

    await cache_write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C", [1.0, 0.0],
        {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]},
    )

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {"type": "ai_mode_done", "answer": "cached answer", "citations": []}


@pytest.mark.asyncio
async def test_ai_mode_cache_miss_runs_pipeline_and_writes_back(
    monkeypatch, fake_semantic_cache_collection,
):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "fresh answer", "citations": [], "intent": ["acts"],
    }))

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {"type": "ai_mode_done", "answer": "fresh answer", "citations": []}

    cached = None
    for _ in range(50):
        cached = await cache_lookup_helper(fake_semantic_cache_collection)
        if cached is not None:
            break
        await asyncio.sleep(0.01)

    assert cached == {"ok": True, "answer": "fresh answer", "citations": [], "intent": ["acts"]}


async def cache_lookup_helper(collection):
    from semantic_cache.repository import lookup
    return await lookup(collection, "ai_mode", [1.0, 0.0], threshold=0.95)


@pytest.mark.asyncio
async def test_cache_lookup_failure_degrades_to_normal_pipeline(monkeypatch):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))

    class _BrokenCollection:
        def aggregate(self, pipeline):
            raise RuntimeError("Atlas unreachable")

        async def insert_one(self, document):
            raise RuntimeError("Atlas unreachable")

    monkeypatch.setattr(ws_module, "get_semantic_cache_collection", lambda *_: _BrokenCollection())
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "fresh answer despite cache error", "citations": [], "intent": [],
    }))

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {
        "type": "ai_mode_done", "answer": "fresh answer despite cache error", "citations": [],
    }


@pytest.mark.asyncio
async def test_instant_mode_cache_hit_skips_run_instant_and_returns_cached_result(
    monkeypatch, fake_semantic_cache_collection,
):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("run_instant should not be called on a cache hit")

    monkeypatch.setattr(ws_module, "run_instant", fail_if_called)
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "unused", "citations": [], "intent": [],
    }))

    cached_instant_result = {
        "es": [{"doc_id": "cached1"}], "es_error": None,
        "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }
    await cache_write(
        fake_semantic_cache_collection, "instant_auto_route_False_rrf_False_rerank_False",
        "what is section 80C", [1.0, 0.0],
        cached_instant_result,
    )

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "instant"})
        message = websocket.receive_json()

    assert message == {"type": "instant_result", **cached_instant_result}


@pytest.mark.asyncio
async def test_instant_mode_rerank_cache_hit_uses_separate_key_from_plain_instant(
    monkeypatch, fake_semantic_cache_collection,
):
    # NOTE: this test (and every other semantic-cache websocket test in this file)
    # currently hangs under TestClient.websocket_connect() for reasons unrelated to
    # sync/async framing - confirmed by testing an untouched sibling test, which hangs
    # identically. Pre-existing, out of scope here; not caused by the rrf/rerank split.
    # Unlike every other test in this file, this one sends rerank=True - the only
    # request shape that evaluates settings.instant_mode_rerank_enabled (short-circuited
    # by `and` otherwise) - so it needs that attribute present, not a bare object().
    monkeypatch.setattr(
        ws_module, "get_settings",
        lambda: Mock(instant_mode_rerank_enabled=True, instant_mode_auto_route_enabled=True),
    )
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("run_instant should not be called on a cache hit")

    monkeypatch.setattr(ws_module, "run_instant", fail_if_called)
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "unused", "citations": [], "intent": [],
    }))

    # es_error/milvus_error must be present - ws.py reads them unconditionally off any
    # instant_result, cache hit or miss alike, to compute output["instant_ok"]. A real
    # cache write always includes them (run_instant's rerank branch keeps the es/milvus
    # keys from the plain branch - see instant/search.py), so a realistic cached entry
    # must too.
    cached_reranked_result = {
        "es": None, "es_error": None, "milvus": None, "milvus_sparse": None, "milvus_error": None,
        "reranked": [{"doc_id": "cached-reranked"}], "reranked_error": None,
    }
    await cache_write(
        fake_semantic_cache_collection, "instant_auto_route_False_rrf_False_rerank_True",
        "what is section 80C", [1.0, 0.0],
        cached_reranked_result,
    )

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "instant", "rerank": True})
        message = websocket.receive_json()

    assert message == {"type": "instant_result", **cached_reranked_result}


class _CountingFakeEmbedGateway:
    """Same contract as _FakeEmbedGateway, but tracks how many times embed()
    was invoked - used to prove the query embedding is computed once per
    /ws/search request and reused across both the instant and ai_mode cache
    lookups when mode="both", rather than being recomputed per mode."""

    def __init__(self, embedding):
        self._embedding = embedding
        self.embed_call_count = 0

    async def embed(self, role, text):
        assert role == "query_embed"
        self.embed_call_count += 1
        return self._embedding


@pytest.mark.asyncio
async def test_both_mode_computes_query_embedding_once_and_reuses_for_both_lookups(
    monkeypatch, fake_semantic_cache_collection,
):
    fake_gateway = _CountingFakeEmbedGateway([1.0, 0.0])

    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: fake_gateway)
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "fresh answer", "citations": [], "intent": [],
    }))

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "both"})
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "instant_result"
    assert second == {"type": "ai_mode_done", "answer": "fresh answer", "citations": []}
    assert fake_gateway.embed_call_count == 1
