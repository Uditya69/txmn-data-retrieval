# packages/retrieval-api/tests/test_ws_integration.py
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module


def test_ws_search_sends_instant_then_ai_mode_events(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query):
        return {"ok": True, "answer": "final answer", "citations": {"d1": {}}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "tax exemption"})

        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "instant_result"
    assert first["es"] == [{"doc_id": "d1"}]
    assert second == {"type": "ai_mode_done", "answer": "final answer", "citations": {"d1": {}}}


def test_ws_search_sends_ai_mode_error_event_on_failure(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query):
        return {"ok": False, "error": "gateway unreachable"}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})
        websocket.receive_json()  # instant_result
        second = websocket.receive_json()

    assert second == {"type": "ai_mode_error", "error": "gateway unreachable"}
