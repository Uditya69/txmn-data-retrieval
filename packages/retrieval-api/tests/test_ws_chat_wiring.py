import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
from retrieval_api.main import app
import retrieval_api.ws as ws_module


def _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, **_kwargs):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(
        ws_module, "get_settings",
        lambda: Mock(instant_mode_auto_route_enabled=False, milvus_sparse_enabled=False, expose_reasoning=False),
    )
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_persona_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_personas_collection", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_persona", AsyncMock(return_value=None))
    monkeypatch.setattr(ws_module, "record_persona_signal", AsyncMock())

    monkeypatch.setattr(ws_module, "get_chat_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_chat_mongo_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_conversations_collection", lambda *_: fake_conversations_collection)


def test_ws_search_logged_in_user_persists_conversation_turn(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    token = create_access_token("user-123", get_auth_settings())
    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({
            "query": "gst rate", "mode": "ai_mode", "access_token": token, "conversation_id": "conv-1",
        })
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}

    from chat.repository import get_conversation
    import asyncio

    for _ in range(50):
        stored = asyncio.run(get_conversation(fake_conversations_collection, "conv-1", "user-123"))
        if stored is not None:
            break
        time.sleep(0.01)

    assert stored is not None
    assert stored["title"] == "gst rate"


def test_ws_search_guest_never_writes_conversation(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "conversation_id": "conv-1"})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert fake_conversations_collection.documents == {}


def test_ws_search_logged_in_user_without_conversation_id_does_not_crash(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    token = create_access_token("user-123", get_auth_settings())
    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert fake_conversations_collection.documents == {}
