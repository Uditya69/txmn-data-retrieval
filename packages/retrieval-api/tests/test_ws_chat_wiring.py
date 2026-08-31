import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
from retrieval_api.main import app
import retrieval_api.ws as ws_module


def _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection, fake_retrieval_traces_collection=None):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, **_kwargs):
        return {
            "es": [{"doc_id": "d1", "score": 1.0}], "es_error": None,
            "milvus": {}, "milvus_error": None, "reranked": [{"doc_id": "d1", "score": 1.0}],
        }

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
    monkeypatch.setattr(ws_module, "get_persona_events_collection", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_persona_topics_collection", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_current_snapshot", AsyncMock(return_value=[]))
    monkeypatch.setattr(ws_module, "migrate_legacy_persona", AsyncMock(return_value=False))
    monkeypatch.setattr(ws_module, "record_persona_signal", AsyncMock())

    monkeypatch.setattr(ws_module, "get_chat_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_chat_mongo_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_conversations_collection", lambda *_: fake_conversations_collection)
    if fake_retrieval_traces_collection is not None:
        monkeypatch.setattr(ws_module, "get_retrieval_traces_collection", lambda *_: fake_retrieval_traces_collection)


def test_ws_search_logged_in_user_persists_conversation_turn(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {"d1": {"heading": "Section 80HH"}}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    token = create_access_token("user-123", get_auth_settings())
    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({
            "query": "gst rate", "mode": "ai_mode", "access_token": token, "conversation_id": "conv-1",
        })
        response = websocket.receive_json()

    assert response == {
        "type": "ai_mode_done", "answer": "final answer", "citations": {"d1": {"heading": "Section 80HH"}},
    }

    from chat.repository import get_conversation
    import asyncio

    for _ in range(50):
        stored = asyncio.run(get_conversation(fake_conversations_collection, "conv-1", "user-123"))
        if stored is not None:
            break
        time.sleep(0.01)

    # The stored assistant message must carry the same citations the client
    # saw live - otherwise reopening this conversation renders the answer
    # with no citation strip and no clickable [n] markers (see
    # useConversations.ts's hydrateStoredMessages).
    assistant_message = next(m for m in stored["messages"] if m["role"] == "assistant")
    assert assistant_message["citations"] == {"d1": {"heading": "Section 80HH"}}

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


def test_ws_search_logged_in_user_persists_instant_and_ai_mode_retrieval_traces(
    monkeypatch, fake_conversations_collection, fake_retrieval_traces_collection,
):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        if on_step is not None:
            await on_step("ai_rrf_merge", {"candidate_count": 3, "top_candidates": [{"doc_id": "d1"}]})
            await on_step("rerank", {"reranked": True, "top_chunks": [{"doc_id": "d1", "rerank_score": 0.9}]})
        return {"ok": True, "answer": "final answer", "citations": {"d1": {}}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection, fake_retrieval_traces_collection)

    token = create_access_token("user-123", get_auth_settings())
    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({
            "query": "gst rate", "mode": "both", "access_token": token, "conversation_id": "conv-1",
        })
        websocket.receive_json()  # instant_result
        websocket.receive_json()  # ai_mode_done

    for _ in range(50):
        if len(fake_retrieval_traces_collection.documents) >= 2:
            break
        time.sleep(0.01)

    docs_by_mode = {doc["mode"]: doc for doc in fake_retrieval_traces_collection.documents}
    assert set(docs_by_mode) == {"instant", "ai_mode"}

    instant_doc = docs_by_mode["instant"]
    assert instant_doc["conversation_id"] == "conv-1"
    assert instant_doc["user_id"] == "user-123"
    assert instant_doc["instant"] == {"doc_ids": ["d1"]}

    ai_mode_doc = docs_by_mode["ai_mode"]
    assert ai_mode_doc["ai_mode"]["rrf_candidates"] == {"candidate_count": 3, "top_candidates": [{"doc_id": "d1"}]}
    assert ai_mode_doc["ai_mode"]["reranked_chunks"] == {"reranked": True, "top_chunks": [{"doc_id": "d1", "rerank_score": 0.9}]}
    assert ai_mode_doc["ai_mode"]["citations"] == {"d1": {}}
    assert ai_mode_doc["ai_mode"]["intent"] == ["caselaws"]


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
