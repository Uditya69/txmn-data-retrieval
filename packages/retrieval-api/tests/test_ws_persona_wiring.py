from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
from retrieval_api.main import app
import retrieval_api.ws as ws_module


def _patch_common(monkeypatch, fake_run_ai_mode):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, **_kwargs):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())
    # Never touch a real MongoDB connection - get_mongo_client/get_personas_collection
    # are stubbed to sentinels that are only ever passed through to get_persona,
    # which is itself stubbed below (per test).
    monkeypatch.setattr(ws_module, "get_persona_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_personas_collection", lambda *_: object())


def test_ws_search_logged_in_user_with_persona_reaches_run_ai_mode_with_rendered_context(monkeypatch):
    """A logged-in user (valid access_token) whose persona has real signal must
    have that persona rendered into a non-empty persona_context string, and that
    exact string must reach run_ai_mode - proving the seam from Mongo lookup ->
    render_persona_context -> run_ai_mode wiring actually works end to end
    (modulo Mongo itself, which is faked out)."""
    captured = {}

    async def fake_get_persona(personas_collection, user_id):
        assert user_id == "user-123"
        return {
            "user_id": "user-123",
            "category_affinity": {"caselaws": 0.9, "acts": 0.1},
            "expertise_level": "expert",
            "query_style": "precise-citation",
            "query_count": 20,
        }

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    record_calls = []

    async def fake_record_persona_signal(personas_collection, gateway, user_id, query, categories):
        record_calls.append((user_id, categories))

    monkeypatch.setattr(ws_module, "record_persona_signal", fake_record_persona_signal)

    auth_settings = get_auth_settings()
    token = create_access_token("user-123", auth_settings)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert captured["persona_context"] != ""
    assert "caselaws" in captured["persona_context"]
    assert "expert" in captured["persona_context"]
    assert "precise-citation" in captured["persona_context"]


def test_ws_search_logged_in_user_with_thin_persona_gets_empty_persona_context(monkeypatch):
    """A logged-in user (valid access_token) whose persona has real category_affinity/
    expertise_level/query_style signal but a query_count below the trust threshold
    (query_count >= 20 in render_persona_context) must still reach run_ai_mode with
    persona_context=="" - same as a guest. Proves the trust gate is actually wired
    end to end (Mongo lookup -> render_persona_context -> run_ai_mode), not just
    correct in render_persona_context's own unit tests."""
    captured = {}

    async def fake_get_persona(personas_collection, user_id):
        assert user_id == "user-123"
        return {
            "user_id": "user-123",
            "category_affinity": {"caselaws": 0.9, "acts": 0.1},
            "expertise_level": "expert",
            "query_style": "precise-citation",
            "query_count": 3,
        }

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    record_calls = []

    async def fake_record_persona_signal(personas_collection, gateway, user_id, query, categories):
        record_calls.append((user_id, categories))

    monkeypatch.setattr(ws_module, "record_persona_signal", fake_record_persona_signal)

    auth_settings = get_auth_settings()
    token = create_access_token("user-123", auth_settings)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert captured["persona_context"] == ""


def test_ws_search_guest_gets_empty_persona_context_and_no_persona_write(monkeypatch):
    """A guest (no access_token) must reach run_ai_mode with persona_context=="",
    and no persona-write background task may be scheduled - record_persona_signal
    must never be called for a guest, even on a successful AI Mode result."""
    captured = {}

    async def fake_get_persona(personas_collection, user_id):
        raise AssertionError("get_persona should never be called for a guest")

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    record_calls = []

    async def fake_record_persona_signal(personas_collection, gateway, user_id, query, categories):
        record_calls.append((user_id, categories))

    monkeypatch.setattr(ws_module, "record_persona_signal", fake_record_persona_signal)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode"})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert captured["persona_context"] == ""
    assert record_calls == []


def test_ws_search_logged_in_user_successful_ai_mode_schedules_persona_write(monkeypatch):
    """A logged-in user with a successful (ok: True) AI Mode result must have the
    persona-write path invoked with the correct user_id and the categories from
    the AI Mode result's "intent" key."""

    async def fake_get_persona(personas_collection, user_id):
        return None

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["acts", "rules"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    record_calls = []

    async def fake_record_persona_signal(personas_collection, gateway, user_id, query, categories):
        record_calls.append((user_id, categories))

    monkeypatch.setattr(ws_module, "record_persona_signal", fake_record_persona_signal)

    auth_settings = get_auth_settings()
    token = create_access_token("user-456", auth_settings)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "input tax credit", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}

    # The persona write is scheduled as a fire-and-forget background task - give
    # the event loop a beat to run it before asserting.
    import time
    for _ in range(50):
        if record_calls:
            break
        time.sleep(0.01)

    assert record_calls == [("user-456", ["acts", "rules"])]


def test_ws_search_accepts_access_token_field_without_crashing(monkeypatch):
    """A valid access_token must not crash the guest-mode (mode=instant) path -
    this doesn't require a live Mongo/ES/Milvus stack since mode=instant never
    touches persona lookup or AI Mode at all."""

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        raise AssertionError("ai_mode should not run in instant-only mode")

    async def fake_get_persona(personas_collection, user_id):
        return None

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    settings = get_auth_settings()
    token = create_access_token("user-123", settings)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/search") as websocket:
            websocket.send_json({"query": "test query", "mode": "instant", "access_token": token})
            response = websocket.receive_json()
            assert response["type"] == "instant_result"


def test_ws_search_sends_session_expired_when_access_token_fails_to_decode(monkeypatch):
    """A token that doesn't decode (expired, forged, wrong secret) must not be silently
    treated as a guest with no signal to the client - the client needs to know its
    stored session has gone stale so it can clear it and prompt re-login, instead of
    persona/history quietly stopping with no visible cause."""

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        raise AssertionError("ai_mode should not run in instant-only mode")

    async def fake_get_persona(personas_collection, user_id):
        raise AssertionError("get_persona should never be called - user_id must resolve to None")

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_persona", fake_get_persona)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "test query", "mode": "instant", "access_token": "not-a-real-jwt"})
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first == {"type": "session_expired"}
    assert second["type"] == "instant_result"


def test_ws_search_omits_session_expired_for_guest(monkeypatch):
    """No access_token at all is a normal guest request, not a stale session -
    session_expired must never fire when none was sent in the first place."""

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        raise AssertionError("ai_mode should not run in instant-only mode")

    _patch_common(monkeypatch, fake_run_ai_mode)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "test query", "mode": "instant"})
        response = websocket.receive_json()

    assert response["type"] == "instant_result"
