from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
from persona.config import get_persona_settings
from retrieval_api.main import app
import retrieval_api.ws as ws_module


def _patch_common(monkeypatch, fake_run_ai_mode):
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
    # Never touch a real MongoDB connection - get_mongo_client/collection getters
    # are stubbed to sentinels only ever passed through to get_current_snapshot/
    # migrate_legacy_persona/record_persona_signal, which are themselves stubbed
    # per test.
    monkeypatch.setattr(ws_module, "get_persona_settings", lambda: get_persona_settings())
    monkeypatch.setattr(ws_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_persona_events_collection", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_persona_topics_collection", lambda *_: object())
    monkeypatch.setattr(ws_module, "get_personas_collection", lambda *_: object())


def test_ws_search_logged_in_user_with_active_topic_reaches_run_ai_mode_with_rendered_context(monkeypatch):
    """A logged-in user whose current snapshot has an active topic above the
    confidence floor must have that reflected in a non-empty persona_context
    string, and that exact string must reach run_ai_mode - proving the seam
    from Mongo snapshot -> render_persona_context -> run_ai_mode wiring works
    end to end (modulo Mongo itself, which is faked out)."""
    captured = {}

    async def fake_get_current_snapshot(topics_collection, user_id):
        assert user_id == "user-123"
        return [{"topic_id": "t1", "state": "active", "score": 0.9, "legal_entities": ["GST"], "categories": ["acts"]}]

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["acts"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)

    record_calls = []

    async def fake_record_persona_signal(events, topics, gateway, user_id, query, categories, timestamp, settings):
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
    assert "GST" in captured["persona_context"]


def test_ws_search_logged_in_user_with_empty_snapshot_gets_empty_persona_context(monkeypatch):
    """A logged-in user with no topics yet (and nothing to migrate) must
    still reach run_ai_mode with persona_context=="" - same as a guest."""
    captured = {}

    async def fake_get_current_snapshot(topics_collection, user_id):
        assert user_id == "user-123"
        return []

    async def fake_migrate_legacy_persona(*args, **kwargs):
        return False

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)
    monkeypatch.setattr(ws_module, "migrate_legacy_persona", fake_migrate_legacy_persona)

    auth_settings = get_auth_settings()
    token = create_access_token("user-123", auth_settings)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert captured["persona_context"] == ""


def test_ws_search_empty_snapshot_triggers_legacy_migration_then_rereads_snapshot(monkeypatch):
    """A user with an empty snapshot but a migratable legacy document should
    have migrate_legacy_persona invoked, and the snapshot re-read afterward."""
    calls = {"migrate": 0, "snapshot": 0}
    snapshots = [[], [{"topic_id": "t1", "state": "active", "score": 0.9, "legal_entities": ["IBC"], "categories": []}]]

    async def fake_get_current_snapshot(topics_collection, user_id):
        calls["snapshot"] += 1
        return snapshots.pop(0)

    async def fake_migrate_legacy_persona(*args, **kwargs):
        calls["migrate"] += 1
        return True

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)
    monkeypatch.setattr(ws_module, "migrate_legacy_persona", fake_migrate_legacy_persona)

    auth_settings = get_auth_settings()
    token = create_access_token("user-123", auth_settings)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        websocket.receive_json()

    assert calls["migrate"] == 1
    assert calls["snapshot"] == 2


def test_ws_search_persona_store_failure_degrades_to_guest_equivalent(monkeypatch):
    """An unreachable persona store during the read path must degrade to
    persona_context="" rather than crashing the request."""
    captured = {}

    async def fake_get_current_snapshot(topics_collection, user_id):
        raise RuntimeError("mongo unreachable")

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)

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

    async def fake_get_current_snapshot(topics_collection, user_id):
        raise AssertionError("get_current_snapshot should never be called for a guest")

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)

    record_calls = []

    async def fake_record_persona_signal(events, topics, gateway, user_id, query, categories, timestamp, settings):
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

    async def fake_get_current_snapshot(topics_collection, user_id):
        return []

    async def fake_migrate_legacy_persona(*args, **kwargs):
        return False

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context="", **_kwargs):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["acts", "rules"]}

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)
    monkeypatch.setattr(ws_module, "migrate_legacy_persona", fake_migrate_legacy_persona)

    record_calls = []

    async def fake_record_persona_signal(events, topics, gateway, user_id, query, categories, timestamp, settings):
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

    async def fake_get_current_snapshot(topics_collection, user_id):
        return []

    async def fake_migrate_legacy_persona(*args, **kwargs):
        return False

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)
    monkeypatch.setattr(ws_module, "migrate_legacy_persona", fake_migrate_legacy_persona)

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

    async def fake_get_current_snapshot(topics_collection, user_id):
        raise AssertionError("get_current_snapshot should never be called - user_id must resolve to None")

    _patch_common(monkeypatch, fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_current_snapshot", fake_get_current_snapshot)

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
