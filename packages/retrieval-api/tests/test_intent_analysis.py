from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from persona.config import get_persona_settings
from retrieval_api.main import app
import retrieval_api.intent_analysis as intent_analysis_module

client = TestClient(app)


def _patch_gateway(monkeypatch):
    monkeypatch.setattr(intent_analysis_module, "get_settings", lambda: AsyncMock(gateway_url="http://gateway"))
    monkeypatch.setattr(intent_analysis_module, "GatewayClient", lambda **_: AsyncMock())


def test_intent_analysis_guest_query_has_no_persona(monkeypatch):
    _patch_gateway(monkeypatch)

    async def fake_extract_intent(gateway, query, persona_context=""):
        return {"original_query": query, "search_query": query, "intent": ["acts"], "filters": {}}

    monkeypatch.setattr(intent_analysis_module, "extract_intent", fake_extract_intent)

    response = client.post("/v1/intent-analysis", json={"query": "section 54F exemption"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == ["acts"]
    assert body["persona_found"] is False
    assert body["persona_context_used"] == ""
    assert body["topic_count"] is None
    assert body["lexicon_check"]["has_anchor"] is True
    assert body["lexicon_check"]["shape"] == "provision"


def test_intent_analysis_user_with_empty_snapshot_gets_no_persona(monkeypatch):
    _patch_gateway(monkeypatch)

    captured = {}

    async def fake_extract_intent(gateway, query, persona_context=""):
        captured["persona_context"] = persona_context
        return {"original_query": query, "search_query": query, "intent": [], "filters": {}}

    async def fake_get_current_snapshot(topics_collection, user_id):
        return []

    monkeypatch.setattr(intent_analysis_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(intent_analysis_module, "get_persona_settings", lambda: get_persona_settings())
    monkeypatch.setattr(intent_analysis_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(intent_analysis_module, "get_persona_topics_collection", lambda *_: object())
    monkeypatch.setattr(intent_analysis_module, "get_current_snapshot", fake_get_current_snapshot)

    response = client.post("/v1/intent-analysis", json={"query": "gst rate", "user_id": "user-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["persona_found"] is False
    assert body["topic_count"] is None
    assert body["persona_context_used"] == ""
    assert captured["persona_context"] == ""


def test_intent_analysis_user_with_active_topic_gets_rendered_context(monkeypatch):
    _patch_gateway(monkeypatch)

    captured = {}

    async def fake_extract_intent(gateway, query, persona_context=""):
        captured["persona_context"] = persona_context
        return {"original_query": query, "search_query": query, "intent": [], "filters": {}}

    async def fake_get_current_snapshot(topics_collection, user_id):
        return [{"topic_id": "t1", "state": "active", "score": 0.9, "legal_entities": ["caselaws"], "categories": []}]

    monkeypatch.setattr(intent_analysis_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(intent_analysis_module, "get_persona_settings", lambda: get_persona_settings())
    monkeypatch.setattr(intent_analysis_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(intent_analysis_module, "get_persona_topics_collection", lambda *_: object())
    monkeypatch.setattr(intent_analysis_module, "get_current_snapshot", fake_get_current_snapshot)

    response = client.post("/v1/intent-analysis", json={"query": "case law on X", "user_id": "user-2"})

    assert response.status_code == 200
    body = response.json()
    assert body["persona_found"] is True
    assert body["topic_count"] == 1
    assert body["persona_context_used"] != ""
    assert "caselaws" in body["persona_context_used"]
    assert captured["persona_context"] == body["persona_context_used"]


def test_intent_analysis_degrades_gracefully_when_persona_lookup_fails(monkeypatch):
    _patch_gateway(monkeypatch)

    async def fake_extract_intent(gateway, query, persona_context=""):
        return {"original_query": query, "search_query": query, "intent": [], "filters": {}}

    def broken_get_persona_settings():
        raise RuntimeError("persona store unreachable")

    monkeypatch.setattr(intent_analysis_module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(intent_analysis_module, "get_persona_settings", broken_get_persona_settings)

    response = client.post("/v1/intent-analysis", json={"query": "gst rate", "user_id": "user-3"})

    assert response.status_code == 200
    body = response.json()
    assert body["persona_found"] is False
    assert body["persona_context_used"] == ""
    assert body["topic_count"] is None


def test_intent_analysis_includes_lexicon_check_for_anchor_free_query(monkeypatch):
    _patch_gateway(monkeypatch)

    async def fake_extract_intent(gateway, query, persona_context=""):
        return {"original_query": query, "search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(intent_analysis_module, "extract_intent", fake_extract_intent)

    response = client.post("/v1/intent-analysis", json={"query": "capital gains"})

    assert response.status_code == 200
    body = response.json()
    assert body["lexicon_check"]["has_anchor"] is False
    assert body["lexicon_check"]["shape"] == "plain"
    assert body["lexicon_check"]["chunks"] == []
