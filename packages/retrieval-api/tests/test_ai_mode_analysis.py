from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ai_mode_analysis as ai_mode_analysis_module

client = TestClient(app)


def _patch_common(monkeypatch, fake_run_ai_mode):
    monkeypatch.setattr(ai_mode_analysis_module, "get_settings", lambda: Mock(gateway_url="http://gateway"))
    monkeypatch.setattr(ai_mode_analysis_module, "GatewayClient", lambda **_: AsyncMock())
    monkeypatch.setattr(ai_mode_analysis_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ai_mode_analysis_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ai_mode_analysis_module, "run_ai_mode", fake_run_ai_mode)


def test_ai_mode_analysis_guest_query_has_no_persona(monkeypatch):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["acts"]}

    _patch_common(monkeypatch, fake_run_ai_mode)

    response = client.post("/v1/ai-mode-analysis", json={"query": "section 54F exemption"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "final answer"
    assert body["intent"] == ["acts"]
    assert body["persona_found"] is False
    assert body["persona_context_used"] == ""
    assert body["query_count"] is None
    assert body["lexicon_check"]["has_anchor"] is True
    assert body["lexicon_check"]["shape"] == "provision"


def test_ai_mode_analysis_forwards_persona_context_when_trusted(monkeypatch):
    captured = {}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        captured["persona_context"] = persona_context
        return {"ok": True, "answer": "answer", "citations": {}, "intent": []}

    _patch_common(monkeypatch, fake_run_ai_mode)

    async def fake_get_persona(personas_collection, user_id):
        return {
            "user_id": user_id,
            "category_affinity": {"caselaws": 0.9},
            "expertise_level": "expert",
            "query_style": "precise-citation",
            "query_count": 25,
        }

    monkeypatch.setattr(ai_mode_analysis_module, "get_persona_settings", lambda: object())
    monkeypatch.setattr(ai_mode_analysis_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(ai_mode_analysis_module, "get_personas_collection", lambda *_: object())
    monkeypatch.setattr(ai_mode_analysis_module, "get_persona", fake_get_persona)

    response = client.post("/v1/ai-mode-analysis", json={"query": "case law on X", "user_id": "user-2"})

    assert response.status_code == 200
    body = response.json()
    assert body["persona_found"] is True
    assert body["query_count"] == 25
    assert body["persona_context_used"] != ""
    assert captured["persona_context"] == body["persona_context_used"]


def test_ai_mode_analysis_returns_error_result_on_pipeline_failure(monkeypatch):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": False, "error": "boom"}

    _patch_common(monkeypatch, fake_run_ai_mode)

    response = client.post("/v1/ai-mode-analysis", json={"query": "gst rate"})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "boom"


def test_ai_mode_analysis_degrades_gracefully_when_persona_lookup_fails(monkeypatch):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "answer", "citations": {}, "intent": []}

    _patch_common(monkeypatch, fake_run_ai_mode)

    def broken_get_persona_settings():
        raise RuntimeError("persona store unreachable")

    monkeypatch.setattr(ai_mode_analysis_module, "get_persona_settings", broken_get_persona_settings)

    response = client.post("/v1/ai-mode-analysis", json={"query": "gst rate", "user_id": "user-3"})

    assert response.status_code == 200
    body = response.json()
    assert body["persona_found"] is False
    assert body["persona_context_used"] == ""
    assert body["query_count"] is None


def test_ai_mode_analysis_closes_clients_on_success(monkeypatch):
    es_client = AsyncMock()
    milvus_client = Mock()

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "answer", "citations": {}, "intent": []}

    monkeypatch.setattr(ai_mode_analysis_module, "get_settings", lambda: Mock(gateway_url="http://gateway"))
    monkeypatch.setattr(ai_mode_analysis_module, "GatewayClient", lambda **_: AsyncMock())
    monkeypatch.setattr(ai_mode_analysis_module, "get_es_client", lambda *_: es_client)
    monkeypatch.setattr(ai_mode_analysis_module, "get_milvus_client", lambda *_: milvus_client)
    monkeypatch.setattr(ai_mode_analysis_module, "run_ai_mode", fake_run_ai_mode)

    response = client.post("/v1/ai-mode-analysis", json={"query": "gst rate"})

    assert response.status_code == 200
    es_client.close.assert_awaited_once()
    milvus_client.close.assert_called_once()


def test_ai_mode_analysis_includes_lexicon_check_for_anchor_free_query(monkeypatch):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "answer", "citations": {}, "intent": []}

    _patch_common(monkeypatch, fake_run_ai_mode)

    response = client.post("/v1/ai-mode-analysis", json={"query": "capital gains"})

    assert response.status_code == 200
    body = response.json()
    assert body["lexicon_check"]["has_anchor"] is False
    assert body["lexicon_check"]["shape"] == "plain"
    assert body["lexicon_check"]["chunks"] == []
