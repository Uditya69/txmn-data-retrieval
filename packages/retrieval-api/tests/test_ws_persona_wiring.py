from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from auth.security import create_access_token
from auth.config import get_auth_settings
from retrieval_api.main import app


def test_ws_search_accepts_access_token_field_without_crashing(monkeypatch):
    """Guests (no access_token) and logged-in users (valid access_token) must
    both be able to complete a /ws/search round-trip with mode=instant only —
    this test only proves the message schema accepts the new optional field
    and the connection doesn't crash resolving it; it does not require a live
    Mongo/ES/Milvus stack (mode=instant with no real ES will itself surface
    an es_error in the payload, which is fine — we're asserting no exception
    escapes the handshake).
    """
    settings = get_auth_settings()
    token = create_access_token("user-123", settings)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/search") as websocket:
            websocket.send_json({"query": "test query", "mode": "instant", "access_token": token})
            response = websocket.receive_json()
            assert response["type"] == "instant_result"
