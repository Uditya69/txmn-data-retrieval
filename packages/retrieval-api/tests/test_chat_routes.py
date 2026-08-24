from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
import chat.router as chat_router_module
from retrieval_api.main import app


def _patch_conversations(monkeypatch, fake_conversations_collection):
    monkeypatch.setattr(chat_router_module, "get_chat_settings", lambda: object())
    monkeypatch.setattr(chat_router_module, "get_mongo_client", lambda *_: object())
    monkeypatch.setattr(chat_router_module, "get_conversations_collection", lambda *_: fake_conversations_collection)


def test_list_conversations_requires_auth(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    client = TestClient(app)
    response = client.get("/conversations")
    assert response.status_code == 401


def test_list_conversations_returns_only_callers_conversations(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    from chat.repository import create_conversation
    import asyncio

    asyncio.run(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-1", get_auth_settings())
    client = TestClient(app)
    response = client.get("/conversations", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert [c["id"] for c in response.json()] == ["conv-1"]


def test_get_conversation_404s_for_other_users_conversation(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    from chat.repository import create_conversation
    import asyncio

    asyncio.run(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-2", get_auth_settings())
    client = TestClient(app)
    response = client.get("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404


def test_delete_conversation_404s_for_other_users_conversation(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    from chat.repository import create_conversation
    import asyncio

    asyncio.run(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-2", get_auth_settings())
    client = TestClient(app)
    response = client.delete("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404

    # user-1's conversation must still exist - the failed cross-user delete
    # attempt must not have removed it.
    owner_token = create_access_token("user-1", get_auth_settings())
    assert client.get(
        "/conversations/conv-1", headers={"Authorization": f"Bearer {owner_token}"}
    ).status_code == 200


def test_delete_conversation_removes_it(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    from chat.repository import create_conversation
    import asyncio

    asyncio.run(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-1", get_auth_settings())
    client = TestClient(app)
    response = client.delete("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 204
    assert client.get("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"}).status_code == 404
