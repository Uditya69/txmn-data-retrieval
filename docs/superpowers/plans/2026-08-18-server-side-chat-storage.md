# Server-Side Chat Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `localStorage`-based chat history with server-side, Mongo-backed persistence for logged-in users, and fully ephemeral (nothing persisted anywhere) chat for guests — plus removing all remaining `localStorage` usage (including UI-preference flags) from the web app.

**Architecture:** A new `packages/chat` Python package (mirrors `packages/persona`'s `config.py`/`db.py`/`repository.py` shape) owns a Mongo `conversations` collection, one document per conversation. `retrieval_api/ws.py`'s `/ws/search` handler accepts an optional `conversation_id` in the incoming message and, only when a `user_id` resolves from the access token, fires a background task after streaming the response that upserts the turn into that conversation (mirrors the existing `record_persona_signal` fire-and-forget pattern). New Bearer-authenticated REST routes (`packages/chat`'s router, following `packages/auth/src/auth/router.py`'s shape) expose list/get/delete for the frontend sidebar. The frontend (`packages/web`) drops all `localStorage`-based conversation and UI-state persistence, replacing it with plain React state plus fetch calls to the new REST routes when a user is logged in.

**Tech Stack:** Python 3.11, FastAPI, Motor (async Mongo driver), pydantic-settings, pytest + pytest-asyncio (backend); React, TypeScript, Vitest + React Testing Library (frontend).

**Spec:** `docs/superpowers/specs/2026-08-18-server-side-chat-storage-design.md`

## Global Constraints

- Python 3.11, not 3.14 (repo-wide `pyproject.toml` pin — `pymilvus`'s `grpcio` has no 3.14 wheel).
- Guests: zero persistence anywhere — no `localStorage`, no DB writes, no `conversation_id` handling at all.
- Logged-in users: conversations persisted indefinitely, no retention cap or TTL.
- Existing users' `localStorage` chat history is discarded, not migrated.
- No `localStorage` usage anywhere in `packages/web` after this plan, including non-chat UI preferences (e.g. sidebar collapsed state) — those become plain `useState`.
- Background persistence failures must never surface to the client or block/fail the streamed WS response (mirrors `record_persona_signal`'s and the persona-lookup try/except's existing resilience pattern in `ws.py`).
- `uv sync --all-packages` (not bare `uv sync`) after adding the new workspace member, or editable installs break.

---

## Task 1: `packages/chat` package scaffold — config, db, repository

**Files:**
- Create: `packages/chat/pyproject.toml`
- Create: `packages/chat/src/chat/__init__.py`
- Create: `packages/chat/src/chat/config.py`
- Create: `packages/chat/src/chat/db.py`
- Create: `packages/chat/src/chat/repository.py`
- Test: `packages/chat/tests/conftest.py`
- Test: `packages/chat/tests/test_repository.py`
- Modify: `pyproject.toml:6-14` (add `packages/chat` to workspace members and sources)

**Interfaces:**
- Produces: `chat.config.get_chat_settings() -> ChatSettings` (fields: `mongo_uri: str`, `mongo_db: str`).
- Produces: `chat.db.get_mongo_client(settings: ChatSettings) -> AsyncIOMotorClient`, `chat.db.get_conversations_collection(client, settings: ChatSettings) -> AsyncIOMotorCollection`.
- Produces: `chat.repository.create_conversation(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict`, `chat.repository.append_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict`, `chat.repository.list_conversations(conversations, user_id: str) -> list[dict]`, `chat.repository.get_conversation(conversations, conversation_id: str, user_id: str) -> dict | None`, `chat.repository.delete_conversation(conversations, conversation_id: str, user_id: str) -> bool`.
- Consumes: nothing from other tasks (this is the foundation task).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "chat"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "motor>=3.5",
  "pydantic-settings>=2.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/chat"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create empty `src/chat/__init__.py`**

Empty file (matches `packages/persona/src/persona/__init__.py`).

- [ ] **Step 3: Write `src/chat/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str


@lru_cache
def get_chat_settings() -> ChatSettings:
    return ChatSettings()
```

- [ ] **Step 4: Write `src/chat/db.py`**

```python
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from chat.config import ChatSettings


@lru_cache
def get_mongo_client(settings: ChatSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_conversations_collection(client: AsyncIOMotorClient, settings: ChatSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["conversations"]
```

- [ ] **Step 5: Register the workspace member**

Edit `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents", "packages/auth", "packages/persona", "packages/chat"]

[tool.uv.sources]
common = { workspace = true }
model-gateway = { workspace = true }
retrieval-api = { workspace = true }
agents = { workspace = true }
auth = { workspace = true }
persona = { workspace = true }
chat = { workspace = true }
```

- [ ] **Step 6: Write `tests/conftest.py`** (in-memory fake collection, mirrors `packages/persona/tests/conftest.py`)

```python
import os

import pytest

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-chat-db")


class FakeConversationsCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    chat.repository needs: find_one, replace_one, find (list), delete_one.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        doc = self.documents.get(filter.get("_id"))
        if doc is None:
            return None
        if "user_id" in filter and doc.get("user_id") != filter["user_id"]:
            return None
        return doc

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[filter["_id"]] = replacement

    def find(self, filter: dict):
        matches = [d for d in self.documents.values() if d.get("user_id") == filter.get("user_id")]

        class _Cursor:
            def __init__(self, docs):
                self._docs = docs

            def sort(self, field, direction):
                reverse = direction < 0
                self._docs = sorted(self._docs, key=lambda d: d[field], reverse=reverse)
                return self

            def __aiter__(self):
                self._iter = iter(self._docs)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        return _Cursor(matches)

    async def delete_one(self, filter: dict) -> "_DeleteResult":
        doc = self.documents.get(filter.get("_id"))
        deleted = 0
        if doc is not None and doc.get("user_id") == filter.get("user_id"):
            del self.documents[filter["_id"]]
            deleted = 1
        return _DeleteResult(deleted)


class _DeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


@pytest.fixture
def fake_conversations_collection():
    return FakeConversationsCollection()
```

- [ ] **Step 7: Write the failing tests in `tests/test_repository.py`**

```python
import pytest

from chat.repository import (
    append_turn,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)


@pytest.mark.asyncio
async def test_create_conversation_stores_document(fake_conversations_collection):
    conversations = fake_conversations_collection
    doc = await create_conversation(conversations, "conv-1", "user-1", "first question", [{"role": "user", "text": "hi"}])
    assert doc["_id"] == "conv-1"
    assert doc["user_id"] == "user-1"
    assert doc["title"] == "first question"
    assert doc["messages"] == [{"role": "user", "text": "hi"}]
    assert doc["created_at"] == doc["updated_at"]


@pytest.mark.asyncio
async def test_append_turn_extends_existing_conversation(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "first question", [{"role": "user", "text": "hi"}])
    updated = await append_turn(
        conversations, "conv-1", "user-1", "first question",
        [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
    )
    assert len(updated["messages"]) == 2
    assert updated["updated_at"] >= updated["created_at"]


@pytest.mark.asyncio
async def test_append_turn_creates_conversation_when_absent(fake_conversations_collection):
    conversations = fake_conversations_collection
    doc = await append_turn(conversations, "conv-new", "user-1", "q", [{"role": "user", "text": "q"}])
    assert doc["_id"] == "conv-new"
    assert doc["messages"] == [{"role": "user", "text": "q"}]


@pytest.mark.asyncio
async def test_list_conversations_returns_only_callers_own_newest_first(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q1", [])
    await create_conversation(conversations, "conv-2", "user-1", "q2", [])
    await create_conversation(conversations, "conv-3", "user-2", "other user", [])

    result = await list_conversations(conversations, "user-1")

    assert [c["_id"] for c in result] == ["conv-2", "conv-1"]


@pytest.mark.asyncio
async def test_get_conversation_returns_none_for_wrong_user(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q", [])
    assert await get_conversation(conversations, "conv-1", "user-2") is None
    assert await get_conversation(conversations, "conv-1", "user-1") is not None


@pytest.mark.asyncio
async def test_delete_conversation_removes_only_owners_document(fake_conversations_collection):
    conversations = fake_conversations_collection
    await create_conversation(conversations, "conv-1", "user-1", "q", [])
    assert await delete_conversation(conversations, "conv-1", "user-2") is False
    assert await delete_conversation(conversations, "conv-1", "user-1") is True
    assert await get_conversation(conversations, "conv-1", "user-1") is None
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `cd packages/chat && uv run pytest tests/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chat.repository'` (or import error) since `repository.py` doesn't exist yet.

- [ ] **Step 9: Write `src/chat/repository.py`**

```python
from datetime import datetime, timezone


async def create_conversation(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "_id": conversation_id,
        "user_id": user_id,
        "title": title,
        "messages": messages,
        "created_at": now,
        "updated_at": now,
    }
    await conversations.replace_one({"_id": conversation_id}, doc, upsert=True)
    return doc


async def append_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> dict:
    existing = await conversations.find_one({"_id": conversation_id, "user_id": user_id})
    if existing is None:
        return await create_conversation(conversations, conversation_id, user_id, title, messages)

    doc = {
        **existing,
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await conversations.replace_one({"_id": conversation_id}, doc, upsert=True)
    return doc


async def list_conversations(conversations, user_id: str) -> list[dict]:
    cursor = conversations.find({"user_id": user_id}).sort("updated_at", -1)
    return [doc async for doc in cursor]


async def get_conversation(conversations, conversation_id: str, user_id: str) -> dict | None:
    return await conversations.find_one({"_id": conversation_id, "user_id": user_id})


async def delete_conversation(conversations, conversation_id: str, user_id: str) -> bool:
    result = await conversations.delete_one({"_id": conversation_id, "user_id": user_id})
    return result.deleted_count > 0
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `cd packages/chat && uv run pytest tests/test_repository.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 11: Sync workspace and run full aggregated suite once for a sanity check**

Run: `uv sync --all-packages && uv run pytest packages/chat/tests -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml packages/chat
git commit -m "feat(chat): add chat package with Mongo-backed conversation repository"
```

---

## Task 2: REST routes for listing/reading/deleting conversations

**Files:**
- Create: `packages/chat/src/chat/router.py`
- Create: `packages/chat/src/chat/models.py`
- Test: `packages/retrieval-api/tests/test_chat_routes.py`
- Modify: `packages/retrieval-api/src/retrieval_api/main.py` (register the new router)
- Modify: `packages/retrieval-api/pyproject.toml` (add `chat` dependency)

**Interfaces:**
- Consumes: `chat.repository.list_conversations`, `chat.repository.get_conversation`, `chat.repository.delete_conversation` (Task 1); `auth.dependency.get_current_user_id` (existing, `packages/auth/src/auth/dependency.py:7`).
- Produces: `chat.router.router` (an `APIRouter` with prefix `/conversations`), mounted in `retrieval_api.main`.

- [ ] **Step 1: Add `chat` as a dependency of `retrieval-api`**

Edit `packages/retrieval-api/pyproject.toml`, add `"chat"` to the `dependencies` list (alongside `"persona"`).

- [ ] **Step 2: Write `src/chat/models.py`**

```python
from pydantic import BaseModel


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[dict]
    created_at: str
    updated_at: str


def to_summary(doc: dict) -> ConversationSummary:
    return ConversationSummary(id=doc["_id"], title=doc["title"], updated_at=doc["updated_at"])


def to_detail(doc: dict) -> ConversationDetail:
    return ConversationDetail(
        id=doc["_id"], title=doc["title"], messages=doc["messages"],
        created_at=doc["created_at"], updated_at=doc["updated_at"],
    )
```

- [ ] **Step 3: Write `src/chat/router.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from auth.dependency import get_current_user_id
from chat.config import get_chat_settings
from chat.db import get_conversations_collection, get_mongo_client
from chat.models import ConversationDetail, ConversationSummary, to_detail, to_summary
from chat.repository import delete_conversation, get_conversation, list_conversations

router = APIRouter(prefix="/conversations", tags=["chat"])


def get_conversations_dependency():
    settings = get_chat_settings()
    client = get_mongo_client(settings)
    return get_conversations_collection(client, settings)


def _require_user_id(user_id: str | None) -> str:
    if user_id is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user_id


@router.get("", response_model=list[ConversationSummary])
async def list_conversations_route(
    user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    docs = await list_conversations(conversations, user_id)
    return [to_summary(doc) for doc in docs]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    doc = await get_conversation(conversations, conversation_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return to_detail(doc)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation_route(
    conversation_id: str, user_id: str | None = Depends(get_current_user_id), conversations=Depends(get_conversations_dependency),
):
    user_id = _require_user_id(user_id)
    deleted = await delete_conversation(conversations, conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
```

- [ ] **Step 4: Register the router in `main.py`**

Edit `packages/retrieval-api/src/retrieval_api/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.router import router as auth_router
from chat.router import router as chat_router
from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router
from retrieval_api.query_analysis import router as query_analysis_router
from retrieval_api.intent_analysis import router as intent_analysis_router
from retrieval_api.ai_mode_analysis import router as ai_mode_analysis_router

app = FastAPI(title="retrieval-api")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
app.include_router(query_analysis_router)
app.include_router(intent_analysis_router)
app.include_router(ai_mode_analysis_router)
app.include_router(auth_router)
app.include_router(chat_router)
```

(Note: `allow_methods` gains `"DELETE"` for the new delete route.)

- [ ] **Step 5: Write the failing tests in `packages/retrieval-api/tests/test_chat_routes.py`**

```python
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

    asyncio.get_event_loop().run_until_complete(
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

    asyncio.get_event_loop().run_until_complete(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-2", get_auth_settings())
    client = TestClient(app)
    response = client.get("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404


def test_delete_conversation_removes_it(monkeypatch, fake_conversations_collection):
    _patch_conversations(monkeypatch, fake_conversations_collection)
    from chat.repository import create_conversation
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        create_conversation(fake_conversations_collection, "conv-1", "user-1", "q1", [])
    )

    token = create_access_token("user-1", get_auth_settings())
    client = TestClient(app)
    response = client.delete("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 204
    assert client.get("/conversations/conv-1", headers={"Authorization": f"Bearer {token}"}).status_code == 404
```

- [ ] **Step 6: Add the `fake_conversations_collection` fixture to `packages/retrieval-api/tests/conftest.py`**

Append the same `FakeConversationsCollection` class and fixture as in `packages/chat/tests/conftest.py` (Task 1, Step 6) to `packages/retrieval-api/tests/conftest.py` — this repo's convention (per `test_persona_signal.py`'s comment on `fake_personas_collection`) is to duplicate fakes per-package rather than share a test-only import across packages, since `--import-mode=importlib` doesn't add `tests/` to `sys.path`.

- [ ] **Step 7: Run tests to verify they fail**

Run: `cd packages/retrieval-api && uv run pytest tests/test_chat_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chat'` or route not found (404 instead of 401/200), since the router isn't registered/doesn't exist yet.

- [ ] **Step 8: Verify implementation from Steps 2-4 makes tests pass**

Run: `uv sync --all-packages && cd packages/retrieval-api && uv run pytest tests/test_chat_routes.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 9: Run full existing suite to check nothing broke (e.g. CORS test)**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS. If `test_cors.py` asserts an exact `allow_methods` list, update it to include `"DELETE"`.

- [ ] **Step 10: Commit**

```bash
git add packages/chat/src/chat/router.py packages/chat/src/chat/models.py \
  packages/retrieval-api/src/retrieval_api/main.py packages/retrieval-api/pyproject.toml \
  packages/retrieval-api/tests/test_chat_routes.py packages/retrieval-api/tests/conftest.py
git commit -m "feat(chat): add REST routes to list, read, and delete conversations"
```

---

## Task 3: WS write path — persist a turn for logged-in users

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py`
- Create: `packages/retrieval-api/src/retrieval_api/ai_mode/chat_signal.py`
- Test: `packages/retrieval-api/tests/test_chat_signal.py`
- Test: `packages/retrieval-api/tests/test_ws_chat_wiring.py`

**Interfaces:**
- Consumes: `chat.repository.append_turn` (Task 1); `chat.config.get_chat_settings`, `chat.db.get_mongo_client`, `chat.db.get_conversations_collection` (Task 1).
- Produces: `retrieval_api.ai_mode.chat_signal.record_conversation_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> None` — a thin wrapper that swallows all exceptions (mirrors `record_persona_signal`'s shape), used as the background-task target in `ws.py`.

- [ ] **Step 1: Write the failing test for the wrapper, `packages/retrieval-api/tests/test_chat_signal.py`**

```python
import pytest

from chat.repository import get_conversation
from retrieval_api.ai_mode.chat_signal import record_conversation_turn


@pytest.mark.asyncio
async def test_record_conversation_turn_writes_conversation(fake_conversations_collection):
    conversations = fake_conversations_collection

    await record_conversation_turn(conversations, "conv-1", "user-1", "gst rate", [{"role": "user", "text": "gst rate"}])

    stored = await get_conversation(conversations, "conv-1", "user-1")
    assert stored is not None
    assert stored["messages"] == [{"role": "user", "text": "gst rate"}]


@pytest.mark.asyncio
async def test_record_conversation_turn_swallows_errors():
    class BrokenCollection:
        async def find_one(self, filter):
            raise RuntimeError("mongo unreachable")

    # Must not raise - background task failures must never propagate.
    await record_conversation_turn(BrokenCollection(), "conv-1", "user-1", "q", [])
```

(Uses the `fake_conversations_collection` fixture added to `packages/retrieval-api/tests/conftest.py` in Task 2, Step 6.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_chat_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.ai_mode.chat_signal'`

- [ ] **Step 3: Write `src/retrieval_api/ai_mode/chat_signal.py`**

```python
import logging

from chat.repository import append_turn

logger = logging.getLogger(__name__)


async def record_conversation_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> None:
    try:
        await append_turn(conversations, conversation_id, user_id, title, messages)
    except Exception:
        logger.warning("conversation turn write failed for user %r, conversation %r", user_id, conversation_id, exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_chat_signal.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write the failing integration test, `packages/retrieval-api/tests/test_ws_chat_wiring.py`**

```python
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from auth.config import get_auth_settings
from auth.security import create_access_token
from retrieval_api.main import app
import retrieval_api.ws as ws_module


def _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection):
    async def fake_run_instant(gateway, es_client, milvus_client, query, on_step=None, rerank=False):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
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
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
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
        stored = asyncio.get_event_loop().run_until_complete(get_conversation(fake_conversations_collection, "conv-1", "user-123"))
        if stored is not None:
            break
        time.sleep(0.01)

    assert stored is not None
    assert stored["title"] == "gst rate"


def test_ws_search_guest_never_writes_conversation(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "conversation_id": "conv-1"})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert fake_conversations_collection.documents == {}


def test_ws_search_logged_in_user_without_conversation_id_does_not_crash(monkeypatch, fake_conversations_collection):
    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context=""):
        return {"ok": True, "answer": "final answer", "citations": {}, "intent": ["caselaws"]}

    _patch_common(monkeypatch, fake_run_ai_mode, fake_conversations_collection)

    token = create_access_token("user-123", get_auth_settings())
    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "gst rate", "mode": "ai_mode", "access_token": token})
        response = websocket.receive_json()

    assert response == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
    assert fake_conversations_collection.documents == {}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_chat_wiring.py -v`
Expected: FAIL with `AttributeError: module 'retrieval_api.ws' has no attribute 'get_chat_settings'` (or similar), since `ws.py` doesn't import/use these yet.

- [ ] **Step 7: Wire conversation persistence into `ws.py`**

Edit `packages/retrieval-api/src/retrieval_api/ws.py`. Add imports (near the existing persona imports):

```python
from chat.config import get_chat_settings
from chat.db import get_conversations_collection, get_mongo_client as get_chat_mongo_client
from retrieval_api.ai_mode.chat_signal import record_conversation_turn
```

In `search()`, after reading `access_token`/`user_id` (after line 73, `user_id = _resolve_user_id(access_token)`), read the conversation id:

```python
    conversation_id = message.get("conversation_id")
```

After the `ai_mode_done` send block (after the existing persona-write background task, i.e. after the `task.add_done_callback(_background_tasks.discard)` line inside the `if user_id is not None and personas_collection is not None:` block), add a second, independent background task — conversation persistence must not depend on persona lookup having succeeded:

```python
                    if user_id is not None and conversation_id is not None:
                        try:
                            chat_settings = get_chat_settings()
                            chat_mongo_client = get_chat_mongo_client(chat_settings)
                            conversations_collection = get_conversations_collection(chat_mongo_client, chat_settings)
                            chat_task = asyncio.create_task(
                                record_conversation_turn(
                                    conversations_collection, conversation_id, user_id, query,
                                    [
                                        {"role": "user", "text": query},
                                        {"role": "assistant", "text": ai_mode_result["answer"]},
                                    ],
                                )
                            )
                            _background_tasks.add(chat_task)
                            chat_task.add_done_callback(_background_tasks.discard)
                        except Exception:
                            # A down/unreachable chat store must never crash the request -
                            # mirrors the persona lookup's resilience pattern above.
                            logger.exception("Failed to schedule conversation write for user %r", user_id)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_chat_wiring.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 9: Run full retrieval-api suite**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS (all tests, including the pre-existing `test_ws_persona_wiring.py` and `test_ws_integration.py`)

- [ ] **Step 10: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ws.py \
  packages/retrieval-api/src/retrieval_api/ai_mode/chat_signal.py \
  packages/retrieval-api/tests/test_chat_signal.py packages/retrieval-api/tests/test_ws_chat_wiring.py
git commit -m "feat(chat): persist conversation turns for logged-in users after ai_mode responses"
```

---

## Task 4: Frontend — remove localStorage, add conversation_id + fetch-based sidebar

**Files:**
- Modify: `packages/web/src/App.tsx`
- Modify: `packages/web/src/App.test.tsx`
- Modify: `packages/web/src/api/useSearch.ts`
- Modify: `packages/web/src/api/useSearch.test.ts`
- Modify: `packages/web/src/types.ts`
- Create: `packages/web/src/api/useConversations.ts`
- Create: `packages/web/src/api/useConversations.test.ts`

**Interfaces:**
- Consumes: `apiBaseUrl` (already computed in `App.tsx:110` via `resolveApiBaseUrl`), `auth.token`/`auth.email` from `useAuth` (existing, `packages/web/src/api/useAuth.ts`).
- Produces: `useConversations(apiBaseUrl: string, token: string | null): { conversations: ConversationSummary[], activeMessages: ChatMessage[] | null, loadConversation: (id: string) => Promise<ChatMessage[]>, refresh: () => Promise<void>, remove: (id: string) => Promise<void> }` (exact shape used by `App.tsx`).
- Modifies `Conversation` type in `types.ts` to no longer be the sole source of truth for persistence — `App.tsx` keeps an in-memory `Conversation[]` for the active session same as before, but never reads/writes `localStorage`.

- [ ] **Step 1: Add `conversation_id` to the WS payload in `useSearch.ts`**

Edit `packages/web/src/api/useSearch.ts`. Change the `search` callback's signature and payload construction (lines 46-66):

```typescript
  const search = useCallback(
    (query: string, trace: boolean, mode: SearchMode = 'both', rerank: boolean = false, conversationId?: string) => {
      socketRef.current?.close()
      setState({ loading: true, instant: null, aiMode: null, traceSteps: [], wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        const payload: Record<string, unknown> = { query, mode, trace, rerank }
        if (accessToken) payload.access_token = accessToken
        if (conversationId) payload.conversation_id = conversationId
        socket.send(JSON.stringify(payload))
      })
```

Update the returned type accordingly:

```typescript
export function useSearch(
  wsUrl: string,
  accessToken?: string | null,
  onSessionExpired?: () => void,
): SearchState & { search: (query: string, trace: boolean, mode?: SearchMode, rerank?: boolean, conversationId?: string) => void } {
```

- [ ] **Step 2: Update `useSearch.test.ts`'s existing exact-payload assertion and add a new test**

The existing test `'sends the query with mode "both" and the trace flag once the socket opens, and stores the instant result'` asserts an exact payload with no `conversation_id` key — that assertion still holds unchanged since `conversationId` is `undefined` in that call and Step 1's code only adds the key `if (conversationId)`. Add one new test right after the `'includes access_token in the payload when a token is provided'` test:

```typescript
  it('includes conversation_id in the payload when provided', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true, 'both', false, 'conv-42')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).toMatchObject({ conversation_id: 'conv-42' })
  })

  it('omits conversation_id from the payload when not provided', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).not.toHaveProperty('conversation_id')
  })
```

- [ ] **Step 3: Run `useSearch.test.ts` to verify all tests pass**

Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts`
Expected: PASS (all tests, including the two new ones). If the exact-payload test (`toEqual([JSON.stringify(...)])`) fails, it means Step 1's `if (conversationId)` guard leaked an `undefined` key into that call's JSON — fix `useSearch.ts` so the key is omitted entirely when absent, not set to `undefined`.

- [ ] **Step 4: Write `src/api/useConversations.ts`**

```typescript
import { useCallback, useState } from 'react'
import type { ChatMessage } from '../types'

export interface ConversationSummary {
  id: string
  title: string
  updated_at: string
}

interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[]
  created_at: string
}

export function useConversations(apiBaseUrl: string, token: string | null) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])

  const refresh = useCallback(async () => {
    if (!token) {
      setConversations([])
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) return
      const data = (await response.json()) as ConversationSummary[]
      setConversations(data)
    } catch {
      // Network failure: leave whatever list is already in state rather than
      // clearing the sidebar on a transient blip.
    }
  }, [apiBaseUrl, token])

  const loadConversation = useCallback(
    async (id: string): Promise<ChatMessage[]> => {
      if (!token) return []
      const response = await fetch(`${apiBaseUrl}/conversations/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) return []
      const data = (await response.json()) as ConversationDetail
      return data.messages
    },
    [apiBaseUrl, token],
  )

  const remove = useCallback(
    async (id: string) => {
      if (!token) return
      await fetch(`${apiBaseUrl}/conversations/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {})
      setConversations((prev) => prev.filter((c) => c.id !== id))
    },
    [apiBaseUrl, token],
  )

  return { conversations, refresh, loadConversation, remove }
}
```

- [ ] **Step 5: Write `src/api/useConversations.test.ts`**

```typescript
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { useConversations } from './useConversations'

describe('useConversations', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('does not fetch and returns an empty list when there is no token', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch')
    const { result } = renderHook(() => useConversations('http://api', null))

    await act(async () => {
      await result.current.refresh()
    })

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(result.current.conversations).toEqual([])
  })

  it('fetches and stores the conversation list when a token is present', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => [{ id: 'conv-1', title: 'q1', updated_at: '2026-08-18T00:00:00Z' }],
    } as Response)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))

    await act(async () => {
      await result.current.refresh()
    })

    await waitFor(() => {
      expect(result.current.conversations).toEqual([{ id: 'conv-1', title: 'q1', updated_at: '2026-08-18T00:00:00Z' }])
    })
  })

  it('loadConversation returns the messages from the detail endpoint', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'conv-1', title: 'q1', created_at: 'x', updated_at: 'x',
        messages: [{ id: 'm1', role: 'user', text: 'hi' }],
      }),
    } as Response)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))
    const messages = await result.current.loadConversation('conv-1')

    expect(messages).toEqual([{ id: 'm1', role: 'user', text: 'hi' }])
  })

  it('remove calls DELETE and drops the conversation from local state', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => [] } as Response)
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))
    await act(async () => {
      await result.current.remove('conv-1')
    })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://api/conversations/conv-1',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })
})
```

- [ ] **Step 6: Run the new test file to verify it fails, then implement is already done in Step 4 — run to verify it passes**

Run: `cd packages/web && npx vitest run src/api/useConversations.test.ts`
Expected: PASS (all 4 tests). If `renderHook`/`waitFor`/`act` aren't already imported elsewhere in this package's test suite (check `App.test.tsx` and any hook test for the existing import source, e.g. `@testing-library/react`), adjust the import to match this repo's actual installed testing-library packages.

- [ ] **Step 7: Remove localStorage from `App.tsx`**

Edit `packages/web/src/App.tsx`. Remove `CONVERSATIONS_KEY_PREFIX`, `SIDEBAR_KEY`, `conversationsKey`, `loadConversations`, `toPersistable`, `isQuotaExceeded`, `persistConversations` (lines 15-92) entirely — replace with nothing (their behavior moves into `useConversations` for logged-in users and plain in-memory state for guests).

Add the import:

```typescript
import { useConversations } from './api/useConversations'
```

Replace the conversations/sidebar state (lines 115-118):

```typescript
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const remoteConversations = useConversations(apiBaseUrl, auth.token)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
```

Replace the login/logout bucket-swap effect (lines 131-138) — when the token appears, fetch the remote list; when it disappears (logout), clear in-memory state:

```typescript
  useEffect(() => {
    if (auth.token) {
      remoteConversations.refresh()
    } else {
      setConversations([])
      setActiveId(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.token])
```

Remove the `persistConversations` effect (lines 140-142) and the `sidebarCollapsed` localStorage effect (lines 144-146) entirely — no replacement needed, state is now purely in-memory.

Add a conversation-select handler that loads remote messages for logged-in users (new function, placed near `handleNewChat`):

```typescript
  async function handleSelectConversation(id: string) {
    if (auth.token) {
      const existing = conversations.find((c) => c.id === id)
      if (!existing) {
        const messages = await remoteConversations.loadConversation(id)
        const summary = remoteConversations.conversations.find((c) => c.id === id)
        setConversations((prev) => [...prev, { id, title: summary?.title ?? id, messages }])
      }
    }
    setActiveId(id)
  }
```

Update `Sidebar`'s props in the JSX (previously `conversations={conversations}` and `onSelect={setActiveId}`) to reflect logged-in vs guest sourcing:

```typescript
      <Sidebar
        conversations={
          auth.token
            ? remoteConversations.conversations.map((c) => ({ id: c.id, title: c.title, messages: [] }))
            : conversations
        }
        activeId={activeId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
      />
```

Pass `conversationId` through to `runQuery`'s `classicSearch.search` call so the WS payload includes it (edit `runQuery`, line ~205):

```typescript
  function runQuery(conversationId: string, assistantId: string, question: string, targetMode: ChatMode) {
    if (targetMode === 'classic') {
      pendingClassicRef.current = { conversationId, assistantId }
      classicSearch.search(question, true, 'both', rerank, auth.token ? conversationId : undefined)
    } else {
      pendingAgentRef.current = { conversationId, assistantId }
      agentSearch.search(question)
    }
  }
```

After a successful `ai_mode_done` for a logged-in user's conversation, refresh the sidebar list so a brand-new conversation's title shows up (add inside the existing `useEffect` that watches `classicSearch.aiMode`, right after the `patchResult` call, guarded to only fire once per completed answer — reuse the same `pending` ref check already in that effect):

```typescript
  useEffect(() => {
    const pending = pendingClassicRef.current
    if (!pending) return
    patchResult(pending.conversationId, pending.assistantId, 'classic', () => ({
      status: classicSearch.loading ? 'loading' : classicSearch.aiMode ? 'done' : 'loading',
      instant: classicSearch.instant,
      aiMode: classicSearch.aiMode,
      traceSteps: classicSearch.traceSteps,
    }))
    if (!classicSearch.loading && classicSearch.aiMode && auth.token) {
      remoteConversations.refresh()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classicSearch.instant, classicSearch.aiMode, classicSearch.traceSteps, classicSearch.loading])
```

- [ ] **Step 8: Update `App.test.tsx`**

Remove the `persistConversations`/`toPersistable` import and their two `describe` blocks (lines 3, 62-135) — those functions no longer exist in `App.tsx`. Remove `localStorage.clear()` from the remaining `beforeEach` (line 21) since nothing in `App` touches `localStorage` anymore. Add `vi.mock('./api/useConversations', ...)` alongside the existing `useSearch`/`useAgentSearch` mocks:

```typescript
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import { useConversations } from './api/useConversations'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useAgentSearch', () => ({ useAgentSearch: vi.fn() }))
vi.mock('./api/useConversations', () => ({ useConversations: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
}

function baseAgentState() {
  return { loading: false, traceSteps: [], result: null, wsError: null, search: vi.fn() }
}

function baseConversationsState() {
  return { conversations: [], refresh: vi.fn(), loadConversation: vi.fn(), remove: vi.fn() }
}

describe('App', () => {
  beforeEach(() => {
    vi.mocked(useSearch).mockReturnValue(baseSearchState())
    vi.mocked(useAgentSearch).mockReturnValue(baseAgentState())
    vi.mocked(useConversations).mockReturnValue(baseConversationsState())
  })

  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })

  it('renders the Classic/Agent mode toggle', () => {
    render(<App />)
    expect(screen.getByText('classic')).toBeInTheDocument()
    expect(screen.getByText('agent')).toBeInTheDocument()
  })

  it('defaults dev mode on with no ?dev URL param', () => {
    render(<App />)
    expect(screen.getByLabelText('Dev mode', { selector: 'input' })).toBeChecked()
  })

  it('turns dev mode off when the URL has ?dev=0', () => {
    window.history.pushState({}, '', '/?dev=0')
    render(<App />)
    expect(screen.getByLabelText('Dev mode', { selector: 'input' })).not.toBeChecked()
    window.history.pushState({}, '', '/')
  })

  it('submits a question via the chat input and triggers classic search', () => {
    const search = vi.fn()
    vi.mocked(useSearch).mockReturnValue({ ...baseSearchState(), search })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'what is section 80HH' } })
    fireEvent.click(screen.getByLabelText('Send'))

    expect(search).toHaveBeenCalledWith('what is section 80HH', true, 'both', false, undefined)
    expect(screen.getAllByText('what is section 80HH').length).toBeGreaterThan(0)
  })

  it('never touches localStorage', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')
    render(<App />)
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'test' } })
    fireEvent.click(screen.getByLabelText('Send'))
    expect(setItemSpy).not.toHaveBeenCalled()
    setItemSpy.mockRestore()
  })
})
```

- [ ] **Step 9: Run the web test suite to verify it fails first (before Step 7's App.tsx edit is considered complete), then passes**

Run: `cd packages/web && npx vitest run src/App.test.tsx src/api/useSearch.test.ts src/api/useConversations.test.ts`
Expected: PASS (all tests). Fix any mismatch between the mocked `useConversations` shape and what `App.tsx` actually destructures.

- [ ] **Step 10: Run the full web test suite**

Run: `cd packages/web && npx vitest run`
Expected: PASS (no other file references the removed `persistConversations`/`loadConversations`/`toPersistable` exports — confirm with a repo-wide search before this step: `grep -rn "persistConversations\|loadConversations\|toPersistable" packages/web/src` should return no matches outside `App.tsx`'s git history).

- [ ] **Step 11: Manually verify in a running browser** (per this repo's UI-change convention)

Run: `cd packages/web && npm run dev`, open the app, ask a question as a guest (confirm `localStorage` stays empty via devtools), sign up/log in, ask a question, refresh the page, confirm the conversation reappears in the sidebar (fetched from the server), and confirm `localStorage` is still empty throughout.

- [ ] **Step 12: Commit**

```bash
git add packages/web/src/App.tsx packages/web/src/App.test.tsx \
  packages/web/src/api/useSearch.ts packages/web/src/api/useSearch.test.ts \
  packages/web/src/api/useConversations.ts packages/web/src/api/useConversations.test.ts
git commit -m "feat(chat): remove localStorage from web app, fetch conversations from server"
```

---

## Task 5: Documentation and env example updates

**Files:**
- Modify: `.env.example` (if it exists at repo root or per-package — check first)
- Modify: `CLAUDE.md` (only if it references localStorage-based chat storage anywhere — check first)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — this task only updates docs/config so the new `packages/chat` package's required env vars (`MONGO_URI`, `MONGO_DB` — already present for `persona`/`auth`, likely no new vars needed since it's the same Mongo deployment) are discoverable.

- [ ] **Step 1: Check whether `.env.example` needs changes**

Run: `grep -rn "MONGO_URI\|MONGO_DB" .env.example 2>/dev/null; find . -maxdepth 2 -name ".env.example"`

If `MONGO_URI`/`MONGO_DB` are already documented (they should be, from `persona`/`auth`), no change is needed — `chat` reuses the same env vars. If not present anywhere, add them following the existing `.env.example` format for `persona`.

- [ ] **Step 2: Check whether `CLAUDE.md` needs an update**

Run: `grep -n "localStorage\|conversation" CLAUDE.md`

If any hits describe the old localStorage-based chat storage as current behavior, update that line to reflect the new server-side model. If no hits, no change needed — do not add new sections proactively.

- [ ] **Step 3: Commit if anything changed**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: note server-side chat storage in place of localStorage" --allow-empty
```

(Use `--allow-empty` only if Steps 1-2 found nothing to change and you still want a marker commit; otherwise skip this commit entirely if there's nothing staged.)

---

## Final Verification

- [ ] Run the full aggregated backend suite: `uv sync --all-packages && uv run pytest` — expect all tests (previous 143 + new ones from Tasks 1-3) to pass.
- [ ] Run the full frontend suite: `cd packages/web && npx vitest run` — expect all tests to pass.
- [ ] Run `grep -rn "localStorage" packages/web/src` — expect zero matches.
- [ ] Manually confirm (per Task 4 Step 11) that guest chats never touch `localStorage` or the DB, and logged-in chats survive a page refresh via the server.
