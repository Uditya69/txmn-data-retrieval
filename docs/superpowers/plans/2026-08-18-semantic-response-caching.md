# Semantic Response Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a semantic cache (MongoDB Atlas `$vectorSearch`, reusing the existing `query_embed` Voyage role) so AI Mode and Instant mode can serve a near-duplicate query from a prior answer instead of re-running retrieval/synthesis.

**Architecture:** New workspace package `packages/semantic_cache` (config/db/repository, mirroring `packages/persona`'s shape) provides `lookup()`/`write()` over one Mongo collection (`semantic_cache`). `ws.py`'s `/ws/search` handler embeds the query once via the existing `gateway.embed(role="query_embed", ...)` call, looks up a cache hit per active mode before dispatching `run_instant`/`run_ai_mode`, and on a miss writes the result back as a fire-and-forget background task using the same `_background_tasks` pattern already used for persona-signal and conversation-turn writes.

**Tech Stack:** Python 3.11, Motor (async Mongo driver), pydantic-settings, pytest + pytest-asyncio, FastAPI/Starlette `TestClient` websocket testing.

**Spec:** `docs/superpowers/specs/2026-08-18-semantic-response-caching-design.md`

## Global Constraints

- Mongo deployment is Atlas — cache lookup uses a real `$vectorSearch` aggregation stage in production code; tests fake it with an in-memory brute-force cosine-similarity collection (no `mongomock`, no real test Mongo — this repo's convention, per `packages/persona/tests/conftest.py`).
- Default similarity threshold: `0.95`, configurable via `SEMANTIC_CACHE_THRESHOLD` env var (field `semantic_cache_threshold: float = 0.95` — no `env_prefix` anywhere in this repo's settings classes).
- No TTL/expiry in this plan (deferred per spec) — cache entries are kept indefinitely.
- Cache is global/unscoped — no per-user or per-persona key component (confirmed safe: allowlist filters derive from query text, not user identity).
- Agentic search (`packages/agents`) is explicitly out of scope.
- Instant mode's `rerank` flag changes `run_instant()`'s return shape entirely (`{"es_error", "milvus_error", ...}` vs `{"reranked", "reranked_error"}` — see `packages/retrieval-api/src/retrieval_api/instant/search.py:103-142`). To avoid ever replaying a mismatched shape, Instant mode caching is keyed on **`"instant_rerank"`** when `rerank=True` and **`"instant"`** when `rerank=False` — two separate cache pools, both filtered under the collection's single `mode` field. This is an implementation-level refinement of the spec (which only names `"instant"`/`"ai_mode"` as mode values) made to prevent a genuine shape-mismatch bug; it does not change scope or user-visible behavior.
- Cached AI Mode `result` is the **full dict returned by `run_ai_mode()`** (`{"ok", "answer", "citations", "intent", "reasoning"?}`), not the trimmed client-facing `ai_mode_message` — this lets a cache hit flow through the exact same downstream code (persona-signal write keyed on `intent`, chat persistence keyed on `answer`) with zero branching. Same principle for Instant: cached `result` is `run_instant()`'s full return dict.
- Package naming: Python import name `semantic_cache` (underscore); uv package/distribution name `semantic-cache` (dash) in `pyproject.toml`, matching the `common`/`model_gateway` vs `model-gateway` convention already in this repo.

---

### Task 1: Scaffold `packages/semantic_cache` — config and Mongo collection access

**Files:**
- Create: `packages/semantic_cache/pyproject.toml`
- Create: `packages/semantic_cache/src/semantic_cache/__init__.py` (empty)
- Create: `packages/semantic_cache/src/semantic_cache/config.py`
- Create: `packages/semantic_cache/src/semantic_cache/db.py`
- Create: `packages/semantic_cache/tests/__init__.py` (empty)
- Create: `packages/semantic_cache/tests/conftest.py`
- Create: `packages/semantic_cache/tests/test_config.py`
- Create: `packages/semantic_cache/tests/test_db.py`
- Modify: `pyproject.toml:7` (root workspace `members` list)
- Modify: `pyproject.toml:10-16` (root workspace `[tool.uv.sources]`)

**Interfaces:**
- Produces: `semantic_cache.config.SemanticCacheSettings` (fields `mongo_uri: str`, `mongo_db: str`, `semantic_cache_threshold: float = 0.95`), `semantic_cache.config.get_semantic_cache_settings() -> SemanticCacheSettings` (`@lru_cache`'d).
- Produces: `semantic_cache.db.get_mongo_client(settings: SemanticCacheSettings) -> AsyncIOMotorClient` (`@lru_cache`'d), `semantic_cache.db.get_semantic_cache_collection(client: AsyncIOMotorClient, settings: SemanticCacheSettings) -> AsyncIOMotorCollection` (collection name `"semantic_cache"`).

- [ ] **Step 1: Write the failing tests for config and db**

`packages/semantic_cache/tests/conftest.py`:
```python
import os

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-semantic-cache-db")
```

`packages/semantic_cache/tests/test_config.py`:
```python
from semantic_cache.config import get_semantic_cache_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-semantic-cache-db"
    get_semantic_cache_settings.cache_clear()


def test_settings_default_threshold(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    monkeypatch.delenv("SEMANTIC_CACHE_THRESHOLD", raising=False)
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.semantic_cache_threshold == 0.95
    get_semantic_cache_settings.cache_clear()


def test_settings_threshold_overridable(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGO_DB", "test-semantic-cache-db")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.9")
    get_semantic_cache_settings.cache_clear()
    settings = get_semantic_cache_settings()
    assert settings.semantic_cache_threshold == 0.9
    get_semantic_cache_settings.cache_clear()
```

`packages/semantic_cache/tests/test_db.py`:
```python
from semantic_cache.config import get_semantic_cache_settings
from semantic_cache.db import get_mongo_client, get_semantic_cache_collection


def test_get_semantic_cache_collection_selects_configured_db_and_collection_name():
    settings = get_semantic_cache_settings()
    client = get_mongo_client(settings)
    collection = get_semantic_cache_collection(client, settings)
    assert collection.name == "semantic_cache"
    assert collection.database.name == settings.mongo_db


def test_get_mongo_client_caches_client():
    settings = get_semantic_cache_settings()
    client1 = get_mongo_client(settings)
    client2 = get_mongo_client(settings)
    assert client1 is client2, "Expected cached client to return the same object"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/semantic_cache && uv run pytest tests/ -v` (package doesn't exist yet, or run from repo root once workspace is registered in step 5 — for now this will fail with `ModuleNotFoundError: No module named 'semantic_cache'`)
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cache'`

- [ ] **Step 3: Write `pyproject.toml`, `config.py`, `db.py`**

`packages/semantic_cache/pyproject.toml`:
```toml
[project]
name = "semantic-cache"
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
packages = ["src/semantic_cache"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`packages/semantic_cache/src/semantic_cache/__init__.py`: empty file.

`packages/semantic_cache/src/semantic_cache/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticCacheSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str
    semantic_cache_threshold: float = 0.95


@lru_cache
def get_semantic_cache_settings() -> SemanticCacheSettings:
    return SemanticCacheSettings()
```

`packages/semantic_cache/src/semantic_cache/db.py`:
```python
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from semantic_cache.config import SemanticCacheSettings


@lru_cache
def get_mongo_client(settings: SemanticCacheSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_semantic_cache_collection(
    client: AsyncIOMotorClient, settings: SemanticCacheSettings,
) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["semantic_cache"]
```

- [ ] **Step 4: Register the new package in the root workspace**

In root `pyproject.toml`, change line 7 from:
```toml
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents", "packages/auth", "packages/persona", "packages/chat"]
```
to:
```toml
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents", "packages/auth", "packages/persona", "packages/chat", "packages/semantic_cache"]
```

And add a line to `[tool.uv.sources]` (after `chat = { workspace = true }`):
```toml
semantic-cache = { workspace = true }
```

Run: `uv sync --all-packages` (repo root — NOT bare `uv sync`, per this repo's CLAUDE.md, to avoid dropping editable installs of other workspace members)
Expected: completes without error, `semantic-cache` listed among installed workspace packages.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/semantic_cache/tests -v`
Expected: PASS (5 tests: 3 config, 2 db)

- [ ] **Step 6: Commit**

```bash
git add packages/semantic_cache pyproject.toml uv.lock
git commit -m "feat(semantic-cache): scaffold config and Mongo collection access"
```

---

### Task 2: `repository.py` — cache lookup and write

**Files:**
- Create: `packages/semantic_cache/src/semantic_cache/repository.py`
- Modify: `packages/semantic_cache/tests/conftest.py` (add fake vector-search collection fixture)
- Create: `packages/semantic_cache/tests/test_repository.py`

**Interfaces:**
- Consumes: nothing beyond stdlib — this module takes a Mongo collection object as its first argument (matching `persona.repository`'s style of taking `personas`/`collection` rather than importing `db.py` directly), so it's exercised in tests against a fake collection.
- Produces: `semantic_cache.repository.lookup(collection, mode: str, query_embedding: list[float], threshold: float) -> dict | None` (returns the cached `result` dict on a hit at/above `threshold`, else `None`). `semantic_cache.repository.write(collection, mode: str, query_text: str, query_embedding: list[float], result: dict) -> None`.

- [ ] **Step 1: Write the failing tests, including the fake vector-search collection fixture**

Add to `packages/semantic_cache/tests/conftest.py` (append after the existing env-var setdefault lines):
```python
import pytest


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _FakeCursor:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for item in self._items:
            yield item


class FakeSemanticCacheCollection:
    """In-memory stand-in for the Atlas $vectorSearch-backed collection.

    Simulates Atlas's $vectorSearch aggregation stage via brute-force cosine
    similarity, since there is no local Atlas cluster to test against and
    this repo's convention (see packages/persona/tests/conftest.py) is a
    hand-rolled in-memory fake rather than mongomock or a real test Mongo.
    """

    def __init__(self):
        self.documents: list[dict] = []

    async def insert_one(self, document: dict) -> None:
        self.documents.append(document)

    def aggregate(self, pipeline: list[dict]):
        stage = pipeline[0]["$vectorSearch"]
        query_vector = stage["queryVector"]
        mode_filter = stage.get("filter", {}).get("mode")
        limit = stage.get("limit", 1)

        candidates = [
            doc for doc in self.documents
            if mode_filter is None or doc["mode"] == mode_filter
        ]
        scored = [
            {**doc, "score": _cosine_similarity(query_vector, doc["query_embedding"])}
            for doc in candidates
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return _FakeCursor(scored[:limit])


@pytest.fixture
def fake_semantic_cache_collection():
    return FakeSemanticCacheCollection()
```

`packages/semantic_cache/tests/test_repository.py`:
```python
import pytest

from semantic_cache.repository import lookup, write


@pytest.mark.asyncio
async def test_lookup_returns_none_when_empty(fake_semantic_cache_collection):
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result is None


@pytest.mark.asyncio
async def test_write_then_lookup_exact_match_hits(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C",
        [1.0, 0.0], {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]},
    )
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result == {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]}


@pytest.mark.asyncio
async def test_lookup_below_threshold_misses(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C",
        [1.0, 0.0], {"ok": True, "answer": "cached answer", "citations": [], "intent": []},
    )
    # Orthogonal vector -> cosine similarity 0.0, well below any reasonable threshold.
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [0.0, 1.0], threshold=0.95)
    assert result is None


@pytest.mark.asyncio
async def test_lookup_is_scoped_by_mode(fake_semantic_cache_collection):
    await write(
        fake_semantic_cache_collection, "instant", "gst rate",
        [1.0, 0.0], {"es_error": None, "milvus_error": None, "es": [], "milvus": [], "milvus_sparse": []},
    )
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0], threshold=0.95)
    assert result is None, "a doc cached under mode='instant' must not satisfy an 'ai_mode' lookup"


@pytest.mark.asyncio
async def test_lookup_returns_closest_of_multiple_candidates(fake_semantic_cache_collection):
    await write(fake_semantic_cache_collection, "ai_mode", "q1", [1.0, 0.0, 0.0], {"answer": "first"})
    await write(fake_semantic_cache_collection, "ai_mode", "q2", [0.99, 0.14, 0.0], {"answer": "second"})
    result = await lookup(fake_semantic_cache_collection, "ai_mode", [1.0, 0.0, 0.0], threshold=0.95)
    assert result == {"answer": "first"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/semantic_cache/tests/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_cache.repository'` (or `ImportError`)

- [ ] **Step 3: Write `repository.py`**

```python
from datetime import datetime, timezone

_VECTOR_INDEX_NAME = "semantic_cache_vector_index"
_NUM_CANDIDATES = 100


async def lookup(
    collection, mode: str, query_embedding: list[float], threshold: float,
) -> dict | None:
    pipeline = [
        {
            "$vectorSearch": {
                "index": _VECTOR_INDEX_NAME,
                "path": "query_embedding",
                "queryVector": query_embedding,
                "numCandidates": _NUM_CANDIDATES,
                "limit": 1,
                "filter": {"mode": mode},
            },
        },
        {
            "$project": {
                "result": 1,
                "score": {"$meta": "vectorSearchScore"},
            },
        },
    ]
    docs = [doc async for doc in collection.aggregate(pipeline)]
    if not docs:
        return None
    top = docs[0]
    if top["score"] < threshold:
        return None
    return top["result"]


async def write(
    collection, mode: str, query_text: str, query_embedding: list[float], result: dict,
) -> None:
    await collection.insert_one({
        "mode": mode,
        "query_text": query_text,
        "query_embedding": query_embedding,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
```

Note: the fake collection's `aggregate()` (Task 2 Step 1) reads `stage["filter"]["mode"]` and `stage["queryVector"]`/`stage["limit"]` directly off the `$vectorSearch` stage dict, and computes `"score"` itself via brute-force cosine similarity — it does not interpret the second `$project` stage at all (the fake only ever looks at `pipeline[0]`). This is fine: the fake's job is to stand in for what *Atlas* would return already scored and filtered, not to be a general aggregation engine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/semantic_cache/tests -v`
Expected: PASS (all tests from Task 1 and Task 2 — 10 total)

- [ ] **Step 5: Commit**

```bash
git add packages/semantic_cache
git commit -m "feat(semantic-cache): add cache lookup/write repository functions"
```

---

### Task 3: Wire semantic caching into `/ws/search`

**Files:**
- Modify: `packages/retrieval-api/pyproject.toml` (add `semantic-cache` dependency)
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py` (imports + `search()` handler)
- Modify: `packages/retrieval-api/tests/test_ws_integration.py` (new tests)

**Interfaces:**
- Consumes: `semantic_cache.config.get_semantic_cache_settings()`, `semantic_cache.db.get_mongo_client()`/`get_semantic_cache_collection()`, `semantic_cache.repository.lookup()`/`write()` (all from Tasks 1-2). `gateway.embed(role: str, text: str) -> list[float]` (existing, `packages/retrieval-api/src/retrieval_api/gateway_client.py:72`).
- Produces: no new public interface — this is the integration point; behavior is verified via `/ws/search` websocket tests.

- [ ] **Step 1: Add the workspace dependency**

In `packages/retrieval-api/pyproject.toml`, add `"semantic-cache"` to that package's `dependencies` list (alongside the existing `"chat"`, `"persona"` entries — match however those are currently listed there).

Run: `uv sync --all-packages`
Expected: completes without error.

- [ ] **Step 2: Write the failing tests**

Add to `packages/retrieval-api/tests/test_ws_integration.py` (follow the existing file's fixture/monkeypatch style — `ws_module = retrieval_api.ws`, `AsyncMock`/`Mock` for clients, `monkeypatch.setattr(ws_module, ...)`):

```python
from unittest.mock import AsyncMock, Mock

import pytest

from semantic_cache.repository import write as cache_write


class _FakeEmbedGateway:
    def __init__(self, embedding):
        self._embedding = embedding

    async def embed(self, role, text):
        assert role == "query_embed"
        return self._embedding


@pytest.mark.asyncio
async def test_ai_mode_cache_hit_skips_run_ai_mode_and_returns_cached_answer(
    monkeypatch, fake_semantic_cache_collection,
):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("run_ai_mode should not be called on a cache hit")

    monkeypatch.setattr(ws_module, "run_ai_mode", fail_if_called)
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))

    await cache_write(
        fake_semantic_cache_collection, "ai_mode", "what is section 80C", [1.0, 0.0],
        {"ok": True, "answer": "cached answer", "citations": [], "intent": ["acts"]},
    )

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {"type": "ai_mode_done", "answer": "cached answer", "citations": []}


@pytest.mark.asyncio
async def test_ai_mode_cache_miss_runs_pipeline_and_writes_back(
    monkeypatch, fake_semantic_cache_collection,
):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))
    monkeypatch.setattr(
        ws_module, "get_semantic_cache_collection", lambda *_: fake_semantic_cache_collection,
    )
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "fresh answer", "citations": [], "intent": ["acts"],
    }))

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {"type": "ai_mode_done", "answer": "fresh answer", "citations": []}

    for _ in range(50):
        cached = await cache_lookup_helper(fake_semantic_cache_collection)
        if cached is not None:
            break
        await asyncio.sleep(0.01)

    assert cached == {"ok": True, "answer": "fresh answer", "citations": [], "intent": ["acts"]}


async def cache_lookup_helper(collection):
    from semantic_cache.repository import lookup
    return await lookup(collection, "ai_mode", [1.0, 0.0], threshold=0.95)


@pytest.mark.asyncio
async def test_cache_lookup_failure_degrades_to_normal_pipeline(monkeypatch):
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: _FakeEmbedGateway([1.0, 0.0]))

    class _BrokenCollection:
        def aggregate(self, pipeline):
            raise RuntimeError("Atlas unreachable")

        async def insert_one(self, document):
            raise RuntimeError("Atlas unreachable")

    monkeypatch.setattr(ws_module, "get_semantic_cache_collection", lambda *_: _BrokenCollection())
    monkeypatch.setattr(ws_module, "run_instant", AsyncMock(return_value={
        "es": [], "es_error": None, "milvus": [], "milvus_sparse": [], "milvus_error": None,
    }))
    monkeypatch.setattr(ws_module, "run_ai_mode", AsyncMock(return_value={
        "ok": True, "answer": "fresh answer despite cache error", "citations": [], "intent": [],
    }))

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "what is section 80C", "mode": "ai_mode"})
        message = websocket.receive_json()

    assert message == {
        "type": "ai_mode_done", "answer": "fresh answer despite cache error", "citations": [],
    }
```

Also add a fixture import at the top of the test file (or in `conftest.py` if the file doesn't already import package-local fixtures cross-package): the `fake_semantic_cache_collection` fixture from `packages/semantic_cache/tests/conftest.py` is not automatically visible to `packages/retrieval-api/tests/` — copy the `FakeSemanticCacheCollection` class and fixture into `packages/retrieval-api/tests/conftest.py` as its own fixture (matching how `fake_conversations_collection`/`fake_personas_collection` already live directly in `packages/retrieval-api/tests/conftest.py`, not imported from `chat`/`persona`'s own test suites).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v -k cache`
Expected: FAIL — `AttributeError: module 'retrieval_api.ws' has no attribute 'get_semantic_cache_collection'` (or the handler runs `run_ai_mode` unconditionally, failing the `fail_if_called` assertion).

- [ ] **Step 4: Update `ws.py` imports**

Add to the import block (`packages/retrieval-api/src/retrieval_api/ws.py`, after the existing `persona`/`chat` imports):
```python
from semantic_cache.config import get_semantic_cache_settings
from semantic_cache.db import get_semantic_cache_collection, get_mongo_client as get_cache_mongo_client
from semantic_cache.repository import lookup as cache_lookup, write as cache_write
```

- [ ] **Step 5: Rewrite the `search()` handler**

Replace the full body of `async def search(websocket: WebSocket):` in `ws.py` with:

```python
@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"
    trace = message.get("trace", False)
    rerank = message.get("rerank", False)
    access_token = message.get("access_token")
    user_id = _resolve_user_id(access_token)
    conversation_id = message.get("conversation_id")

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        logger.exception("Milvus connection failed; proceeding without Milvus for this request")
        milvus_client = None

    personas_collection = None
    persona_context = ""
    if user_id is not None:
        try:
            persona_settings = get_persona_settings()
            mongo_client = get_mongo_client(persona_settings)
            personas_collection = get_personas_collection(mongo_client, persona_settings)
            persona = await get_persona(personas_collection, user_id)
            persona_context = render_persona_context(persona)
        except Exception:
            logger.exception("Persona lookup failed for user %r; proceeding without persona context", user_id)
            personas_collection = None
            persona_context = ""

    cache_settings = get_semantic_cache_settings()
    cache_mongo_client = get_cache_mongo_client(cache_settings)
    cache_collection = get_semantic_cache_collection(cache_mongo_client, cache_settings)
    instant_cache_key = "instant_rerank" if rerank else "instant"

    query_embedding = None
    try:
        query_embedding = await gateway.embed(role="query_embed", text=query)
    except Exception:
        logger.exception("Query embedding for semantic cache failed; proceeding without cache")

    instant_cache_hit = None
    ai_mode_cache_hit = None
    if query_embedding is not None:
        if mode in ("instant", "both"):
            try:
                instant_cache_hit = await cache_lookup(
                    cache_collection, instant_cache_key, query_embedding,
                    cache_settings.semantic_cache_threshold,
                )
            except Exception:
                logger.exception("Semantic cache lookup failed for instant mode; proceeding without cache")
        if mode in ("ai_mode", "both"):
            try:
                ai_mode_cache_hit = await cache_lookup(
                    cache_collection, "ai_mode", query_embedding,
                    cache_settings.semantic_cache_threshold,
                )
            except Exception:
                logger.exception("Semantic cache lookup failed for ai_mode; proceeding without cache")

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

    if access_token and user_id is None:
        await send({"type": "session_expired"})

    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span", name="ws-search", input={"query": query, "mode": mode},
        ) as root_span:
            instant_task = (
                asyncio.create_task(
                    run_instant(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None, rerank=rerank,
                    )
                )
                if mode in ("instant", "both") and instant_cache_hit is None else None
            )
            ai_mode_task = (
                asyncio.create_task(
                    run_ai_mode(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None,
                        persona_context=persona_context,
                    )
                )
                if mode in ("ai_mode", "both") and ai_mode_cache_hit is None else None
            )

            output: dict = {}

            if instant_cache_hit is not None or instant_task is not None:
                if instant_cache_hit is not None:
                    instant_result = instant_cache_hit
                else:
                    instant_result = await instant_task
                    if instant_result["es_error"] is None and instant_result["milvus_error"] is None:
                        write_task = asyncio.create_task(
                            cache_write(
                                cache_collection, instant_cache_key, query, query_embedding, instant_result,
                            )
                        )
                        _background_tasks.add(write_task)
                        write_task.add_done_callback(_background_tasks.discard)
                output["instant_ok"] = instant_result["es_error"] is None and instant_result["milvus_error"] is None
                root_span.update(metadata={
                    "instant_es_error": instant_result["es_error"] or "",
                    "instant_milvus_error": instant_result["milvus_error"] or "",
                })
                await send({"type": "instant_result", **instant_result})

            if ai_mode_cache_hit is not None or ai_mode_task is not None:
                if ai_mode_cache_hit is not None:
                    ai_mode_result = ai_mode_cache_hit
                else:
                    ai_mode_result = await ai_mode_task
                    if ai_mode_result["ok"]:
                        write_task = asyncio.create_task(
                            cache_write(cache_collection, "ai_mode", query, query_embedding, ai_mode_result)
                        )
                        _background_tasks.add(write_task)
                        write_task.add_done_callback(_background_tasks.discard)

                if ai_mode_result["ok"]:
                    output["answer"] = ai_mode_result["answer"]
                    ai_mode_message = {
                        "type": "ai_mode_done", "answer": ai_mode_result["answer"],
                        "citations": ai_mode_result["citations"],
                    }
                    if ai_mode_result.get("reasoning"):
                        ai_mode_message["reasoning"] = ai_mode_result["reasoning"]
                    await send(ai_mode_message)

                    if user_id is not None and personas_collection is not None:
                        task = asyncio.create_task(
                            record_persona_signal(
                                personas_collection, gateway, user_id, query,
                                categories=ai_mode_result.get("intent", []),
                            )
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)

                    if user_id is not None and conversation_id is not None:
                        try:
                            chat_settings = get_chat_settings()
                            chat_mongo_client = get_chat_mongo_client(chat_settings)
                            conversations_collection = get_conversations_collection(chat_mongo_client, chat_settings)
                            chat_task = asyncio.create_task(
                                record_conversation_turn(
                                    conversations_collection, conversation_id, user_id, _title_from_query(query),
                                    [
                                        {"role": "user", "text": query},
                                        {"role": "assistant", "text": ai_mode_result["answer"]},
                                    ],
                                )
                            )
                            _background_tasks.add(chat_task)
                            chat_task.add_done_callback(_background_tasks.discard)
                        except Exception:
                            logger.exception("Failed to schedule conversation write for user %r", user_id)
                else:
                    output["ai_mode_error"] = ai_mode_result["error"]
                    await send({"type": "ai_mode_error", "error": ai_mode_result["error"]})

            root_span.update(output=output)
            root_span.set_trace_io(input={"query": query, "mode": mode}, output=output)

        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
        langfuse.flush()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v -k cache`
Expected: PASS (3 new tests)

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, all 4 packages' tests plus the new package (no regressions in existing `/ws/search` tests — the cache-miss path with `query_embedding` unavailable, e.g. existing tests that don't mock `get_gateway_client`'s `.embed`, must still work; if any pre-existing test's fake gateway lacks an `embed` method, add a minimal `async def embed(self, role, text): return [0.0]` to that fake so the handler's `try/except` around the embed call still degrades cleanly rather than the fake raising `AttributeError` — which the `except Exception` clause already catches, so no behavior change is needed, only confirm no unrelated test asserts on `logger.exception` call counts).

- [ ] **Step 8: Commit**

```bash
git add packages/retrieval-api pyproject.toml uv.lock
git commit -m "feat(semantic-cache): wire semantic cache lookup/write into /ws/search"
```

---

## Self-Review Notes

- **Spec coverage:** data model (Task 1-2), Atlas `$vectorSearch` lookup + threshold config (Task 2), lookup/write flow hooked into `ws.py` before/after pipeline dispatch (Task 3), error handling — lookup/write failures degrade to a normal pipeline run (Task 3 test 3), background-task write pattern reused (Task 3), no TTL (Global Constraints), global/unscoped cache (Global Constraints). All covered.
- **Placeholder scan:** none — every step has literal code, exact file paths, and concrete run/expect pairs.
- **Type consistency:** `lookup(collection, mode, query_embedding, threshold) -> dict | None` and `write(collection, mode, query_text, query_embedding, result) -> None` signatures are identical across Task 2's implementation, tests, and Task 3's `ws.py` call sites.
- **Known deviation from spec wording, called out explicitly in Global Constraints:** cached `result` is the full pipeline-internal dict (not the trimmed client message), and Instant mode uses two mode-key variants (`instant`/`instant_rerank`) instead of one — both are implementation-necessary refinements, not scope changes.
