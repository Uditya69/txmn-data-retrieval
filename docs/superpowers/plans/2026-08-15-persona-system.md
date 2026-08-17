# Persona System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give logged-in users a flat, evolving persona (category affinity + inferred
expertise/style) that's read on every AI Mode request to shape synthesis tone/depth, and written
to asynchronously after each response from signal already available in the pipeline plus one
cheap extra model call. Guests are completely unaffected.

**Architecture:** New workspace package `packages/persona` holds pure, DB-agnostic logic (merge
math, repository read/write) unit-tested against an in-memory fake, mirroring `packages/auth`'s
`service.py`/`tests/fakes.py` pattern and `packages/common`'s `FakeAsyncES`/`FakeMilvusClient`
pattern. `retrieval-api` owns all HTTP/gateway/websocket-specific orchestration: resolving
`user_id` from the access token carried in the `/ws/search` message payload (not a header — this
is a WebSocket route, and the existing protocol already carries fields like `query`/`mode` in the
first JSON message), loading/rendering persona context before the AI Mode pipeline runs, and
scheduling the async write after it completes.

**Tech Stack:** Same as `packages/auth` — `motor` for MongoDB, `pydantic-settings` for config.
Reuses the existing `GatewayClient` (`packages/retrieval-api/src/retrieval_api/gateway_client.py`)
and its `role="slm"` chat role for the one new cheap model call — no new model-gateway role or
config is introduced.

**Spec:** `docs/superpowers/specs/2026-08-15-user-persona-system-design.md` (Persona storage /
Extraction pipeline sections)

**Depends on:** `docs/superpowers/plans/2026-08-15-auth-service.md` must be merged first —
this plan uses `auth.security.decode_access_token` and `auth.config.get_auth_settings` to resolve
`user_id`.

## Global Constraints

- Python 3.11, not 3.14 (root `CLAUDE.md` hard rule 5).
- Persona must **never** feed into RRF fusion weights or Milvus collection routing — only into
  prompt text (synthesis system prompt). This is hard rule 4 territory; don't touch
  `common/schemas.py::collections_for_intent` or any RRF weight in this plan.
- Guest requests (no/invalid access token) must produce byte-identical behavior to today — no
  persona load, no persona write, no latency added to the response path.
- The async persona write must never block or fail the user-visible response — schedule it
  as fire-and-forget (`asyncio.create_task`), and any exception inside it must be caught and
  logged, never raised into the websocket handler.
- `pydantic-settings` field names must match env var names exactly (root `CLAUDE.md` gotcha).
- Persona reuses the **same** `MONGO_URI`/`MONGO_DB` env vars the auth plan already added to
  `.env.example` — different collection (`personas` vs `users`) in the same database, not a
  second Mongo dependency.

## File Structure

```
packages/persona/
  pyproject.toml
  src/persona/
    __init__.py
    config.py       # PersonaSettings: mongo_uri, mongo_db (same field names as auth.config.AuthSettings)
    db.py           # get_mongo_client(settings), get_personas_collection(client, settings)
    merge.py        # pure functions: merge_category_affinity, merge_expertise_patch
    repository.py   # get_persona(personas, user_id), record_signal(personas, user_id, categories, expertise_patch)
    prompt.py       # render_persona_context(persona) -> str
  tests/
    conftest.py
    fakes.py        # FakePersonasCollection
    test_config.py
    test_db.py
    test_merge.py
    test_repository.py
    test_prompt.py
```

Wiring into `retrieval-api`:

```
pyproject.toml                                              # add packages/persona to workspace
packages/retrieval-api/pyproject.toml                       # add "persona" dependency
packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py   # accept persona_context param
packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py     # thread persona_context through; return intent categories
packages/retrieval-api/src/retrieval_api/ai_mode/persona_signal.py  # new: cheap expertise/style extraction + write orchestration
packages/retrieval-api/src/retrieval_api/ws.py               # resolve user_id, load/render persona, schedule async write
```

---

### Task 1: `persona` package scaffold + config

**Files:**
- Create: `packages/persona/pyproject.toml`
- Create: `packages/persona/src/persona/__init__.py`
- Create: `packages/persona/src/persona/config.py`
- Create: `packages/persona/tests/conftest.py`
- Create: `packages/persona/tests/test_config.py`

**Interfaces:**
- Produces: `PersonaSettings` (pydantic-settings `BaseSettings`): `mongo_uri: str`, `mongo_db:
  str`. `get_persona_settings() -> PersonaSettings` (`lru_cache`d).

- [ ] **Step 1: Write `packages/persona/pyproject.toml`**

```toml
[project]
name = "persona"
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
packages = ["src/persona"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Write `packages/persona/src/persona/__init__.py`** (empty file)

- [ ] **Step 3: Write `packages/persona/tests/conftest.py`**

```python
import os

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB", "test-auth-db")
```

- [ ] **Step 4: Write the failing test `packages/persona/tests/test_config.py`**

```python
from persona.config import get_persona_settings


def test_settings_load_from_env():
    settings = get_persona_settings()
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test-auth-db"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd packages/persona && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`. Before this can pass, add `"packages/persona"` to
`members` and `persona = { workspace = true }` to `[tool.uv.sources]` in the root
`pyproject.toml`, then run `uv sync --all-packages` from repo root.

- [ ] **Step 6: Write minimal implementation `packages/persona/src/persona/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings


class PersonaSettings(BaseSettings):
    mongo_uri: str
    mongo_db: str


@lru_cache
def get_persona_settings() -> PersonaSettings:
    return PersonaSettings()
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd packages/persona && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml packages/persona
git commit -m "feat(persona): scaffold persona package with settings config"
```

---

### Task 2: Real MongoDB wiring

**Files:**
- Create: `packages/persona/src/persona/db.py`
- Create: `packages/persona/tests/test_db.py`

**Interfaces:**
- Consumes: `PersonaSettings` from Task 1.
- Produces: `get_mongo_client(settings: PersonaSettings) -> AsyncIOMotorClient`,
  `get_personas_collection(client, settings) -> AsyncIOMotorCollection` (selects
  `client[settings.mongo_db]["personas"]`).

- [ ] **Step 1: Write the failing test `packages/persona/tests/test_db.py`**

```python
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_personas_collection


def test_get_personas_collection_selects_configured_db_and_collection_name():
    settings = get_persona_settings()
    client = get_mongo_client(settings)
    try:
        collection = get_personas_collection(client, settings)
        assert collection.name == "personas"
        assert collection.database.name == settings.mongo_db
    finally:
        client.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/persona && uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.db'`

- [ ] **Step 3: Write minimal implementation `packages/persona/src/persona/db.py`**

```python
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from persona.config import PersonaSettings


def get_mongo_client(settings: PersonaSettings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongo_uri)


def get_personas_collection(client: AsyncIOMotorClient, settings: PersonaSettings) -> AsyncIOMotorCollection:
    return client[settings.mongo_db]["personas"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/persona && uv run pytest tests/test_db.py -v`
Expected: PASS (no live Mongo needed — `motor` connects lazily, same as `packages/auth`).

- [ ] **Step 5: Commit**

```bash
git add packages/persona/src/persona/db.py packages/persona/tests/test_db.py
git commit -m "feat(persona): wire real MongoDB client and personas collection"
```

---

### Task 3: Merge math (pure functions)

**Files:**
- Create: `packages/persona/src/persona/merge.py`
- Create: `packages/persona/tests/test_merge.py`

**Interfaces:**
- Produces:
  - `KNOWN_CATEGORIES: list[str]` = `["acts", "rules", "caselaws", "articles", "commentary", "tariff"]`
    (matches the categories `extract_intent()` can tag, per `common/schemas.py`).
  - `merge_category_affinity(existing_affinity: dict[str, float], existing_count: int,
    categories: list[str]) -> dict[str, float]` — rolling average per known category: each
    category present in `categories` this round counts as `1.0`, absent counts as `0.0`; new
    average = `(existing_avg * existing_count + indicator) / (existing_count + 1)` for every
    entry in `KNOWN_CATEGORIES`. `existing_affinity` may be `{}` (first-ever signal for a user).
  - `merge_expertise_patch(existing: dict, patch: dict | None) -> dict` — returns `existing`
    unchanged if `patch` is `None` or empty; otherwise returns a new dict with only the keys
    present in `patch` overwritten (`expertise_level`, `query_style`), all other existing keys
    preserved.

- [ ] **Step 1: Write the failing tests `packages/persona/tests/test_merge.py`**

```python
from persona.merge import KNOWN_CATEGORIES, merge_category_affinity, merge_expertise_patch


def test_merge_category_affinity_starts_from_empty():
    result = merge_category_affinity({}, existing_count=0, categories=["caselaws"])
    assert result["caselaws"] == 1.0
    assert result["acts"] == 0.0
    assert set(result.keys()) == set(KNOWN_CATEGORIES)


def test_merge_category_affinity_averages_across_rounds():
    first = merge_category_affinity({}, existing_count=0, categories=["caselaws"])
    second = merge_category_affinity(first, existing_count=1, categories=["acts"])
    assert second["caselaws"] == 0.5
    assert second["acts"] == 0.5
    assert second["commentary"] == 0.0


def test_merge_category_affinity_handles_multiple_tags_in_one_round():
    result = merge_category_affinity({}, existing_count=0, categories=["acts", "rules"])
    assert result["acts"] == 1.0
    assert result["rules"] == 1.0
    assert result["caselaws"] == 0.0


def test_merge_expertise_patch_returns_existing_unchanged_when_patch_none():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    assert merge_expertise_patch(existing, None) == existing


def test_merge_expertise_patch_overwrites_only_provided_keys():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    result = merge_expertise_patch(existing, {"expertise_level": "student"})
    assert result == {"expertise_level": "student", "query_style": "precise-citation"}


def test_merge_expertise_patch_handles_empty_existing():
    result = merge_expertise_patch({}, {"expertise_level": "student"})
    assert result == {"expertise_level": "student"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/persona && uv run pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.merge'`

- [ ] **Step 3: Write minimal implementation `packages/persona/src/persona/merge.py`**

```python
KNOWN_CATEGORIES = ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]


def merge_category_affinity(
    existing_affinity: dict[str, float], existing_count: int, categories: list[str],
) -> dict[str, float]:
    result = {}
    for category in KNOWN_CATEGORIES:
        prior = existing_affinity.get(category, 0.0)
        indicator = 1.0 if category in categories else 0.0
        result[category] = (prior * existing_count + indicator) / (existing_count + 1)
    return result


def merge_expertise_patch(existing: dict, patch: dict | None) -> dict:
    if not patch:
        return existing
    return {**existing, **patch}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/persona && uv run pytest tests/test_merge.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/persona/src/persona/merge.py packages/persona/tests/test_merge.py
git commit -m "feat(persona): add category affinity and expertise merge math"
```

---

### Task 4: Repository — read and record signal against a fake collection

**Files:**
- Create: `packages/persona/tests/fakes.py`
- Create: `packages/persona/src/persona/repository.py`
- Create: `packages/persona/tests/test_repository.py`

**Interfaces:**
- Consumes: `merge_category_affinity`, `merge_expertise_patch`, `KNOWN_CATEGORIES` from Task 3
  (`persona.merge`).
- Produces:
  - `async def get_persona(personas, user_id: str) -> dict | None` — returns the stored
    document, or `None` if the user has no persona yet.
  - `async def record_signal(personas, user_id: str, categories: list[str], expertise_patch:
    dict | None) -> dict` — reads the existing document (or treats it as absent/empty), merges
    `category_affinity` and `expertise_patch` via Task 3's pure functions, increments
    `query_count`, upserts the merged document, and returns it.
  - `personas` is any object exposing `async def find_one(filter: dict) -> dict | None` and
    `async def replace_one(filter: dict, replacement: dict, upsert: bool = False) -> None` — the
    shape both `tests/fakes.py::FakePersonasCollection` and the real `motor` collection (Task 2)
    satisfy.

- [ ] **Step 1: Write `packages/persona/tests/fakes.py`**

```python
class FakePersonasCollection:
    """In-memory stand-in for a motor AsyncIOMotorCollection, shaped to what
    persona.repository needs: find_one and replace_one. Mirrors
    packages/auth/tests/fakes.py::FakeUsersCollection.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        return self.documents.get(filter.get("user_id"))

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[filter["user_id"]] = replacement
```

- [ ] **Step 2: Write the failing tests `packages/persona/tests/test_repository.py`**

```python
import pytest

from persona.repository import get_persona, record_signal
from tests.fakes import FakePersonasCollection


@pytest.mark.asyncio
async def test_get_persona_returns_none_when_absent():
    personas = FakePersonasCollection()
    assert await get_persona(personas, "user-1") is None


@pytest.mark.asyncio
async def test_record_signal_creates_persona_on_first_call():
    personas = FakePersonasCollection()
    result = await record_signal(personas, "user-1", categories=["caselaws"], expertise_patch={"expertise_level": "student"})
    assert result["user_id"] == "user-1"
    assert result["category_affinity"]["caselaws"] == 1.0
    assert result["expertise_level"] == "student"
    assert result["query_count"] == 1


@pytest.mark.asyncio
async def test_record_signal_merges_into_existing_persona():
    personas = FakePersonasCollection()
    await record_signal(personas, "user-1", categories=["caselaws"], expertise_patch=None)
    result = await record_signal(personas, "user-1", categories=["acts"], expertise_patch={"query_style": "precise-citation"})
    assert result["category_affinity"]["caselaws"] == 0.5
    assert result["category_affinity"]["acts"] == 0.5
    assert result["query_style"] == "precise-citation"
    assert result["query_count"] == 2


@pytest.mark.asyncio
async def test_get_persona_returns_stored_document():
    personas = FakePersonasCollection()
    await record_signal(personas, "user-1", categories=["commentary"], expertise_patch=None)
    stored = await get_persona(personas, "user-1")
    assert stored["category_affinity"]["commentary"] == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/persona && uv run pytest tests/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.repository'`

- [ ] **Step 4: Write minimal implementation `packages/persona/src/persona/repository.py`**

```python
from datetime import datetime, timezone

from persona.merge import merge_category_affinity, merge_expertise_patch


async def get_persona(personas, user_id: str) -> dict | None:
    return await personas.find_one({"user_id": user_id})


async def record_signal(
    personas, user_id: str, categories: list[str], expertise_patch: dict | None,
) -> dict:
    existing = await personas.find_one({"user_id": user_id}) or {}
    existing_count = existing.get("query_count", 0)

    merged = merge_expertise_patch(existing, expertise_patch)
    merged["user_id"] = user_id
    merged["category_affinity"] = merge_category_affinity(
        existing.get("category_affinity", {}), existing_count, categories,
    )
    merged["query_count"] = existing_count + 1
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()

    await personas.replace_one({"user_id": user_id}, merged, upsert=True)
    return merged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/persona && uv run pytest tests/test_repository.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add packages/persona/src/persona/repository.py packages/persona/tests/fakes.py packages/persona/tests/test_repository.py
git commit -m "feat(persona): add persona repository with read-modify-write signal recording"
```

---

### Task 5: Persona-to-prompt rendering

**Files:**
- Create: `packages/persona/src/persona/prompt.py`
- Create: `packages/persona/tests/test_prompt.py`

**Interfaces:**
- Produces: `render_persona_context(persona: dict | None) -> str` — returns `""` if `persona` is
  `None` or has no `expertise_level`/`category_affinity` signal yet; otherwise returns a short
  natural-language line naming the top 1-2 categories by affinity (ties broken by
  `persona.merge.KNOWN_CATEGORIES` order) and the `expertise_level`/`query_style` if present.

- [ ] **Step 1: Write the failing tests `packages/persona/tests/test_prompt.py`**

```python
from persona.prompt import render_persona_context


def test_render_persona_context_returns_empty_string_for_none():
    assert render_persona_context(None) == ""


def test_render_persona_context_returns_empty_string_for_no_signal_yet():
    assert render_persona_context({"user_id": "u1", "query_count": 0, "category_affinity": {}}) == ""


def test_render_persona_context_names_top_category_and_expertise():
    persona = {
        "category_affinity": {"acts": 0.1, "caselaws": 0.8, "commentary": 0.1, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "expertise_level": "practitioner",
    }
    context = render_persona_context(persona)
    assert "caselaws" in context
    assert "practitioner" in context


def test_render_persona_context_includes_query_style_when_present():
    persona = {
        "category_affinity": {"acts": 0.9, "caselaws": 0.0, "commentary": 0.0, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "query_style": "precise-citation",
    }
    context = render_persona_context(persona)
    assert "precise-citation" in context
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/persona && uv run pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.prompt'`

- [ ] **Step 3: Write minimal implementation `packages/persona/src/persona/prompt.py`**

```python
from persona.merge import KNOWN_CATEGORIES


def render_persona_context(persona: dict | None) -> str:
    if not persona:
        return ""

    affinity = persona.get("category_affinity") or {}
    top_categories = [c for c in KNOWN_CATEGORIES if affinity.get(c, 0.0) > 0.0]
    top_categories.sort(key=lambda c: affinity.get(c, 0.0), reverse=True)

    expertise_level = persona.get("expertise_level")
    query_style = persona.get("query_style")

    if not top_categories and not expertise_level and not query_style:
        return ""

    parts = []
    if top_categories:
        parts.append(f"frequently asks about {', '.join(top_categories[:2])}")
    if expertise_level:
        parts.append(f"expertise level: {expertise_level}")
    if query_style:
        parts.append(f"query style: {query_style}")

    return "This user " + "; ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/persona && uv run pytest tests/test_prompt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/persona/src/persona/prompt.py packages/persona/tests/test_prompt.py
git commit -m "feat(persona): render persona documents into prompt context strings"
```

---

### Task 6: Thread `persona_context` through `synthesize`/`run_ai_mode` and return intent categories

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Modify: `packages/retrieval-api/tests/test_ai_mode_synthesize.py`
- Modify: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Produces: `synthesize(gateway, es_client, query, top_chunks, citations, on_step=None,
  model=None, persona_context: str = "")` — appends `persona_context` to the system prompt when
  non-empty. `run_ai_mode(gateway, es_client, milvus_client, query, on_step=None, persona_context:
  str = "")` — passes `persona_context` through to `synthesize`, and the returned result dict now
  additionally carries `"intent": intent_result["intent"]` (the category tags) so the caller
  (Task 7's `ws.py` changes) can use them for the persona write without re-running extraction.

- [ ] **Step 1: Add a failing test to `packages/retrieval-api/tests/test_ai_mode_synthesize.py`**

```python
@pytest.mark.asyncio
async def test_synthesize_appends_persona_context_to_system_prompt(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", None)

    await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}},
        persona_context="This user frequently asks about caselaws; expertise level: practitioner.",
    )

    system_message = gateway.chat_with_reasoning.call_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "expertise level: practitioner" in system_message["content"]


@pytest.mark.asyncio
async def test_synthesize_omits_persona_context_when_empty(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", None)

    await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}},
    )

    system_message = gateway.chat_with_reasoning.call_args.kwargs["messages"][0]
    assert system_message["content"].strip().endswith("genuinely unrelated to each other.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ai_mode_synthesize.py -v`
Expected: FAIL — `synthesize()` doesn't accept `persona_context` yet (`TypeError`).

- [ ] **Step 3: Modify `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`**

Change the `synthesize` signature and system prompt assembly:

```python
async def synthesize(
    gateway, es_client, query: str, top_chunks: list[dict], citations: dict,
    on_step: OnStep | None = None, model: str | None = None, persona_context: str = "",
) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}"

    if on_step is not None:
        await on_step("synthesis_prompt", {"prompt": prompt})

    system_prompt = _SYSTEM_PROMPT if not persona_context else f"{_SYSTEM_PROMPT}\n{persona_context}"

    answer, reasoning = await gateway.chat_with_reasoning(
        role="synthesis",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

    return {"answer": answer, "citations": citations, "reasoning": reasoning}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ai_mode_synthesize.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Add a failing test to `packages/retrieval-api/tests/test_ai_mode_pipeline.py`**

```python
@pytest.mark.asyncio
async def test_run_ai_mode_passes_persona_context_to_synthesize_and_returns_intent(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None):
        return {"original_query": query, "search_query": "rewritten", "intent": ["acts"], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, search_query, doc_id_allowlist, intent, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    received_persona_context = {}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        received_persona_context["value"] = persona_context
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(
        gateway=object(), es_client=object(), milvus_client=object(), query="q",
        persona_context="This user frequently asks about acts.",
    )

    assert received_persona_context["value"] == "This user frequently asks about acts."
    assert result["intent"] == ["acts"]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ai_mode_pipeline.py -v`
Expected: FAIL — `run_ai_mode()` doesn't accept `persona_context` yet, and doesn't return
`"intent"`.

- [ ] **Step 7: Modify `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`**

```python
async def run_ai_mode(
    gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None,
    persona_context: str = "",
) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="span", name="ai-mode", input={"query": query}) as root_span:
        try:
            with langfuse.start_as_current_observation(
                as_type="chain", name="extract-intent", input={"query": query},
            ) as span:
                intent_result = await extract_intent(gateway, query, on_step=on_step)
                span.update(output=intent_result)

            with langfuse.start_as_current_observation(
                as_type="retriever", name="resolve-allowlist", input={"filters": intent_result["filters"]},
            ) as span:
                doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"], on_step=on_step)
                span.update(output={"num_allowed": None if doc_id_allowlist is None else len(doc_id_allowlist)})

            with langfuse.start_as_current_observation(
                as_type="chain", name="retrieve", input={"search_query": intent_result["search_query"]},
            ) as span:
                candidates = await retrieve(
                    gateway, milvus_client, intent_result["search_query"], doc_id_allowlist,
                    intent_result["intent"], on_step=on_step,
                )
                span.update(output={"num_candidates": len(candidates)})

            with langfuse.start_as_current_observation(
                as_type="chain", name="rerank-and-prefetch", input={"query": query, "num_candidates": len(candidates)},
            ) as span:
                top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates, on_step=on_step)
                span.update(output={"num_top_chunks": len(top_chunks), "num_citations": len(citations)})

            with langfuse.start_as_current_observation(as_type="chain", name="synthesize", input={"query": query}) as span:
                synthesis = await synthesize(
                    gateway, es_client, query, top_chunks, citations, on_step=on_step,
                    persona_context=persona_context,
                )
                span.update(output=synthesis["answer"])
                if synthesis.get("reasoning"):
                    span.update(metadata={"reasoning": synthesis["reasoning"]})

            result = {
                "ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"],
                "intent": intent_result["intent"],
            }
            if synthesis.get("reasoning"):
                result["reasoning"] = synthesis["reasoning"]
            root_span.update(output=result)
            return result
        except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
            root_span.update(level="ERROR", status_message=str(exc), output={"ok": False, "error": str(exc)})
            return {"ok": False, "error": str(exc)}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ai_mode_pipeline.py tests/test_ai_mode_synthesize.py -v`
Expected: PASS (all tests, existing and new — note the existing
`test_run_ai_mode_success_path` asserts `result == {"ok": True, "answer": "final answer",
"citations": {"d1": {}}}`; this now fails because `result` gains an `"intent"` key. Update that
assertion in `test_ai_mode_pipeline.py` to `assert result == {"ok": True, "answer": "final
answer", "citations": {"d1": {}}, "intent": ["caselaws"]}`.)

- [ ] **Step 9: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py packages/retrieval-api/tests/test_ai_mode_synthesize.py packages/retrieval-api/tests/test_ai_mode_pipeline.py
git commit -m "feat(ai-mode): thread persona_context into synthesis and return intent categories"
```

---

### Task 7: Cheap expertise/style extraction + persona write orchestration

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/ai_mode/persona_signal.py`
- Create: `packages/retrieval-api/tests/test_persona_signal.py`

**Interfaces:**
- Consumes: `record_signal` from `persona.repository` (Task 4); `GatewayClient` shape (`.chat`
  method) from `retrieval_api.gateway_client`.
- Produces:
  - `async def extract_expertise_signal(gateway, query: str) -> dict` — one cheap `role="slm"`
    JSON-mode call (same role/pattern as `ai_mode/intent.py::extract_intent`), returns
    `{"expertise_level": ..., "query_style": ...}` (keys omitted if the model didn't confidently
    return them — never raises; malformed JSON returns `{}`).
  - `async def record_persona_signal(personas, gateway, user_id: str, query: str, categories:
    list[str]) -> None` — calls `extract_expertise_signal`, then `persona.repository.
    record_signal`. Any exception anywhere in this function is caught and logged, never
    propagated (this always runs as a fire-and-forget background task — see Task 8).

- [ ] **Step 1: Write the failing tests `packages/retrieval-api/tests/test_persona_signal.py`**

```python
import json
from unittest.mock import AsyncMock

import pytest

from persona.repository import get_persona
from retrieval_api.ai_mode.persona_signal import extract_expertise_signal, record_persona_signal
from tests_persona_fakes import FakePersonasCollection  # see Step 2

@pytest.mark.asyncio
async def test_extract_expertise_signal_parses_valid_json():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({"expertise_level": "practitioner", "query_style": "precise-citation"})

    result = await extract_expertise_signal(gateway, "Section 54F capital gains exemption query")

    assert result == {"expertise_level": "practitioner", "query_style": "precise-citation"}
    assert gateway.chat.call_args.kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_expertise_signal_returns_empty_dict_on_malformed_json():
    gateway = AsyncMock()
    gateway.chat.return_value = "not valid json"

    result = await extract_expertise_signal(gateway, "some query")

    assert result == {}


@pytest.mark.asyncio
async def test_record_persona_signal_writes_merged_persona():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({"expertise_level": "student"})
    personas = FakePersonasCollection()

    await record_persona_signal(personas, gateway, "user-1", "query text", categories=["rules"])

    stored = await get_persona(personas, "user-1")
    assert stored["expertise_level"] == "student"
    assert stored["category_affinity"]["rules"] == 1.0


@pytest.mark.asyncio
async def test_record_persona_signal_swallows_gateway_errors():
    gateway = AsyncMock()
    gateway.chat.side_effect = RuntimeError("gateway unreachable")
    personas = FakePersonasCollection()

    await record_persona_signal(personas, gateway, "user-1", "query text", categories=["rules"])

    assert await get_persona(personas, "user-1") is None
```

- [ ] **Step 2: Create `packages/retrieval-api/tests/tests_persona_fakes.py`**

A local copy of the fake, since `retrieval-api`'s tests don't import another package's `tests/`
directory (not part of its installed distribution):

```python
class FakePersonasCollection:
    def __init__(self):
        self.documents: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        return self.documents.get(filter.get("user_id"))

    async def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> None:
        self.documents[filter["user_id"]] = replacement
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/retrieval-api && uv run pytest tests/test_persona_signal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.ai_mode.persona_signal'`

- [ ] **Step 4: Write minimal implementation `packages/retrieval-api/src/retrieval_api/ai_mode/persona_signal.py`**

```python
import json
import logging

from persona.repository import record_signal

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Given a single legal-research query, classify the asker's likely
expertise and phrasing style. Respond with JSON only:
{"expertise_level": "student" | "practitioner" | "expert", "query_style": "broad" | "precise-citation"}
Omit a key if you are not reasonably confident about it."""

_RESPONSE_FORMAT = {"type": "json_object"}


async def extract_expertise_signal(gateway, query: str) -> dict:
    try:
        response = await gateway.chat(
            role="slm",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format=_RESPONSE_FORMAT,
        )
        result = json.loads(response)
    except Exception:
        logger.debug("expertise signal extraction failed for query %r", query, exc_info=True)
        return {}
    return result if isinstance(result, dict) else {}


async def record_persona_signal(personas, gateway, user_id: str, query: str, categories: list[str]) -> None:
    try:
        expertise_patch = await extract_expertise_signal(gateway, query)
        await record_signal(personas, user_id, categories, expertise_patch)
    except Exception:
        logger.warning("persona signal recording failed for user %r", user_id, exc_info=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/retrieval-api && uv run pytest tests/test_persona_signal.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/persona_signal.py packages/retrieval-api/tests/test_persona_signal.py packages/retrieval-api/tests/tests_persona_fakes.py
git commit -m "feat(ai-mode): add cheap expertise-signal extraction and persona write orchestration"
```

---

### Task 8: Wire persona read/write into `/ws/search`

**Files:**
- Modify: `pyproject.toml` (workspace members — confirm `packages/persona` present from Task 1)
- Modify: `packages/retrieval-api/pyproject.toml`
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py`
- Create: `packages/retrieval-api/tests/test_ws_persona_wiring.py`

**Interfaces:**
- Consumes: `auth.security.decode_access_token`, `auth.config.get_auth_settings` (from the auth
  plan); `persona.config.get_persona_settings`, `persona.db.get_mongo_client`,
  `persona.db.get_personas_collection`, `persona.repository.get_persona`,
  `persona.prompt.render_persona_context` (Tasks 1, 2, 4, 5); `record_persona_signal` (Task 7).

- [ ] **Step 1: Add `"persona"` and `"auth"` to `packages/retrieval-api/pyproject.toml` dependencies**

```toml
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "langchain-core>=0.3",
  "langfuse>=4.14",
  "common",
  "agents",
  "auth",
  "persona",
]
```

(`"auth"` may already be present if the auth plan's Task 7 ran first — do not duplicate the entry.)

- [ ] **Step 2: Confirm root `pyproject.toml` workspace registration**

```toml
[tool.uv.workspace]
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents", "packages/auth", "packages/persona"]

[tool.uv.sources]
common = { workspace = true }
model-gateway = { workspace = true }
retrieval-api = { workspace = true }
agents = { workspace = true }
auth = { workspace = true }
persona = { workspace = true }
```

Run `uv sync --all-packages` from repo root.

- [ ] **Step 3: Write the failing test `packages/retrieval-api/tests/test_ws_persona_wiring.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_persona_wiring.py -v`
Expected: FAIL only if `message["access_token"]` access raises — currently `ws.py` ignores
unknown fields in the JSON message (Python dicts don't error on extra keys), so this specific
test may already pass at the schema level; the meaningful failure is that no persona
loading/writing happens yet. Proceed to wire it regardless — this test is the regression guard
for Step 5 breaking the guest path.

- [ ] **Step 5: Modify `packages/retrieval-api/src/retrieval_api/ws.py`**

```python
import asyncio
import logging

from fastapi import APIRouter, WebSocket
from langfuse import get_client

from agents.pipeline import run_agentic_search
from auth.config import get_auth_settings
from auth.security import decode_access_token
from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from persona.config import get_persona_settings
from persona.db import get_mongo_client, get_personas_collection
from persona.prompt import render_persona_context
from persona.repository import get_persona
from retrieval_api.ai_mode.persona_signal import record_persona_signal
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.ai_mode.pipeline import run_ai_mode

router = APIRouter()

logger = logging.getLogger(__name__)


def get_gateway_client(settings) -> GatewayClient:
    return GatewayClient(base_url=settings.gateway_url)


def _resolve_user_id(access_token: str | None) -> str | None:
    if not access_token:
        return None
    return decode_access_token(access_token, get_auth_settings())


async def _emit_trace_step(send, step: str, data: dict) -> None:
    """Swallows any exception from `send` (e.g. the client disconnected
    mid-stream) - a dead trace channel must never fail the AI Mode
    pipeline or its final answer."""
    try:
        await send({"type": "ai_mode_trace", "step": step, "data": data})
    except Exception as exc:
        logger.debug("trace step %r dropped: %s", step, exc)


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"
    trace = message.get("trace", False)
    rerank = message.get("rerank", False)
    user_id = _resolve_user_id(message.get("access_token"))

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
        persona_settings = get_persona_settings()
        mongo_client = get_mongo_client(persona_settings)
        personas_collection = get_personas_collection(mongo_client, persona_settings)
        persona = await get_persona(personas_collection, user_id)
        persona_context = render_persona_context(persona)

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

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
                if mode in ("instant", "both") else None
            )
            ai_mode_task = (
                asyncio.create_task(
                    run_ai_mode(
                        gateway, es_client, milvus_client, query,
                        on_step=emit_trace_step if trace else None,
                        persona_context=persona_context,
                    )
                )
                if mode in ("ai_mode", "both") else None
            )

            output: dict = {}

            if instant_task is not None:
                instant_result = await instant_task
                output["instant_ok"] = instant_result["es_error"] is None and instant_result["milvus_error"] is None
                root_span.update(metadata={
                    "instant_es_error": instant_result["es_error"] or "",
                    "instant_milvus_error": instant_result["milvus_error"] or "",
                })
                await send({"type": "instant_result", **instant_result})

            if ai_mode_task is not None:
                ai_mode_result = await ai_mode_task
                output["ai_mode_ok"] = ai_mode_result.get("ok", False)
                await send({"type": "ai_mode_result", **ai_mode_result})

                if user_id is not None and ai_mode_result.get("ok"):
                    asyncio.create_task(
                        record_persona_signal(
                            personas_collection, gateway, user_id, query,
                            categories=ai_mode_result.get("intent", []),
                        )
                    )

            root_span.update(output=output)
    finally:
        await websocket.close()
```

This preserves everything below `ai_mode_task` handling and the rest of the file (the
`/ws/agent` route) unchanged — only the `/ws/search` route body and the new imports/helper
change. (If the actual current tail of `search()` differs slightly from what's shown above,
e.g. additional error handling around `root_span.update`, keep that existing structure and graft
in only: the `user_id`/`persona_context` resolution before the task-creation block, the
`persona_context=persona_context` kwarg on the `run_ai_mode` call, and the
`asyncio.create_task(record_persona_signal(...))` line right after `ai_mode_result` is sent.)

- [ ] **Step 6: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_persona_wiring.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite to confirm no regressions**

Run: `uv run pytest` from repo root
Expected: all previously-passing tests still pass, plus all new `persona` package tests and the
new `retrieval-api` tests from Tasks 6-8.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml packages/retrieval-api packages/persona
git commit -m "feat(ai-mode): wire persona read/write into /ws/search"
```

---

## Self-Review Notes

- **Spec coverage:** Persona storage (Mongo `personas` collection, flat evolving schema),
  extraction pipeline (category tally from `extract_intent()` output + one cheap async `slm`
  call, merge-never-overwrite), and read-injection into synthesis prompt only (never RRF/routing)
  are all covered by Tasks 1-8. The spec's explicit-deferral list (graph, vector memory,
  category-weighted routing) has no corresponding task, by design.
- **Placeholder scan:** No TBD/TODO markers; every step has runnable code. Task 8's Step 5 has a
  fallback note for exact-line drift in `ws.py` since that file may have shifted slightly by the
  time this task executes — this is a graft instruction, not a placeholder, and names the exact
  three changes required regardless of surrounding line numbers.
- **Type consistency:** `user_id: str | None` end to end (matches `auth.dependency`'s return
  type from the auth plan). `categories: list[str]` matches `intent_result["intent"]`'s type in
  `ai_mode/intent.py`. `persona: dict | None` is consistent across `repository.get_persona`,
  `prompt.render_persona_context`, and `merge.py`'s functions (which take the persona's
  sub-fields, not the whole document, avoiding a signature mismatch).
