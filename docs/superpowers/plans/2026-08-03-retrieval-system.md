# Retrieval System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two-loop ("Loop 1" instant preview + "Loop 2" AI Mode: SLM rewrite → RRF → rerank → LLM synthesis) legal-caselaw retrieval service described in `docs/superpowers/specs/2026-08-03-retrieval-system-design.md`, as a standalone uv-workspace repo with a `model-gateway` provider-abstraction service and a `retrieval-api` orchestration service.

**Architecture:** Two FastAPI services in docker-compose. `model-gateway` exposes `/v1/chat`, `/v1/embed`, `/v1/rerank`, each keyed by a `role` resolved to a DeepInfra model via config — the only seam that knows about providers. `retrieval-api` implements Loop 1 (parallel raw ES+Milvus fetch) and Loop 2 (SLM→filter→Milvus→RRF→rerank/citations→synthesis) over one WebSocket per query, calling `model-gateway` for every model call.

**Tech Stack:** Python 3.11, uv workspace, FastAPI, pydantic-settings, httpx, pymilvus, elasticsearch-py, LangChain (chain/prompt orchestration in `retrieval-api`), pytest, pytest-asyncio, respx (HTTP mocking).

## Global Constraints

- Python 3.11 (not 3.14 — `pymilvus`'s `grpcio` has no prebuilt wheel for 3.14, per data-extraction-pipeline's documented gotcha).
- Milvus: dedicated `aic` database, never touch `default`. Connect via `MILVUS_URI`/`MILVUS_TOKEN`/`MILVUS_DB` env vars.
- Milvus collections and chunking (verified against code, see spec): `case_summary`/`headnotes`/`metadata` are single-row (no `chunk_part`); `digest`/`facts`/`held`/`ruling` are chunked (`chunk_part`/`total_chunks` present, `CHUNK_SIZE_TOKENS=1024`).
- `sparse_vector` is server-computed BM25 in Milvus — never send a client-side value for it.
- No ranking fusion between ES and Milvus — `doc_id` is join-only.
- AI Mode searches all 7 Milvus collections every query — no intent-based collection routing.
- v1 queries `dense_vector` (Voyage-equivalent slot, populated via DeepInfra through model-gateway) only — `dense_vector_2` parity is explicitly deferred.
- Deterministic pipeline with an intent-routing layer — not agentic.
- Every LLM/embedding/rerank call in `retrieval-api` goes through `model-gateway` — no direct provider SDK usage in `retrieval-api`.

---

## File Structure

```
retrieval-system/
  pyproject.toml                          # uv workspace root
  docker-compose.yml
  .env.example
  .gitignore
  packages/
    common/
      pyproject.toml
      src/common/
        __init__.py
        config.py                         # Settings (pydantic-settings)
        schemas.py                        # collection/field constants
        milvus_client.py                  # pymilvus wrapper, hybrid search
        es_client.py                      # ES wrapper, raw + filter + citation search
      tests/
        test_config.py
        test_schemas.py
        test_milvus_client.py
        test_es_client.py
    model-gateway/
      pyproject.toml
      src/model_gateway/
        __init__.py
        config.py                         # ROLE_MODEL_MAP
        adapters/
          __init__.py
          base.py                         # ModelAdapter Protocol
          deepinfra.py                    # DeepInfraAdapter
        routes.py                         # /v1/chat /v1/embed /v1/rerank
        main.py
      tests/
        test_deepinfra_adapter.py
        test_routes.py
    retrieval-api/
      pyproject.toml
      src/retrieval_api/
        __init__.py
        gateway_client.py                 # httpx client -> model-gateway
        loop1/
          __init__.py
          search.py                       # parallel ES + Milvus raw fetch
        loop2/
          __init__.py
          intent.py                       # SLM rewrite/intent/filters
          filter_resolve.py               # ES filter -> doc_id allowlist
          retrieve.py                     # rewritten-query Milvus fetch + RRF merge
          rerank.py                       # gateway rerank call
          citations.py                    # ES citation prefetch + fallback lookup
          synthesize.py                   # gateway chat (synthesis), streamed
          pipeline.py                     # wires 1-6 together, yields events
        ws.py                             # /ws/search websocket endpoint
        main.py
      tests/
        test_gateway_client.py
        test_loop1_search.py
        test_loop2_intent.py
        test_loop2_filter_resolve.py
        test_loop2_retrieve.py
        test_loop2_rerank_citations.py
        test_loop2_synthesize.py
        test_loop2_pipeline.py
        test_ws_integration.py
```

---

### Task 1: Workspace scaffold

**Files:**
- Create: `pyproject.toml` (repo root)
- Create: `.env.example`
- Create: `.gitignore`
- Create: `packages/common/pyproject.toml`
- Create: `packages/model-gateway/pyproject.toml`
- Create: `packages/retrieval-api/pyproject.toml`

**Interfaces:**
- Produces: a `uv sync`-able workspace with three empty-but-importable packages (`common`, `model_gateway`, `retrieval_api`).

- [ ] **Step 1: Write root `pyproject.toml`**

```toml
[project]
name = "retrieval-system"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
common = { workspace = true }
model-gateway = { workspace = true }
retrieval-api = { workspace = true }

[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "respx>=0.21"]
```

- [ ] **Step 2: Write `packages/common/pyproject.toml`**

```toml
[project]
name = "common"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "pydantic-settings>=2.5",
  "pymilvus>=2.4",
  "elasticsearch>=8.15,<9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/common"]
```

- [ ] **Step 3: Write `packages/model-gateway/pyproject.toml`**

```toml
[project]
name = "model-gateway"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "pydantic-settings>=2.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/model_gateway"]
```

- [ ] **Step 4: Write `packages/retrieval-api/pyproject.toml`**

```toml
[project]
name = "retrieval-api"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "langchain-core>=0.3",
  "common",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/retrieval_api"]
```

- [ ] **Step 5: Create empty package `__init__.py` files**

```bash
mkdir -p packages/common/src/common packages/common/tests
mkdir -p packages/model-gateway/src/model_gateway/adapters packages/model-gateway/tests
mkdir -p packages/retrieval-api/src/retrieval_api/loop1 packages/retrieval-api/src/retrieval_api/loop2 packages/retrieval-api/tests
touch packages/common/src/common/__init__.py
touch packages/model-gateway/src/model_gateway/__init__.py packages/model-gateway/src/model_gateway/adapters/__init__.py
touch packages/retrieval-api/src/retrieval_api/__init__.py packages/retrieval-api/src/retrieval_api/loop1/__init__.py packages/retrieval-api/src/retrieval_api/loop2/__init__.py
```

- [ ] **Step 6: Write `.env.example`**

```bash
# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
MILVUS_DB=aic

# Elasticsearch
ES_URL=http://localhost:9200

# model-gateway
GATEWAY_URL=http://model-gateway:8001
DEEPINFRA_API_KEY=
DEEPINFRA_CHAT_MODEL_SLM=meta-llama/Meta-Llama-3.1-8B-Instruct
DEEPINFRA_CHAT_MODEL_SYNTHESIS=meta-llama/Meta-Llama-3.1-70B-Instruct
DEEPINFRA_EMBED_MODEL=BAAI/bge-large-en-v1.5
DEEPINFRA_RERANK_MODEL=BAAI/bge-reranker-large
```

- [ ] **Step 7: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
uv.lock
.pytest_cache/
```

- [ ] **Step 8: Verify workspace resolves**

Run: `cd /Users/uditya/dev/taxmann/retrieval-system && uv sync`
Expected: succeeds, creates `.venv` and `uv.lock`, no dependency errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .env.example .gitignore packages
git commit -m "chore: scaffold uv workspace with 3 packages"
```

---

### Task 2: `common` — config

**Files:**
- Create: `packages/common/src/common/config.py`
- Test: `packages/common/tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields `milvus_uri: str`, `milvus_token: str`, `milvus_db: str = "aic"`, `es_url: str`, `gateway_url: str`; `get_settings() -> Settings` (lru-cached factory).

- [ ] **Step 1: Write the failing test**

```python
# packages/common/tests/test_config.py
import os
from common.config import Settings, get_settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "root:Milvus")
    monkeypatch.setenv("ES_URL", "http://es:9200")
    monkeypatch.setenv("GATEWAY_URL", "http://model-gateway:8001")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.milvus_uri == "http://milvus:19530"
    assert settings.milvus_db == "aic"  # default
    assert settings.es_url == "http://es:9200"
    assert settings.gateway_url == "http://model-gateway:8001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/common && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/common/src/common/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str
    milvus_token: str
    milvus_db: str = "aic"
    es_url: str
    gateway_url: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/common && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/config.py packages/common/tests/test_config.py
git commit -m "feat(common): add Settings config"
```

---

### Task 3: `common` — schema constants

**Files:**
- Create: `packages/common/src/common/schemas.py`
- Test: `packages/common/tests/test_schemas.py`

**Interfaces:**
- Produces: `MILVUS_COLLECTIONS: list[str]` (7 names), `CHUNKED_COLLECTIONS: set[str]` (`digest`, `facts`, `held`, `ruling`), `BM25_SOURCE_FIELD: dict[str, str]` (collection -> source field name, `"text"` for the 6 chunked/section collections, `"heading_subheading_text"` for `metadata`), `MASTERINFO_CITATION_FIELDS: list[str]` (`["masterinfo.citations", "masterinfo.court", "masterinfo.bench", "masterinfo.judge", "masterinfo.partyname"]`).

- [ ] **Step 1: Write the failing test**

```python
# packages/common/tests/test_schemas.py
from common.schemas import (
    MILVUS_COLLECTIONS,
    CHUNKED_COLLECTIONS,
    BM25_SOURCE_FIELD,
    MASTERINFO_CITATION_FIELDS,
)


def test_seven_collections():
    assert set(MILVUS_COLLECTIONS) == {
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    }


def test_chunked_collections_match_verified_code_behavior():
    assert CHUNKED_COLLECTIONS == {"digest", "facts", "held", "ruling"}
    assert "case_summary" not in CHUNKED_COLLECTIONS
    assert "headnotes" not in CHUNKED_COLLECTIONS
    assert "metadata" not in CHUNKED_COLLECTIONS


def test_bm25_source_field_metadata_uses_heading_subheading():
    assert BM25_SOURCE_FIELD["metadata"] == "heading_subheading_text"
    assert BM25_SOURCE_FIELD["ruling"] == "text"


def test_masterinfo_citation_fields():
    assert MASTERINFO_CITATION_FIELDS == [
        "masterinfo.citations", "masterinfo.court", "masterinfo.bench",
        "masterinfo.judge", "masterinfo.partyname",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/common && uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.schemas'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/common/src/common/schemas.py
MILVUS_COLLECTIONS = [
    "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
]

CHUNKED_COLLECTIONS = {"digest", "facts", "held", "ruling"}

BM25_SOURCE_FIELD = {name: "text" for name in MILVUS_COLLECTIONS}
BM25_SOURCE_FIELD["metadata"] = "heading_subheading_text"

MASTERINFO_CITATION_FIELDS = [
    "masterinfo.citations",
    "masterinfo.court",
    "masterinfo.bench",
    "masterinfo.judge",
    "masterinfo.partyname",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/common && uv run pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/schemas.py packages/common/tests/test_schemas.py
git commit -m "feat(common): add Milvus/ES schema constants"
```

---

### Task 4: `common` — Milvus client wrapper

**Files:**
- Create: `packages/common/src/common/milvus_client.py`
- Test: `packages/common/tests/test_milvus_client.py`

**Interfaces:**
- Consumes: `common.config.Settings`, `common.schemas.MILVUS_COLLECTIONS`.
- Produces:
  - `get_milvus_client(settings: Settings) -> pymilvus.MilvusClient`
  - `async def hybrid_search(client, collections: list[str], dense_vector: list[float] | None, sparse_query_text: str, doc_id_allowlist: list[str] | None = None, limit: int = 50) -> dict[str, list[dict]]` — returns `{collection_name: [{"chunk_id": str, "doc_id": str, "text": str, "score": float, **extra_fields}, ...]}`. Runs one search per collection concurrently via `asyncio.gather` (pymilvus calls wrapped in `asyncio.to_thread` since the client is sync).

- [ ] **Step 1: Write the failing test**

```python
# packages/common/tests/test_milvus_client.py
import asyncio
from unittest.mock import MagicMock
import pytest
from common.milvus_client import hybrid_search


class FakeMilvusClient:
    def __init__(self, hits_by_collection):
        self.hits_by_collection = hits_by_collection
        self.calls = []

    def search(self, collection_name, data, anns_field, limit, filter=None, output_fields=None, **kwargs):
        self.calls.append((collection_name, anns_field, filter))
        return [self.hits_by_collection.get(collection_name, [])]


def _hit(chunk_id, doc_id, text, score):
    return {"id": chunk_id, "distance": score, "entity": {"doc_id": doc_id, "text": text}}


@pytest.mark.asyncio
async def test_hybrid_search_runs_all_collections_concurrently_and_shapes_rows():
    client = FakeMilvusClient({
        "ruling": [_hit("d1::ruling::0", "d1", "ruling text", 0.9)],
        "facts": [_hit("d2::facts::0", "d2", "facts text", 0.8)],
    })

    result = await hybrid_search(
        client, collections=["ruling", "facts"],
        dense_vector=[0.1, 0.2], sparse_query_text="income tax",
    )

    assert result["ruling"] == [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "ruling text", "score": 0.9}]
    assert result["facts"] == [{"chunk_id": "d2::facts::0", "doc_id": "d2", "text": "facts text", "score": 0.8}]
    assert {c for c, _, _ in client.calls} == {"ruling", "facts"}


@pytest.mark.asyncio
async def test_hybrid_search_applies_doc_id_allowlist_filter():
    client = FakeMilvusClient({"ruling": []})

    await hybrid_search(
        client, collections=["ruling"],
        dense_vector=[0.1], sparse_query_text="q",
        doc_id_allowlist=["d1", "d2"],
    )

    _, _, filter_expr = client.calls[0]
    assert filter_expr == 'doc_id in ["d1", "d2"]'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/common && uv run pytest tests/test_milvus_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.milvus_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/common/src/common/milvus_client.py
import asyncio
import json

from pymilvus import MilvusClient

from common.config import Settings


def get_milvus_client(settings: Settings) -> MilvusClient:
    return MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token, db_name=settings.milvus_db)


def _doc_id_filter(doc_id_allowlist: list[str] | None) -> str | None:
    if not doc_id_allowlist:
        return None
    quoted = ", ".join(f'"{d}"' for d in doc_id_allowlist)
    return f"doc_id in [{quoted}]"


def _search_one(client, collection: str, dense_vector, limit: int, filter_expr: str | None) -> list[dict]:
    hits = client.search(
        collection_name=collection,
        data=[dense_vector],
        anns_field="dense_vector",
        limit=limit,
        filter=filter_expr,
        output_fields=["doc_id", "text"],
    )[0]
    return [
        {
            "chunk_id": h["id"],
            "doc_id": h["entity"]["doc_id"],
            "text": h["entity"]["text"],
            "score": h["distance"],
        }
        for h in hits
    ]


async def hybrid_search(
    client,
    collections: list[str],
    dense_vector: list[float] | None,
    sparse_query_text: str,
    doc_id_allowlist: list[str] | None = None,
    limit: int = 50,
) -> dict[str, list[dict]]:
    filter_expr = _doc_id_filter(doc_id_allowlist)
    results = await asyncio.gather(*[
        asyncio.to_thread(_search_one, client, collection, dense_vector, limit, filter_expr)
        for collection in collections
    ])
    return dict(zip(collections, results))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/common && uv run pytest tests/test_milvus_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/milvus_client.py packages/common/tests/test_milvus_client.py
git commit -m "feat(common): add Milvus hybrid_search wrapper"
```

---

### Task 5: `common` — Elasticsearch client wrapper

**Files:**
- Create: `packages/common/src/common/es_client.py`
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: `common.config.Settings`, `common.schemas.MASTERINFO_CITATION_FIELDS`.
- Produces:
  - `get_es_client(settings: Settings) -> elasticsearch.AsyncElasticsearch`
  - `async def raw_search(client, query: str, limit: int = 20) -> list[dict]` — BM25 multi-match over `facts_text`/`held_text`/`headnotes_text`/`judgment_text`/`case_review_text`, returns `[{"doc_id": str, "score": float, "snippet": str}]`.
  - `async def resolve_doc_id_allowlist(client, filters: dict) -> list[str] | None` — queries `masterinfo.*` term/range filters (court, act, date range), returns matching `doc_id`s or `None` if `filters` is empty.
  - `async def fetch_citations(client, doc_ids: list[str]) -> dict[str, dict]` — `doc_id -> {field: value}` for `MASTERINFO_CITATION_FIELDS`, via `mget`.

- [ ] **Step 1: Write the failing test**

```python
# packages/common/tests/test_es_client.py
import pytest
from common.es_client import raw_search, resolve_doc_id_allowlist, fetch_citations


class FakeAsyncES:
    def __init__(self, search_hits=None, mget_docs=None):
        self.search_hits = search_hits or []
        self.mget_docs = mget_docs or {}
        self.search_calls = []
        self.mget_calls = []

    async def search(self, index, query, size):
        self.search_calls.append(query)
        return {"hits": {"hits": self.search_hits}}

    async def mget(self, index, ids):
        self.mget_calls.append(ids)
        return {"docs": [{"_id": i, "found": True, "_source": self.mget_docs.get(i, {})} for i in ids]}


@pytest.mark.asyncio
async def test_raw_search_returns_doc_id_score_snippet():
    client = FakeAsyncES(search_hits=[
        {"_source": {"doc_id": "d1", "facts_text": "assessee claimed exemption"}, "_score": 4.2},
    ])

    results = await raw_search(client, "exemption claim", limit=20)

    assert results == [{"doc_id": "d1", "score": 4.2, "snippet": "assessee claimed exemption"}]


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_returns_none_when_no_filters():
    client = FakeAsyncES()
    assert await resolve_doc_id_allowlist(client, {}) is None


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_and_returns_doc_ids():
    client = FakeAsyncES(search_hits=[{"_source": {"doc_id": "d1"}}, {"_source": {"doc_id": "d2"}}])

    result = await resolve_doc_id_allowlist(client, {"court": "Supreme Court"})

    assert result == ["d1", "d2"]
    assert client.search_calls  # a query was actually issued


@pytest.mark.asyncio
async def test_fetch_citations_returns_doc_id_keyed_masterinfo_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {"masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]}},
    })

    result = await fetch_citations(client, ["d1"])

    assert result == {"d1": {"masterinfo": {"court": "Supreme Court", "citations": ["2020 SCC 1"]}}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/common && uv run pytest tests/test_es_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'common.es_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/common/src/common/es_client.py
from elasticsearch import AsyncElasticsearch

from common.config import Settings

_RAW_SEARCH_FIELDS = [
    "facts_text", "held_text", "headnotes_text", "judgment_text", "case_review_text",
]

_INDEX = "taxmann_caselaw"


def get_es_client(settings: Settings) -> AsyncElasticsearch:
    return AsyncElasticsearch(settings.es_url)


async def raw_search(client, query: str, limit: int = 20) -> list[dict]:
    body = {"multi_match": {"query": query, "fields": _RAW_SEARCH_FIELDS, "fuzziness": "AUTO"}}
    response = await client.search(index=_INDEX, query=body, size=limit)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        snippet = next((source[f] for f in _RAW_SEARCH_FIELDS if source.get(f)), "")
        results.append({"doc_id": source["doc_id"], "score": hit["_score"], "snippet": snippet})
    return results


async def resolve_doc_id_allowlist(client, filters: dict) -> list[str] | None:
    if not filters:
        return None
    must = []
    if "court" in filters:
        must.append({"term": {"masterinfo.court": filters["court"]}})
    if "act" in filters:
        must.append({"term": {"masterinfo.act": filters["act"]}})
    if "date_range" in filters:
        must.append({"range": {"masterinfo.date": filters["date_range"]}})
    response = await client.search(index=_INDEX, query={"bool": {"must": must}}, size=1000)
    return [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]


async def fetch_citations(client, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    response = await client.mget(index=_INDEX, ids=doc_ids)
    return {
        doc["_id"]: doc["_source"]
        for doc in response["docs"]
        if doc.get("found")
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/common && uv run pytest tests/test_es_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(common): add Elasticsearch raw search, filter resolve, citation fetch"
```

---

### Task 6: `model-gateway` — adapter Protocol + DeepInfra adapter

**Files:**
- Create: `packages/model-gateway/src/model_gateway/adapters/base.py`
- Create: `packages/model-gateway/src/model_gateway/adapters/deepinfra.py`
- Test: `packages/model-gateway/tests/test_deepinfra_adapter.py`

**Interfaces:**
- Produces:
  - `ModelAdapter` Protocol: `async def chat(self, model: str, messages: list[dict]) -> str`, `async def embed(self, model: str, text: str) -> list[float]`, `async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]` (relevance scores, same order as `documents`).
  - `DeepInfraAdapter(api_key: str)` implementing the above via `httpx.AsyncClient` against `https://api.deepinfra.com/v1/openai/chat/completions`, `.../embeddings`, and `https://api.deepinfra.com/v1/inference/{model}` for rerank.

- [ ] **Step 1: Write the failing test**

```python
# packages/model-gateway/tests/test_deepinfra_adapter.py
import httpx
import pytest
import respx

from model_gateway.adapters.deepinfra import DeepInfraAdapter


@pytest.mark.asyncio
@respx.mock
async def test_chat_posts_openai_shape_and_returns_content():
    respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    assert result == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_embed_returns_vector():
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.embed("embed-model", "some text")

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@respx.mock
async def test_rerank_returns_scores_in_input_order():
    respx.post("https://api.deepinfra.com/v1/inference/rerank-model").mock(
        return_value=httpx.Response(200, json={"scores": [0.9, 0.2]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    result = await adapter.rerank("rerank-model", "query", ["doc a", "doc b"])

    assert result == [0.9, 0.2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/model-gateway && uv run pytest tests/test_deepinfra_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_gateway.adapters.deepinfra'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/model-gateway/src/model_gateway/adapters/base.py
from typing import Protocol


class ModelAdapter(Protocol):
    async def chat(self, model: str, messages: list[dict]) -> str: ...
    async def embed(self, model: str, text: str) -> list[float]: ...
    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]: ...
```

```python
# packages/model-gateway/src/model_gateway/adapters/deepinfra.py
import httpx

_BASE_URL = "https://api.deepinfra.com/v1"


class DeepInfraAdapter:
    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def chat(self, model: str, messages: list[dict]) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/openai/chat/completions",
                json={"model": model, "messages": messages},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def embed(self, model: str, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/openai/embeddings",
                json={"model": model, "input": text},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_BASE_URL}/inference/{model}",
                json={"query": query, "documents": documents},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()["scores"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/model-gateway && uv run pytest tests/test_deepinfra_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/model-gateway/src/model_gateway/adapters packages/model-gateway/tests/test_deepinfra_adapter.py
git commit -m "feat(model-gateway): add ModelAdapter protocol and DeepInfra adapter"
```

---

### Task 7: `model-gateway` — role config + routes

**Files:**
- Create: `packages/model-gateway/src/model_gateway/config.py`
- Create: `packages/model-gateway/src/model_gateway/routes.py`
- Create: `packages/model-gateway/src/model_gateway/main.py`
- Test: `packages/model-gateway/tests/test_routes.py`

**Interfaces:**
- Consumes: `model_gateway.adapters.deepinfra.DeepInfraAdapter`.
- Produces:
  - `GatewaySettings` (pydantic-settings): `deepinfra_api_key: str`, `chat_model_slm: str`, `chat_model_synthesis: str`, `embed_model: str`, `rerank_model: str`.
  - `ROLE_MODEL_MAP: dict[str, str]` built from `GatewaySettings` (`"slm"`, `"synthesis"` -> chat models; `"query_embed"` -> embed model; `"reranker"` -> rerank model).
  - FastAPI app (`main.app`) with routes `POST /v1/chat {"role": str, "messages": list[dict]} -> {"content": str}`, `POST /v1/embed {"role": str, "text": str} -> {"embedding": list[float]}`, `POST /v1/rerank {"role": str, "query": str, "documents": list[str]} -> {"scores": list[float]}`. Unknown `role` -> 400.

- [ ] **Step 1: Write the failing test**

```python
# packages/model-gateway/tests/test_routes.py
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from model_gateway.main import app
import model_gateway.routes as routes_module


def test_chat_route_resolves_role_and_calls_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = "the answer"
    monkeypatch.setattr(routes_module, "get_adapter", lambda: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.json() == {"content": "the answer"}
    fake_adapter.chat.assert_awaited_once_with("big-model", [{"role": "user", "content": "hi"}])


def test_chat_route_rejects_unknown_role(monkeypatch):
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    client = TestClient(app)

    response = client.post("/v1/chat", json={"role": "nonexistent", "messages": []})

    assert response.status_code == 400


def test_embed_route(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.embed.return_value = [0.1, 0.2]
    monkeypatch.setattr(routes_module, "get_adapter", lambda: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"query_embed": "embed-model"})

    client = TestClient(app)
    response = client.post("/v1/embed", json={"role": "query_embed", "text": "hello"})

    assert response.json() == {"embedding": [0.1, 0.2]}


def test_rerank_route(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    monkeypatch.setattr(routes_module, "get_adapter", lambda: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"reranker": "rerank-model"})

    client = TestClient(app)
    response = client.post("/v1/rerank", json={"role": "reranker", "query": "q", "documents": ["a", "b"]})

    assert response.json() == {"scores": [0.9, 0.1]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/model-gateway && uv run pytest tests/test_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_gateway.main'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/model-gateway/src/model_gateway/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepinfra_api_key: str
    chat_model_slm: str
    chat_model_synthesis: str
    embed_model: str
    rerank_model: str


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


def build_role_model_map(settings: GatewaySettings) -> dict[str, str]:
    return {
        "slm": settings.chat_model_slm,
        "synthesis": settings.chat_model_synthesis,
        "query_embed": settings.embed_model,
        "reranker": settings.rerank_model,
    }
```

```python
# packages/model-gateway/src/model_gateway/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model_gateway.adapters.deepinfra import DeepInfraAdapter
from model_gateway.config import build_role_model_map, get_gateway_settings

router = APIRouter()

ROLE_MODEL_MAP: dict[str, str] = build_role_model_map(get_gateway_settings())


def get_adapter():
    return DeepInfraAdapter(api_key=get_gateway_settings().deepinfra_api_key)


def _resolve_model(role: str) -> str:
    if role not in ROLE_MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"unknown role: {role}")
    return ROLE_MODEL_MAP[role]


class ChatRequest(BaseModel):
    role: str
    messages: list[dict]


class EmbedRequest(BaseModel):
    role: str
    text: str


class RerankRequest(BaseModel):
    role: str
    query: str
    documents: list[str]


@router.post("/v1/chat")
async def chat(req: ChatRequest):
    model = _resolve_model(req.role)
    content = await get_adapter().chat(model, req.messages)
    return {"content": content}


@router.post("/v1/embed")
async def embed(req: EmbedRequest):
    model = _resolve_model(req.role)
    embedding = await get_adapter().embed(model, req.text)
    return {"embedding": embedding}


@router.post("/v1/rerank")
async def rerank(req: RerankRequest):
    model = _resolve_model(req.role)
    scores = await get_adapter().rerank(model, req.query, req.documents)
    return {"scores": scores}
```

```python
# packages/model-gateway/src/model_gateway/main.py
from fastapi import FastAPI

from model_gateway.routes import router

app = FastAPI(title="model-gateway")
app.include_router(router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/model-gateway && uv run pytest tests/test_routes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/model-gateway/src/model_gateway/config.py packages/model-gateway/src/model_gateway/routes.py packages/model-gateway/src/model_gateway/main.py packages/model-gateway/tests/test_routes.py
git commit -m "feat(model-gateway): add role-based /v1/chat /v1/embed /v1/rerank routes"
```

---

### Task 8: `retrieval-api` — gateway client

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/gateway_client.py`
- Test: `packages/retrieval-api/tests/test_gateway_client.py`

**Interfaces:**
- Produces: `GatewayClient(base_url: str)` with `async def chat(self, role: str, messages: list[dict]) -> str`, `async def embed(self, role: str, text: str) -> list[float]`, `async def rerank(self, role: str, query: str, documents: list[str]) -> list[float]` — each POSTs to `model-gateway`'s matching route and returns the unwrapped field.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_gateway_client.py
import httpx
import pytest
import respx

from retrieval_api.gateway_client import GatewayClient


@pytest.mark.asyncio
@respx.mock
async def test_chat_calls_gateway_and_unwraps_content():
    respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    assert result == "hi there"


@pytest.mark.asyncio
@respx.mock
async def test_embed_unwraps_embedding():
    respx.post("http://gateway/v1/embed").mock(
        return_value=httpx.Response(200, json={"embedding": [1.0, 2.0]})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.embed(role="query_embed", text="hello")

    assert result == [1.0, 2.0]


@pytest.mark.asyncio
@respx.mock
async def test_rerank_unwraps_scores():
    respx.post("http://gateway/v1/rerank").mock(
        return_value=httpx.Response(200, json={"scores": [0.5]})
    )
    client = GatewayClient(base_url="http://gateway")

    result = await client.rerank(role="reranker", query="q", documents=["a"])

    assert result == [0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_gateway_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.gateway_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/gateway_client.py
import httpx


class GatewayClient:
    def __init__(self, base_url: str):
        self._base_url = base_url

    async def chat(self, role: str, messages: list[dict]) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self._base_url}/v1/chat", json={"role": role, "messages": messages})
            response.raise_for_status()
            return response.json()["content"]

    async def embed(self, role: str, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self._base_url}/v1/embed", json={"role": role, "text": text})
            response.raise_for_status()
            return response.json()["embedding"]

    async def rerank(self, role: str, query: str, documents: list[str]) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/rerank", json={"role": role, "query": query, "documents": documents}
            )
            response.raise_for_status()
            return response.json()["scores"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_gateway_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/gateway_client.py packages/retrieval-api/tests/test_gateway_client.py
git commit -m "feat(retrieval-api): add GatewayClient httpx wrapper"
```

---

### Task 9: `retrieval-api` — Loop 1 parallel search

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop1/search.py`
- Test: `packages/retrieval-api/tests/test_loop1_search.py`

**Interfaces:**
- Consumes: `common.es_client.raw_search`, `common.milvus_client.hybrid_search`, `common.schemas.MILVUS_COLLECTIONS`.
- Produces: `async def run_loop1(es_client, milvus_client, query: str) -> dict` — returns `{"es": list[dict] | None, "es_error": str | None, "milvus": dict[str, list[dict]] | None, "milvus_error": str | None}`. Each branch's exception is caught independently; a failing branch never prevents the other's result from being returned. Loop 1 does not call `model-gateway` — no query embedding, so `hybrid_search` is invoked with `dense_vector=None` and callers must treat that as "dense skipped, sparse-only" (documented here; Milvus wrapper already supports sparse-only via its own `anns_field`/BM25 function server-side — this task only needs `dense_vector` to be an accepted `None` passthrough).

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop1_search.py
import pytest
from retrieval_api.loop1.search import run_loop1


@pytest.mark.asyncio
async def test_run_loop1_returns_both_branches_on_success(monkeypatch):
    import retrieval_api.loop1.search as search_module

    async def fake_raw_search(client, query, limit=20):
        return [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    result = await run_loop1(es_client=object(), milvus_client=object(), query="tax exemption")

    assert result["es"] == [{"doc_id": "d1", "score": 4.2, "snippet": "text"}]
    assert result["es_error"] is None
    assert result["milvus"] == {"ruling": [{"chunk_id": "d1::ruling::0", "doc_id": "d1", "text": "t", "score": 0.9}]}
    assert result["milvus_error"] is None


@pytest.mark.asyncio
async def test_run_loop1_returns_partial_result_when_es_fails(monkeypatch):
    import retrieval_api.loop1.search as search_module

    async def failing_raw_search(client, query, limit=20):
        raise RuntimeError("ES down")

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", failing_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    result = await run_loop1(es_client=object(), milvus_client=object(), query="q")

    assert result["es"] is None
    assert result["es_error"] == "ES down"
    assert result["milvus"] == {"ruling": []}
    assert result["milvus_error"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop1_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop1.search'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop1/search.py
import asyncio

from common.es_client import raw_search
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS


async def _run_es(es_client, query: str) -> tuple[list[dict] | None, str | None]:
    try:
        return await raw_search(es_client, query), None
    except Exception as exc:  # noqa: BLE001 - branch isolation is the point
        return None, str(exc)


async def _run_milvus(milvus_client, query: str) -> tuple[dict | None, str | None]:
    try:
        result = await hybrid_search(
            milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None, sparse_query_text=query,
        )
        return result, None
    except Exception as exc:  # noqa: BLE001 - branch isolation is the point
        return None, str(exc)


async def run_loop1(es_client, milvus_client, query: str) -> dict:
    (es_result, es_error), (milvus_result, milvus_error) = await asyncio.gather(
        _run_es(es_client, query),
        _run_milvus(milvus_client, query),
    )
    return {
        "es": es_result,
        "es_error": es_error,
        "milvus": milvus_result,
        "milvus_error": milvus_error,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop1_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop1/search.py packages/retrieval-api/tests/test_loop1_search.py
git commit -m "feat(retrieval-api): add Loop 1 parallel ES+Milvus raw search with branch isolation"
```

---

### Task 10: `retrieval-api` — Loop 2 intent extraction

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/intent.py`
- Test: `packages/retrieval-api/tests/test_loop2_intent.py`

**Interfaces:**
- Consumes: `retrieval_api.gateway_client.GatewayClient.chat`.
- Produces: `async def extract_intent(gateway: GatewayClient, query: str) -> dict` — calls `gateway.chat(role="slm", messages=[...])` with a prompt instructing JSON output `{"rewritten_query": str, "intent": str, "filters": dict}`, parses the JSON response, returns that dict. Raises `ValueError` on unparseable output (caller in Task 15 turns this into a `loop2_error` event).

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_intent.py
import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.loop2.intent import extract_intent


@pytest.mark.asyncio
async def test_extract_intent_parses_json_response():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "section_lookup",
        "filters": {"act": "BNS"},
    })

    result = await extract_intent(gateway, "IPC 302 punishment")

    assert result == {
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "section_lookup",
        "filters": {"act": "BNS"},
    }
    gateway.chat.assert_awaited_once()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_raises_on_unparseable_response():
    gateway = AsyncMock()
    gateway.chat.return_value = "not json"

    with pytest.raises(ValueError):
        await extract_intent(gateway, "some query")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_intent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.intent'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/intent.py
import json

from retrieval_api.gateway_client import GatewayClient

_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": the query rewritten for search, expanding any old-law
  references to their new-law equivalent (IPC -> BNS, CrPC -> BNSS, Evidence
  Act -> BSA) where applicable.
- "intent": one short intent category label.
- "filters": an object with any of "court", "act", "date_range", "party"
  the query explicitly mentions; omit keys that aren't mentioned.
"""


async def extract_intent(gateway: GatewayClient, query: str) -> dict:
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SLM did not return valid JSON: {response!r}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_intent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/intent.py packages/retrieval-api/tests/test_loop2_intent.py
git commit -m "feat(retrieval-api): add Loop 2 SLM intent/rewrite/filter extraction"
```

---

### Task 11: `retrieval-api` — Loop 2 ES filter resolution

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/filter_resolve.py`
- Test: `packages/retrieval-api/tests/test_loop2_filter_resolve.py`

**Interfaces:**
- Consumes: `common.es_client.resolve_doc_id_allowlist`.
- Produces: `async def resolve_allowlist(es_client, filters: dict) -> list[str] | None` — thin passthrough to `common.es_client.resolve_doc_id_allowlist`, kept as its own module per the design's step numbering (Loop 2 step 2) so `pipeline.py` (Task 15) reads as a 1:1 mirror of the spec's numbered steps.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_filter_resolve.py
import pytest
from retrieval_api.loop2.filter_resolve import resolve_allowlist


@pytest.mark.asyncio
async def test_resolve_allowlist_returns_none_for_empty_filters():
    assert await resolve_allowlist(es_client=object(), filters={}) is None


@pytest.mark.asyncio
async def test_resolve_allowlist_delegates_to_common_es_client(monkeypatch):
    import retrieval_api.loop2.filter_resolve as module

    async def fake_resolve(client, filters):
        assert filters == {"court": "Supreme Court"}
        return ["d1", "d2"]

    monkeypatch.setattr(module, "resolve_doc_id_allowlist", fake_resolve)

    result = await resolve_allowlist(es_client=object(), filters={"court": "Supreme Court"})

    assert result == ["d1", "d2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_filter_resolve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.filter_resolve'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/filter_resolve.py
from common.es_client import resolve_doc_id_allowlist


async def resolve_allowlist(es_client, filters: dict) -> list[str] | None:
    return await resolve_doc_id_allowlist(es_client, filters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_filter_resolve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/filter_resolve.py packages/retrieval-api/tests/test_loop2_filter_resolve.py
git commit -m "feat(retrieval-api): add Loop 2 ES filter -> doc_id allowlist step"
```

---

### Task 12: `retrieval-api` — Loop 2 rewritten-query retrieval + RRF merge

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/retrieve.py`
- Test: `packages/retrieval-api/tests/test_loop2_retrieve.py`

**Interfaces:**
- Consumes: `retrieval_api.gateway_client.GatewayClient.embed`, `common.milvus_client.hybrid_search`, `common.schemas.MILVUS_COLLECTIONS`.
- Produces:
  - `def rrf_merge(dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60) -> list[dict]` — pure function. Each input item is `{"chunk_id": str, ...}` already sorted best-first. Score per chunk_id = `sum(1 / (k + rank))` across whichever list(s) it appears in (rank is 1-indexed). Returns items (deduped by `chunk_id`, first-seen fields kept) sorted by combined RRF score descending, with an added `"rrf_score"` field.
  - `async def retrieve(gateway, milvus_client, rewritten_query: str, doc_id_allowlist: list[str] | None) -> list[dict]` — embeds `rewritten_query` via `gateway.embed(role="query_embed", ...)`, runs `hybrid_search` twice (once ranked by dense, once by sparse — both against the same 7 collections, `limit=50`, scoped to `doc_id_allowlist`), flattens each collection's hits into one ranked list per side, and returns `rrf_merge(...)`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_retrieve.py
from unittest.mock import AsyncMock
import pytest

from retrieval_api.loop2.retrieve import rrf_merge, retrieve


def test_rrf_merge_combines_and_ranks_by_reciprocal_rank():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"  # appears rank-1 sparse + rank-2 dense: highest combined score
    assert set(ids) == {"a", "b", "c"}
    assert merged[0]["rrf_score"] > merged[-1]["rrf_score"]


def test_rrf_merge_dedupes_by_chunk_id():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "a", "text": "A"}]

    merged = rrf_merge(dense, sparse)

    assert len(merged) == 1


@pytest.mark.asyncio
async def test_retrieve_embeds_rewritten_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.loop2.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await retrieve(gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=["d1"])

    gateway.embed.assert_awaited_once_with(role="query_embed", text="q")
    assert result[0]["chunk_id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.retrieve'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/retrieve.py
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.gateway_client import GatewayClient


def rrf_merge(dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list in (dense_ranked, sparse_ranked):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]


def _flatten(by_collection: dict[str, list[dict]]) -> list[dict]:
    flattened = [row for rows in by_collection.values() for row in rows]
    return sorted(flattened, key=lambda row: row["score"], reverse=True)


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    rewritten_query: str,
    doc_id_allowlist: list[str] | None,
) -> list[dict]:
    dense_vector = await gateway.embed(role="query_embed", text=rewritten_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )

    return rrf_merge(_flatten(dense_by_collection), _flatten(sparse_by_collection))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_retrieve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/retrieve.py packages/retrieval-api/tests/test_loop2_retrieve.py
git commit -m "feat(retrieval-api): add Loop 2 rewritten-query retrieval with RRF merge"
```

---

### Task 13: `retrieval-api` — Loop 2 rerank + citation prefetch fork

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/rerank.py`
- Create: `packages/retrieval-api/src/retrieval_api/loop2/citations.py`
- Test: `packages/retrieval-api/tests/test_loop2_rerank_citations.py`

**Interfaces:**
- Consumes: `retrieval_api.gateway_client.GatewayClient.rerank`, `common.es_client.fetch_citations`.
- Produces:
  - `async def rerank_top_chunks(gateway, query: str, candidates: list[dict], top_n: int = 3) -> list[dict]` — calls `gateway.rerank(role="reranker", query=query, documents=[c["text"] for c in candidates])`, attaches each score as `"rerank_score"`, returns the `top_n` candidates sorted by that score descending.
  - `async def prefetch_citations(es_client, candidates: list[dict], top_n_docs: int = 20) -> dict[str, dict]` — takes the top `top_n_docs` unique `doc_id`s (by `rrf_score`, first occurrence order) from `candidates`, calls `common.es_client.fetch_citations`.
  - `async def rerank_and_prefetch(gateway, es_client, query: str, candidates: list[dict]) -> tuple[list[dict], dict[str, dict]]` — runs both concurrently via `asyncio.gather`, returns `(top_chunks, citations_by_doc_id)`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_rerank_citations.py
from unittest.mock import AsyncMock
import pytest

from retrieval_api.loop2.rerank import rerank_top_chunks
from retrieval_api.loop2.citations import prefetch_citations, rerank_and_prefetch


@pytest.mark.asyncio
async def test_rerank_top_chunks_sorts_by_score_and_truncates():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.2, 0.9, 0.5]
    candidates = [
        {"chunk_id": "a", "text": "A", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "B", "rrf_score": 0.02},
        {"chunk_id": "c", "text": "C", "rrf_score": 0.01},
    ]

    result = await rerank_top_chunks(gateway, "query", candidates, top_n=2)

    assert [row["chunk_id"] for row in result] == ["b", "c"]
    assert result[0]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_prefetch_citations_dedupes_doc_ids_and_caps_at_top_n(monkeypatch):
    import retrieval_api.loop2.citations as module

    async def fake_fetch_citations(client, doc_ids):
        assert doc_ids == ["d1", "d2"]
        return {"d1": {"masterinfo": {"court": "SC"}}}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)
    candidates = [
        {"chunk_id": "x", "doc_id": "d1", "rrf_score": 0.9},
        {"chunk_id": "y", "doc_id": "d1", "rrf_score": 0.8},
        {"chunk_id": "z", "doc_id": "d2", "rrf_score": 0.7},
    ]

    result = await prefetch_citations(es_client=object(), candidates=candidates, top_n_docs=2)

    assert result == {"d1": {"masterinfo": {"court": "SC"}}}


@pytest.mark.asyncio
async def test_rerank_and_prefetch_runs_both_concurrently(monkeypatch):
    import retrieval_api.loop2.citations as citations_module
    import retrieval_api.loop2.rerank as rerank_module

    async def fake_prefetch(es_client, candidates, top_n_docs=20):
        return {"d1": {"masterinfo": {}}}

    async def fake_rerank_top(gateway, query, candidates, top_n=3):
        return [{"chunk_id": "a"}]

    monkeypatch.setattr(citations_module, "prefetch_citations", fake_prefetch)
    monkeypatch.setattr(rerank_module, "rerank_top_chunks", fake_rerank_top)

    top_chunks, citations = await rerank_and_prefetch(
        gateway=object(), es_client=object(), query="q", candidates=[{"chunk_id": "a", "doc_id": "d1"}],
    )

    assert top_chunks == [{"chunk_id": "a"}]
    assert citations == {"d1": {"masterinfo": {}}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_rerank_citations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.rerank'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/rerank.py
from retrieval_api.gateway_client import GatewayClient


async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict], top_n: int = 3
) -> list[dict]:
    scores = await gateway.rerank(role="reranker", query=query, documents=[c["text"] for c in candidates])
    scored = [{**c, "rerank_score": score} for c, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    return scored[:top_n]
```

```python
# packages/retrieval-api/src/retrieval_api/loop2/citations.py
import asyncio

from common.es_client import fetch_citations
from retrieval_api.loop2.rerank import rerank_top_chunks


async def prefetch_citations(es_client, candidates: list[dict], top_n_docs: int = 20) -> dict[str, dict]:
    ordered_by_score = sorted(candidates, key=lambda row: row["rrf_score"], reverse=True)
    seen: list[str] = []
    for row in ordered_by_score:
        doc_id = row["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) == top_n_docs:
            break
    return await fetch_citations(es_client, seen)


async def rerank_and_prefetch(
    gateway, es_client, query: str, candidates: list[dict]
) -> tuple[list[dict], dict[str, dict]]:
    top_chunks, citations = await asyncio.gather(
        rerank_top_chunks(gateway, query, candidates),
        prefetch_citations(es_client, candidates),
    )
    return top_chunks, citations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_rerank_citations.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/rerank.py packages/retrieval-api/src/retrieval_api/loop2/citations.py packages/retrieval-api/tests/test_loop2_rerank_citations.py
git commit -m "feat(retrieval-api): add Loop 2 cross-encoder rerank + ES citation prefetch fork"
```

---

### Task 14: `retrieval-api` — Loop 2 synthesis with fallback citation lookup

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/synthesize.py`
- Test: `packages/retrieval-api/tests/test_loop2_synthesize.py`

**Interfaces:**
- Consumes: `retrieval_api.gateway_client.GatewayClient.chat`, `common.es_client.fetch_citations`.
- Produces: `async def synthesize(gateway, es_client, query: str, top_chunks: list[dict], citations: dict[str, dict]) -> dict` — for each chunk in `top_chunks` whose `doc_id` is missing from `citations`, does one on-demand `fetch_citations(es_client, [doc_id])` and merges it in. Builds a prompt from `query` + chunk texts + citations, calls `gateway.chat(role="synthesis", ...)`, returns `{"answer": str, "citations": dict[str, dict]}`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_synthesize.py
from unittest.mock import AsyncMock
import pytest

from retrieval_api.loop2.synthesize import synthesize


@pytest.mark.asyncio
async def test_synthesize_uses_prefetched_citations_without_extra_lookup(monkeypatch):
    import retrieval_api.loop2.synthesize as module

    fetch_calls = []

    async def fake_fetch_citations(client, doc_ids):
        fetch_calls.append(doc_ids)
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat.return_value = "Final answer with citation."

    result = await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {"masterinfo": {"court": "SC"}}},
    )

    assert fetch_calls == []  # already prefetched, no fallback lookup needed
    assert result == {"answer": "Final answer with citation.", "citations": {"d1": {"masterinfo": {"court": "SC"}}}}


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_on_demand_lookup_for_missing_doc(monkeypatch):
    import retrieval_api.loop2.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        assert doc_ids == ["d2"]
        return {"d2": {"masterinfo": {"court": "HC"}}}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat.return_value = "Answer."

    result = await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "b", "doc_id": "d2", "text": "chunk text"}],
        citations={},
    )

    assert result["citations"] == {"d2": {"masterinfo": {"court": "HC"}}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_synthesize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.synthesize'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/synthesize.py
from common.es_client import fetch_citations


async def synthesize(gateway, es_client, query: str, top_chunks: list[dict], citations: dict) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = (
        f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}\n\n"
        "Answer the question citing the doc_id in brackets for each claim."
    )
    answer = await gateway.chat(role="synthesis", messages=[{"role": "user", "content": prompt}])

    return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_synthesize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/synthesize.py packages/retrieval-api/tests/test_loop2_synthesize.py
git commit -m "feat(retrieval-api): add Loop 2 LLM synthesis with fallback citation lookup"
```

---

### Task 15: `retrieval-api` — Loop 2 pipeline wiring

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/loop2/pipeline.py`
- Test: `packages/retrieval-api/tests/test_loop2_pipeline.py`

**Interfaces:**
- Consumes: `extract_intent`, `resolve_allowlist`, `retrieve`, `rerank_and_prefetch`, `synthesize` (all from this task's sibling modules).
- Produces: `async def run_loop2(gateway, es_client, milvus_client, query: str) -> dict` — runs the six spec-numbered steps in order, wrapping the whole thing in one `try/except`. On success returns `{"ok": True, "answer": str, "citations": dict}`. On any exception returns `{"ok": False, "error": str}` — this is what `ws.py` (Task 16) turns into `loop2_done` vs `loop2_error`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_loop2_pipeline.py
import pytest

from retrieval_api.loop2.pipeline import run_loop2


@pytest.mark.asyncio
async def test_run_loop2_success_path(monkeypatch):
    import retrieval_api.loop2.pipeline as module

    async def fake_extract_intent(gateway, query):
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters):
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_loop2(gateway=object(), es_client=object(), milvus_client=object(), query="original query")

    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}}


@pytest.mark.asyncio
async def test_run_loop2_returns_error_on_any_stage_failure(monkeypatch):
    import retrieval_api.loop2.pipeline as module

    async def failing_extract_intent(gateway, query):
        raise ValueError("SLM did not return valid JSON")

    monkeypatch.setattr(module, "extract_intent", failing_extract_intent)

    result = await run_loop2(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": False, "error": "SLM did not return valid JSON"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.loop2.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/loop2/pipeline.py
from retrieval_api.loop2.intent import extract_intent
from retrieval_api.loop2.filter_resolve import resolve_allowlist
from retrieval_api.loop2.retrieve import retrieve
from retrieval_api.loop2.citations import rerank_and_prefetch
from retrieval_api.loop2.synthesize import synthesize


async def run_loop2(gateway, es_client, milvus_client, query: str) -> dict:
    try:
        intent_result = await extract_intent(gateway, query)
        doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"])
        candidates = await retrieve(gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist)
        top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates)
        synthesis = await synthesize(gateway, es_client, query, top_chunks, citations)
        return {"ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"]}
    except Exception as exc:  # noqa: BLE001 - Loop 2 failure must never crash Loop 1's result
        return {"ok": False, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_loop2_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/loop2/pipeline.py packages/retrieval-api/tests/test_loop2_pipeline.py
git commit -m "feat(retrieval-api): wire Loop 2 pipeline stages 1-6 with error isolation"
```

---

### Task 16: `retrieval-api` — WebSocket endpoint + app wiring

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/ws.py`
- Create: `packages/retrieval-api/src/retrieval_api/main.py`
- Test: `packages/retrieval-api/tests/test_ws_integration.py`

**Interfaces:**
- Consumes: `retrieval_api.loop1.search.run_loop1`, `retrieval_api.loop2.pipeline.run_loop2`, `retrieval_api.gateway_client.GatewayClient`.
- Produces: FastAPI `WebSocket` route `/ws/search`. Protocol: client sends `{"query": str}`; server sends, in order: one `{"type": "loop1_result", **run_loop1_output}` (Loop 1 and Loop 2 dispatched via `asyncio.gather` at the same time, but Loop 1's message is sent as soon as its own task resolves — it does not wait for Loop 2), then either `{"type": "loop2_done", "answer": str, "citations": dict}` or `{"type": "loop2_error", "error": str}`.

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_ws_integration.py
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module


def test_ws_search_sends_loop1_then_loop2_events(monkeypatch):
    async def fake_run_loop1(es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_loop2(gateway, es_client, milvus_client, query):
        return {"ok": True, "answer": "final answer", "citations": {"d1": {}}}

    monkeypatch.setattr(ws_module, "run_loop1", fake_run_loop1)
    monkeypatch.setattr(ws_module, "run_loop2", fake_run_loop2)
    monkeypatch.setattr(ws_module, "get_es_client", lambda: object())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda: object())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "tax exemption"})

        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "loop1_result"
    assert first["es"] == [{"doc_id": "d1"}]
    assert second == {"type": "loop2_done", "answer": "final answer", "citations": {"d1": {}}}


def test_ws_search_sends_loop2_error_event_on_failure(monkeypatch):
    async def fake_run_loop1(es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_loop2(gateway, es_client, milvus_client, query):
        return {"ok": False, "error": "gateway unreachable"}

    monkeypatch.setattr(ws_module, "run_loop1", fake_run_loop1)
    monkeypatch.setattr(ws_module, "run_loop2", fake_run_loop2)
    monkeypatch.setattr(ws_module, "get_es_client", lambda: object())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda: object())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})
        websocket.receive_json()  # loop1_result
        second = websocket.receive_json()

    assert second == {"type": "loop2_error", "error": "gateway unreachable"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.ws'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/retrieval-api/src/retrieval_api/ws.py
import asyncio

from fastapi import APIRouter, WebSocket

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.loop1.search import run_loop1
from retrieval_api.loop2.pipeline import run_loop2

router = APIRouter()


def get_gateway_client() -> GatewayClient:
    return GatewayClient(base_url=get_settings().gateway_url)


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]

    es_client = get_es_client()
    milvus_client = get_milvus_client()
    gateway = get_gateway_client()

    loop1_task = asyncio.create_task(run_loop1(es_client, milvus_client, query))
    loop2_task = asyncio.create_task(run_loop2(gateway, es_client, milvus_client, query))

    loop1_result = await loop1_task
    await websocket.send_json({"type": "loop1_result", **loop1_result})

    loop2_result = await loop2_task
    if loop2_result["ok"]:
        await websocket.send_json({
            "type": "loop2_done", "answer": loop2_result["answer"], "citations": loop2_result["citations"],
        })
    else:
        await websocket.send_json({"type": "loop2_error", "error": loop2_result["error"]})

    await websocket.close()
```

```python
# packages/retrieval-api/src/retrieval_api/main.py
from fastapi import FastAPI

from retrieval_api.ws import router

app = FastAPI(title="retrieval-api")
app.include_router(router)
```

**Note for implementer:** `common.config.get_settings` and `common.es_client.get_es_client`/`common.milvus_client.get_milvus_client` must already exist from Tasks 2, 4, 5 — this task only adds `get_gateway_client` locally in `ws.py` since it's `retrieval-api`-specific (the gateway is not something `common` needs to know about).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/retrieval-api && uv run pytest tests/test_ws_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ws.py packages/retrieval-api/src/retrieval_api/main.py packages/retrieval-api/tests/test_ws_integration.py
git commit -m "feat(retrieval-api): add /ws/search endpoint wiring Loop 1 + Loop 2"
```

---

### Task 17: docker-compose wiring

**Files:**
- Create: `docker-compose.yml`
- Create: `packages/model-gateway/Dockerfile`
- Create: `packages/retrieval-api/Dockerfile`

**Interfaces:**
- Produces: a runnable two-service stack (`model-gateway` on `:8001`, `retrieval-api` on `:8000`), both built via `uv sync --package <name>` + `uvicorn`, `retrieval-api` given `GATEWAY_URL=http://model-gateway:8001` so the two containers can reach each other by service name.

- [ ] **Step 1: Write `packages/model-gateway/Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --package model-gateway
CMD ["uv", "run", "--package", "model-gateway", "uvicorn", "model_gateway.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 2: Write `packages/retrieval-api/Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --package retrieval-api
CMD ["uv", "run", "--package", "retrieval-api", "uvicorn", "retrieval_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  model-gateway:
    build:
      context: .
      dockerfile: packages/model-gateway/Dockerfile
    ports: ["8001:8001"]
    env_file: .env

  retrieval-api:
    build:
      context: .
      dockerfile: packages/retrieval-api/Dockerfile
    ports: ["8000:8000"]
    env_file: .env
    environment:
      GATEWAY_URL: http://model-gateway:8001
    depends_on: [model-gateway]
```

- [ ] **Step 4: Verify the stack builds**

Run: `docker compose build`
Expected: both images build without error.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml packages/model-gateway/Dockerfile packages/retrieval-api/Dockerfile
git commit -m "chore: add docker-compose stack for model-gateway + retrieval-api"
```

---

## Self-Review Notes

- **Spec coverage:** Loop 1 (Task 9), Loop 2 steps 1-6 (Tasks 10-15), WebSocket wiring (Task 16), model-gateway (Tasks 6-7), error handling/partial-failure isolation (Tasks 9, 15, 16), docker-compose (Task 17) — all spec sections have a task. `dense_vector_2` and a second gateway provider are explicitly out of scope per the spec's "Open items" and left untouched.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `chunk_id`/`doc_id`/`text`/`score` row shape introduced in Task 4 (`common.milvus_client`) is reused unchanged through Tasks 9, 12, 13, 14. `GatewayClient.chat/embed/rerank` signatures from Task 8 match every call site in Tasks 10, 12, 13, 14. `run_loop2`'s `{"ok": bool, ...}` shape from Task 15 matches its consumption in Task 16.
