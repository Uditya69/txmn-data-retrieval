# Agentic Search Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third retrieval path — a tool-calling LLM agent over ES/Milvus search tools, with citation validation and a standalone trace UI — that can be A/B evaluated against AI Mode via the existing eval CLI.

**Architecture:** New `packages/agents` package holds tool schemas, the tool-calling loop, and citation validation, all independent of `retrieval-api`. `model-gateway` gains generic tool-calling passthrough (new `tools`/`tool_choice` fields on `/v1/chat`, new `agent_chat` role) — it does not know about search tools, it just forwards OpenAI-shaped tool-calling to DeepInfra. `retrieval-api` wires `packages/agents` to a new `/ws/agent` WebSocket route (mirrors the existing `/ws/search` streaming pattern) and to the eval CLI. `packages/web` gets a new standalone page at `/agent`.

**Tech Stack:** Python 3.11, FastAPI, httpx, pydantic-settings, pytest + pytest-asyncio + respx (backend); React + TypeScript, Vite, vitest + @testing-library/react (frontend). uv workspace monorepo.

## Global Constraints

- Python 3.11, not 3.14 (pymilvus/grpcio has no 3.14 wheel).
- `query_embed` role goes through Voyage only, never DeepInfra — do not touch this routing.
- Milvus `sparse_vector` is queried by passing raw query text to `data=[...]`, never a computed vector, never set by client code.
- No ranking fusion between ES and Milvus scores anywhere in this feature — the agent's tool results are surfaced to the model as separate, unmerged sets.
- AI Mode's existing "search all 7 collections every query" rule is unaffected by this work — the agent path instead lets the model choose which collection(s) to query.
- pydantic-settings env var matching is literal on the field name (`deepinfra_chat_model_agent` → `DEEPINFRA_CHAT_MODEL_AGENT`) — any new required `GatewaySettings` field needs a matching dummy env var added to `packages/model-gateway/tests/conftest.py` or all gateway tests fail at import time.
- `monkeypatch.setattr` must target the *consuming* module's namespace (e.g. `ws_module`, `routes_module`), not the original definition module — see existing test files for the pattern.
- Citation-retry loop is capped at 3 total attempts; on exhaustion, return an explicit unverifiable-answer state — never a best-effort/unverified guess.
- Tool-calling loop itself has no step cap (may be added later; out of scope here).

---

### Task 1: model-gateway — tool-calling passthrough on `/v1/chat`

**Files:**
- Modify: `packages/model-gateway/src/model_gateway/adapters/base.py`
- Modify: `packages/model-gateway/src/model_gateway/adapters/deepinfra.py`
- Modify: `packages/model-gateway/src/model_gateway/routes.py`
- Modify (existing tests, signature changed): `packages/model-gateway/tests/test_deepinfra_adapter.py`
- Modify (existing tests, signature changed): `packages/model-gateway/tests/test_routes.py`

**Interfaces:**
- Produces: `DeepInfraAdapter.chat(model, messages, tools=None, tool_choice=None) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]` (content, usage_details, reasoning, tool_calls).
- Produces: `POST /v1/chat` request body gains optional `tools: list[dict] | None`, `tool_choice: str | None`; response body gains `tool_calls: list[dict] | None`.

- [ ] **Step 1: Update existing DeepInfra adapter tests for the new 4-tuple return**

Edit `packages/model-gateway/tests/test_deepinfra_adapter.py` — change the two `chat` tests' unpacking from 3-tuple to 4-tuple:

```python
    content, usage, reasoning, tool_calls = await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    assert content == "hello"
    assert usage == {"input": 10, "output": 5}
    assert reasoning is None
    assert tool_calls is None
```

and

```python
    _content, _usage, reasoning, _tool_calls = await adapter.chat("reasoning-model", [{"role": "user", "content": "hi"}])

    assert reasoning == "thinking it through..."
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/model-gateway/tests/test_deepinfra_adapter.py -v`
Expected: FAIL — `too many values to unpack` (current `chat` still returns a 3-tuple).

- [ ] **Step 3: Add a new failing test for tool-call passthrough**

Append to `packages/model-gateway/tests/test_deepinfra_adapter.py`:

```python
@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_tools_and_returns_tool_calls():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
            }}],
        })
    )
    adapter = DeepInfraAdapter(api_key="k")
    tools = [{"type": "function", "function": {"name": "search_es", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    content, _usage, _reasoning, tool_calls = await adapter.chat(
        "some-model", [{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto",
    )

    assert content is None
    assert tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}]
    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_tools_key_when_not_given():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "tools" not in sent
    assert "tool_choice" not in sent
```

- [ ] **Step 4: Run to verify both new tests fail**

Run: `uv run pytest packages/model-gateway/tests/test_deepinfra_adapter.py -v`
Expected: FAIL — `chat() got an unexpected keyword argument 'tools'`.

- [ ] **Step 5: Implement — update `base.py` Protocol**

Replace the `chat` line in `packages/model-gateway/src/model_gateway/adapters/base.py`:

```python
from typing import Protocol


class ModelAdapter(Protocol):
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]: ...
    async def embed(self, model: str, text: str) -> list[float]: ...
    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]: ...
```

- [ ] **Step 6: Implement — update `DeepInfraAdapter.chat`**

In `packages/model-gateway/src/model_gateway/adapters/deepinfra.py`, replace the `chat` method:

```python
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]:
        payload = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{_BASE_URL}/openai/chat/completions",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            message = data["choices"][0]["message"]
            return (
                message.get("content"),
                _openai_usage_details(usage),
                message.get("reasoning_content"),
                message.get("tool_calls"),
            )
```

- [ ] **Step 7: Run to verify the adapter tests now pass**

Run: `uv run pytest packages/model-gateway/tests/test_deepinfra_adapter.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 8: Update existing route tests for the new 4-tuple mock and route response shape**

Edit `packages/model-gateway/tests/test_routes.py` — update the two mocks that set `fake_adapter.chat.return_value` and their assertions:

```python
def test_chat_route_resolves_role_and_calls_deepinfra_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {"input": 3, "output": 2}, None, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.json() == {"content": "the answer", "reasoning": None, "tool_calls": None}
    fake_adapter.chat.assert_awaited_once_with("big-model", [{"role": "user", "content": "hi"}], None, None)


def test_chat_route_surfaces_reasoning_when_present(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, "thinking it through...", None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"synthesis": "big-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"synthesis": "deepinfra"})

    client = TestClient(app)
    response = client.post("/v1/chat", json={"role": "synthesis", "messages": [{"role": "user", "content": "hi"}]})

    assert response.json() == {"content": "the answer", "reasoning": "thinking it through...", "tool_calls": None}
```

- [ ] **Step 9: Add a new failing route test for tool-call passthrough**

Append to `packages/model-gateway/tests/test_routes.py`:

```python
def test_chat_route_passes_tools_and_returns_tool_calls(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = (None, {}, None, [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{}"}}])
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"agent_chat": "agent-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"agent_chat": "deepinfra"})
    tools = [{"type": "function", "function": {"name": "search_es", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    client = TestClient(app)
    response = client.post("/v1/chat", json={
        "role": "agent_chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools, "tool_choice": "auto",
    })

    assert response.json()["tool_calls"] == [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{}"}}]
    fake_adapter.chat.assert_awaited_once_with("agent-model", [{"role": "user", "content": "hi"}], tools, "auto")
```

- [ ] **Step 10: Run to verify failures**

Run: `uv run pytest packages/model-gateway/tests/test_routes.py -v`
Expected: FAIL — `ValidationError`/`AttributeError` (route doesn't accept `tools`/`tool_choice` yet, doesn't unpack 4-tuple).

- [ ] **Step 11: Implement — update `ChatRequest` and the `/v1/chat` route**

In `packages/model-gateway/src/model_gateway/routes.py`, update `ChatRequest` and the `chat` function:

```python
class ChatRequest(BaseModel):
    role: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | None = None


@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    model, provider = _resolve(req.role)
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"chat:{req.role}",
        model=model,
        input=req.messages,
        metadata={"provider": provider, "has_tools": bool(req.tools)},
        trace_context=_trace_context_from_headers(request),
    ) as generation:
        content, usage_details, reasoning, tool_calls = await get_adapter(provider).chat(
            model, req.messages, req.tools, req.tool_choice,
        )
        generation.update(output=content if content is not None else {"tool_calls": tool_calls}, usage_details=usage_details)
        if reasoning:
            generation.update(metadata={"reasoning": reasoning})
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}
```

- [ ] **Step 12: Run all model-gateway tests to verify everything passes**

Run: `uv run pytest packages/model-gateway/tests -v`
Expected: PASS (all tests, including the unrelated embed/rerank route tests which are untouched).

- [ ] **Step 13: Commit**

```bash
git add packages/model-gateway/src/model_gateway/adapters/base.py \
        packages/model-gateway/src/model_gateway/adapters/deepinfra.py \
        packages/model-gateway/src/model_gateway/routes.py \
        packages/model-gateway/tests/test_deepinfra_adapter.py \
        packages/model-gateway/tests/test_routes.py
git commit -m "feat: add tool-calling passthrough to model-gateway /v1/chat"
```

---

### Task 2: model-gateway — `agent_chat` role

**Files:**
- Modify: `packages/model-gateway/src/model_gateway/config.py`
- Modify: `packages/model-gateway/tests/conftest.py`
- Modify: `.env.example`
- Test: `packages/model-gateway/tests/test_config.py` (new file)

**Interfaces:**
- Produces: `build_role_model_map(settings)["agent_chat"]`, `build_role_provider_map()["agent_chat"] == "deepinfra"`.

- [ ] **Step 1: Write the failing test**

Create `packages/model-gateway/tests/test_config.py`:

```python
from model_gateway.config import GatewaySettings, build_role_model_map, build_role_provider_map


def test_agent_chat_role_maps_to_its_own_model_and_deepinfra():
    settings = GatewaySettings(
        deepinfra_api_key="k",
        deepinfra_chat_model_slm="slm-model",
        deepinfra_chat_model_synthesis="synthesis-model",
        deepinfra_chat_model_agent="agent-model",
        deepinfra_rerank_model="rerank-model",
        voyage_api_key="k",
        voyage_embed_model="embed-model",
    )

    model_map = build_role_model_map(settings)
    provider_map = build_role_provider_map()

    assert model_map["agent_chat"] == "agent-model"
    assert provider_map["agent_chat"] == "deepinfra"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/model-gateway/tests/test_config.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'deepinfra_chat_model_agent'`.

- [ ] **Step 3: Implement**

In `packages/model-gateway/src/model_gateway/config.py`, add the field and both map entries:

```python
class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepinfra_api_key: str
    deepinfra_chat_model_slm: str
    deepinfra_chat_model_synthesis: str
    deepinfra_chat_model_agent: str
    deepinfra_rerank_model: str
    voyage_api_key: str
    voyage_embed_model: str
```

```python
def build_role_model_map(settings: GatewaySettings) -> dict[str, str]:
    return {
        "slm": settings.deepinfra_chat_model_slm,
        "synthesis": settings.deepinfra_chat_model_synthesis,
        "agent_chat": settings.deepinfra_chat_model_agent,
        "query_embed": settings.voyage_embed_model,
        "reranker": settings.deepinfra_rerank_model,
    }


def build_role_provider_map() -> dict[str, str]:
    return {
        "slm": "deepinfra",
        "synthesis": "deepinfra",
        "agent_chat": "deepinfra",
        "reranker": "deepinfra",
        "query_embed": "voyage",
    }
```

- [ ] **Step 4: Add the dummy env var to conftest so the rest of the suite still imports cleanly**

In `packages/model-gateway/tests/conftest.py`, add one line:

```python
os.environ.setdefault("DEEPINFRA_CHAT_MODEL_AGENT", "test-agent-model")
```

- [ ] **Step 5: Add the env var to `.env.example`**

In `.env.example`, under the `# model-gateway` section, add after `DEEPINFRA_CHAT_MODEL_SYNTHESIS`:

```
DEEPINFRA_CHAT_MODEL_AGENT=meta-llama/Meta-Llama-3.1-70B-Instruct
```

- [ ] **Step 6: Run to verify it passes, then run the full model-gateway suite**

Run: `uv run pytest packages/model-gateway/tests -v`
Expected: PASS (all tests — this confirms the conftest change didn't break existing import-time config building).

- [ ] **Step 7: Commit**

```bash
git add packages/model-gateway/src/model_gateway/config.py \
        packages/model-gateway/tests/conftest.py \
        packages/model-gateway/tests/test_config.py \
        .env.example
git commit -m "feat: add agent_chat role to model-gateway"
```

---

### Task 3: retrieval-api — `GatewayClient.chat_with_tools`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/gateway_client.py`
- Test: `packages/retrieval-api/tests/test_gateway_client.py` (check if it exists first; create if not, following existing test conventions in that directory)

**Interfaces:**
- Consumes: model-gateway's `POST /v1/chat` with `tools`/`tool_choice` (Task 1).
- Produces: `GatewayClient.chat_with_tools(role: str, messages: list[dict], tools: list[dict], tool_choice: str | None = None) -> dict` returning `{"content": str | None, "tool_calls": list[dict] | None, "reasoning": str | None}`. This is what `packages/agents` (Task 5) calls.

- [ ] **Step 1: Check for an existing gateway client test file**

Run: `ls packages/retrieval-api/tests/ | grep -i gateway`

If `test_gateway_client.py` exists, read it and match its exact mocking style (likely `respx` against the gateway's base URL) for the new test below. If it doesn't exist, create it using the pattern shown in Step 2.

- [ ] **Step 2: Write the failing test**

In `packages/retrieval-api/tests/test_gateway_client.py` (create or extend):

```python
import json

import httpx
import pytest
import respx

from retrieval_api.gateway_client import GatewayClient


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_tools_posts_tools_and_returns_tool_calls():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={
            "content": None,
            "reasoning": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
        })
    )
    client = GatewayClient(base_url="http://gateway", trace_enabled=False)
    tools = [{"type": "function", "function": {"name": "search_es", "description": "d", "parameters": {"type": "object", "properties": {}}}}]

    result = await client.chat_with_tools("agent_chat", [{"role": "user", "content": "hi"}], tools, tool_choice="auto")

    assert result == {
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"gst\"}"}}],
        "reasoning": None,
    }
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"role": "agent_chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools, "tool_choice": "auto"}
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: FAIL — `AttributeError: 'GatewayClient' object has no attribute 'chat_with_tools'`.

- [ ] **Step 4: Implement**

In `packages/retrieval-api/src/retrieval_api/gateway_client.py`, add the method (after `chat_with_reasoning`):

```python
    async def chat_with_tools(
        self, role: str, messages: list[dict], tools: list[dict], tool_choice: str | None = None,
    ) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat",
                json={"role": role, "messages": messages, "tools": tools, "tool_choice": tool_choice},
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            return {"content": data.get("content"), "tool_calls": data.get("tool_calls"), "reasoning": data.get("reasoning")}
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/gateway_client.py packages/retrieval-api/tests/test_gateway_client.py
git commit -m "feat: add chat_with_tools to retrieval-api GatewayClient"
```

---

### Task 4: `packages/agents` — package scaffold + tool schemas/dispatch

**Files:**
- Create: `packages/agents/pyproject.toml`
- Create: `packages/agents/src/agents/__init__.py`
- Create: `packages/agents/src/agents/tools.py`
- Create: `packages/agents/tests/__init__.py` (empty)
- Create: `packages/agents/tests/test_tools.py`
- Modify: `pyproject.toml` (workspace root)

**Interfaces:**
- Consumes: `common.es_client.raw_search(client, query, limit=20) -> list[dict]`, `common.es_client.fetch_citations(client, doc_ids) -> dict[str, dict]`, `common.milvus_client.hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50) -> dict[str, list[dict]]`, `common.schemas.MILVUS_COLLECTIONS`.
- Produces: `TOOL_SCHEMAS: list[dict]` (OpenAI function-calling format, 4 tools), `async def dispatch_tool_call(name: str, arguments: dict, *, gateway, es_client, milvus_client) -> dict` — returns `{"rows": [...]}` for the three search tools or `{"citation": dict | None}` for `lookup_doc`; raises `ValueError` for an unknown tool name. This is consumed by `agents/loop.py` (Task 5).

- [ ] **Step 1: Create the package scaffold**

`packages/agents/pyproject.toml`:

```toml
[project]
name = "agents"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "common",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agents"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`packages/agents/src/agents/__init__.py`: empty file.

`packages/agents/tests/__init__.py`: empty file.

- [ ] **Step 2: Register the package in the workspace**

In root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api", "packages/agents"]

[tool.uv.sources]
common = { workspace = true }
model-gateway = { workspace = true }
retrieval-api = { workspace = true }
agents = { workspace = true }
```

- [ ] **Step 3: Sync the workspace so the new package installs editable**

Run: `uv sync --all-packages`
Expected: completes without error; `agents` package now importable.

- [ ] **Step 4: Write the failing tests**

Create `packages/agents/tests/test_tools.py`:

```python
import pytest

from agents.tools import TOOL_SCHEMAS, dispatch_tool_call


def test_tool_schemas_cover_all_four_tools_with_milvus_collection_enum():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"search_es", "search_milvus_dense", "search_milvus_sparse", "lookup_doc"}
    dense_schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_milvus_dense")
    assert dense_schema["function"]["parameters"]["properties"]["collection"]["enum"] == [
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    ]


@pytest.mark.asyncio
async def test_dispatch_search_es_calls_raw_search(monkeypatch):
    import agents.tools as tools_module

    async def fake_raw_search(client, query, limit=20):
        assert query == "gst exemption"
        return [{"doc_id": "d1", "score": 1.0}]

    monkeypatch.setattr(tools_module, "raw_search", fake_raw_search)

    result = await dispatch_tool_call(
        "search_es", {"query": "gst exemption"}, gateway=None, es_client=object(), milvus_client=None,
    )

    assert result == {"rows": [{"doc_id": "d1", "score": 1.0}]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_dense_embeds_then_searches_one_collection(monkeypatch):
    import agents.tools as tools_module

    class FakeGateway:
        async def embed(self, role, text):
            assert role == "query_embed"
            return [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert collections == ["held"]
        assert dense_vector == [0.1, 0.2]
        return {"held": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9}]}

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_dense", {"collection": "held", "query": "gst"},
        gateway=FakeGateway(), es_client=None, milvus_client=object(),
    )

    assert result == {"rows": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9}]}


@pytest.mark.asyncio
async def test_dispatch_search_milvus_sparse_passes_none_dense_vector(monkeypatch):
    import agents.tools as tools_module

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        assert dense_vector is None
        assert sparse_query_text == "gst"
        assert collections == ["digest"]
        return {"digest": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1}]}

    monkeypatch.setattr(tools_module, "hybrid_search", fake_hybrid_search)

    result = await dispatch_tool_call(
        "search_milvus_sparse", {"collection": "digest", "query": "gst"},
        gateway=object(), es_client=None, milvus_client=object(),
    )

    assert result == {"rows": [{"chunk_id": "c2", "doc_id": "d2", "score": 3.1}]}


@pytest.mark.asyncio
async def test_dispatch_lookup_doc_returns_citation_or_none(monkeypatch):
    import agents.tools as tools_module

    async def fake_fetch_citations(client, doc_ids):
        assert doc_ids == ["d1"]
        return {"d1": {"court": "SC"}}

    monkeypatch.setattr(tools_module, "fetch_citations", fake_fetch_citations)

    found = await dispatch_tool_call("lookup_doc", {"doc_id": "d1"}, gateway=None, es_client=object(), milvus_client=None)
    assert found == {"citation": {"court": "SC"}}

    missing = await dispatch_tool_call("lookup_doc", {"doc_id": "d999"}, gateway=None, es_client=object(), milvus_client=None)
    assert missing == {"citation": None}


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises_value_error():
    with pytest.raises(ValueError, match="unknown tool"):
        await dispatch_tool_call("not_a_tool", {}, gateway=None, es_client=None, milvus_client=None)
```

- [ ] **Step 5: Run to verify it fails**

Run: `uv run pytest packages/agents/tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.tools'`.

- [ ] **Step 6: Implement**

Create `packages/agents/src/agents/tools.py`:

```python
from common.es_client import fetch_citations, raw_search
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_es",
            "description": "Full-text search over the Elasticsearch case-law index (facts, held, headnotes).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_milvus_dense",
            "description": "Dense embedding similarity search within one Milvus collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": MILVUS_COLLECTIONS},
                    "query": {"type": "string"},
                },
                "required": ["collection", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_milvus_sparse",
            "description": "BM25 sparse search within one Milvus collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": MILVUS_COLLECTIONS},
                    "query": {"type": "string"},
                },
                "required": ["collection", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_doc",
            "description": "Fetch citation metadata for a doc_id already seen from another tool's results.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
]


async def dispatch_tool_call(name: str, arguments: dict, *, gateway, es_client, milvus_client) -> dict:
    if name == "search_es":
        rows = await raw_search(es_client, arguments["query"])
        return {"rows": rows}
    if name == "search_milvus_dense":
        collection = arguments["collection"]
        vector = await gateway.embed(role="query_embed", text=arguments["query"])
        result = await hybrid_search(milvus_client, [collection], vector, arguments["query"])
        return {"rows": result.get(collection, [])}
    if name == "search_milvus_sparse":
        collection = arguments["collection"]
        result = await hybrid_search(milvus_client, [collection], None, arguments["query"])
        return {"rows": result.get(collection, [])}
    if name == "lookup_doc":
        citations = await fetch_citations(es_client, [arguments["doc_id"]])
        return {"citation": citations.get(arguments["doc_id"])}
    raise ValueError(f"unknown tool: {name}")
```

- [ ] **Step 7: Run to verify it passes**

Run: `uv run pytest packages/agents/tests/test_tools.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 8: Commit**

```bash
git add packages/agents pyproject.toml
git commit -m "feat: scaffold agents package with search tool schemas/dispatch"
```

---

### Task 5: `packages/agents` — tool-calling loop

**Files:**
- Create: `packages/agents/src/agents/loop.py`
- Create: `packages/agents/tests/test_loop.py`

**Interfaces:**
- Consumes: `agents.tools.TOOL_SCHEMAS`, `agents.tools.dispatch_tool_call` (Task 4); `gateway.chat_with_tools(role, messages, tools, tool_choice=None) -> dict` (Task 3).
- Produces: `OnStep = Callable[[str, dict], Awaitable[None]]`; `SYSTEM_PROMPT: str`; `build_initial_messages(query: str) -> list[dict]`; `async def run_agent_loop(gateway, es_client, milvus_client, messages: list[dict], seen_doc_ids: set[str], on_step: OnStep | None = None) -> dict` returning `{"answer": str, "seen_doc_ids": set[str], "messages": list[dict]}`. Consumed by `agents/pipeline.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

Create `packages/agents/tests/test_loop.py`:

```python
import pytest

from agents.loop import build_initial_messages, run_agent_loop


def test_build_initial_messages_has_system_and_user_turns():
    messages = build_initial_messages("what is the rate for GST on X")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "what is the rate for GST on X"}


@pytest.mark.asyncio
async def test_loop_returns_answer_immediately_when_no_tool_calls():
    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            return {"content": "final answer, no tools needed", "tool_calls": None, "reasoning": None}

    result = await run_agent_loop(
        FakeGateway(), es_client=None, milvus_client=None,
        messages=build_initial_messages("q"), seen_doc_ids=set(),
    )

    assert result["answer"] == "final answer, no tools needed"
    assert result["seen_doc_ids"] == set()


@pytest.mark.asyncio
async def test_loop_dispatches_tool_call_tracks_doc_ids_and_continues_until_final_answer(monkeypatch):
    import agents.loop as loop_module

    calls = {"n": 0}

    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "search_es", "arguments": '{"query": "gst"}',
                    }}],
                    "reasoning": None,
                }
            return {"content": "answer citing [d1]", "tool_calls": None, "reasoning": None}

    async def fake_dispatch_tool_call(name, arguments, *, gateway, es_client, milvus_client):
        assert name == "search_es"
        assert arguments == {"query": "gst"}
        return {"rows": [{"doc_id": "d1", "score": 1.0}]}

    monkeypatch.setattr(loop_module, "dispatch_tool_call", fake_dispatch_tool_call)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await run_agent_loop(
        FakeGateway(), es_client=object(), milvus_client=object(),
        messages=build_initial_messages("q"), seen_doc_ids=set(), on_step=on_step,
    )

    assert result["answer"] == "answer citing [d1]"
    assert result["seen_doc_ids"] == {"d1"}
    assert [s for s, _ in steps] == ["agent_tool_call", "agent_tool_result"]
    assert steps[0][1] == {"name": "search_es", "arguments": {"query": "gst"}}
    assert steps[1][1] == {"name": "search_es", "result": {"rows": [{"doc_id": "d1", "score": 1.0}]}}


@pytest.mark.asyncio
async def test_loop_records_lookup_doc_citation_doc_id_and_survives_tool_error(monkeypatch):
    import agents.loop as loop_module

    calls = {"n": 0}

    class FakeGateway:
        async def chat_with_tools(self, role, messages, tools, tool_choice=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search_es", "arguments": "{\"query\": \"x\"}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "lookup_doc", "arguments": "{\"doc_id\": \"d2\"}"}},
                    ],
                    "reasoning": None,
                }
            return {"content": "done", "tool_calls": None, "reasoning": None}

    async def fake_dispatch_tool_call(name, arguments, *, gateway, es_client, milvus_client):
        if name == "search_es":
            raise RuntimeError("ES timed out")
        return {"citation": {"court": "SC"}}

    monkeypatch.setattr(loop_module, "dispatch_tool_call", fake_dispatch_tool_call)

    result = await run_agent_loop(
        FakeGateway(), es_client=object(), milvus_client=object(),
        messages=build_initial_messages("q"), seen_doc_ids=set(),
    )

    assert result["answer"] == "done"
    assert result["seen_doc_ids"] == {"d2"}
    tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
    assert "RuntimeError: ES timed out" in tool_messages[0]["content"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/agents/tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.loop'`.

- [ ] **Step 3: Implement**

Create `packages/agents/src/agents/loop.py`:

```python
import json
from typing import Awaitable, Callable

from agents.tools import TOOL_SCHEMAS, dispatch_tool_call

OnStep = Callable[[str, dict], Awaitable[None]]

SYSTEM_PROMPT = (
    "You are a legal research assistant over Indian case-law. Use the available "
    "search tools to find evidence before answering. Every claim in your final "
    "answer must be backed by a doc_id you retrieved via a tool call this "
    "session. Cite doc_ids inline in square brackets, e.g. [12345]. Never cite "
    "a doc_id you did not actually retrieve."
)


def build_initial_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def _collect_doc_ids(tool_name: str, arguments: dict, result: dict) -> set[str]:
    doc_ids: set[str] = set()
    for row in result.get("rows", []):
        if "doc_id" in row:
            doc_ids.add(str(row["doc_id"]))
    if tool_name == "lookup_doc" and result.get("citation") is not None:
        doc_ids.add(str(arguments.get("doc_id")))
    return doc_ids


async def run_agent_loop(
    gateway, es_client, milvus_client, messages: list[dict], seen_doc_ids: set[str], on_step: OnStep | None = None,
) -> dict:
    messages = list(messages)
    seen_doc_ids = set(seen_doc_ids)

    while True:
        response = await gateway.chat_with_tools(role="agent_chat", messages=messages, tools=TOOL_SCHEMAS)
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return {"answer": response.get("content") or "", "seen_doc_ids": seen_doc_ids, "messages": messages}

        messages.append({"role": "assistant", "content": response.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            if on_step:
                await on_step("agent_tool_call", {"name": name, "arguments": arguments})
            try:
                result = await dispatch_tool_call(
                    name, arguments, gateway=gateway, es_client=es_client, milvus_client=milvus_client,
                )
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            seen_doc_ids |= _collect_doc_ids(name, arguments, result)
            if on_step:
                await on_step("agent_tool_result", {"name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/agents/tests/test_loop.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/agents/src/agents/loop.py packages/agents/tests/test_loop.py
git commit -m "feat: add uncapped tool-calling agent loop"
```

---

### Task 6: `packages/agents` — citation validator

**Files:**
- Create: `packages/agents/src/agents/citations.py`
- Create: `packages/agents/tests/test_citations.py`

**Interfaces:**
- Produces: `extract_cited_doc_ids(answer: str) -> set[str]`; `validate_citations(answer: str, seen_doc_ids: set[str]) -> list[str]` (sorted list of invalid cited doc_ids; empty means fully valid). Consumed by `agents/pipeline.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

Create `packages/agents/tests/test_citations.py`:

```python
from agents.citations import extract_cited_doc_ids, validate_citations


def test_extract_cited_doc_ids_finds_all_bracketed_ids():
    answer = "The rate is 10% per [12345] and confirmed in [67890]."
    assert extract_cited_doc_ids(answer) == {"12345", "67890"}


def test_extract_cited_doc_ids_returns_empty_set_with_no_citations():
    assert extract_cited_doc_ids("No citations here.") == set()


def test_validate_citations_returns_empty_list_when_all_cited_ids_were_seen():
    answer = "See [12345] and [67890]."
    assert validate_citations(answer, {"12345", "67890", "99999"}) == []


def test_validate_citations_returns_sorted_invalid_ids():
    answer = "See [999] and [111] and [222]."
    assert validate_citations(answer, {"222"}) == ["111", "999"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/agents/tests/test_citations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.citations'`.

- [ ] **Step 3: Implement**

Create `packages/agents/src/agents/citations.py`:

```python
import re

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_cited_doc_ids(answer: str) -> set[str]:
    return set(_CITATION_PATTERN.findall(answer))


def validate_citations(answer: str, seen_doc_ids: set[str]) -> list[str]:
    return sorted(extract_cited_doc_ids(answer) - seen_doc_ids)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/agents/tests/test_citations.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/agents/src/agents/citations.py packages/agents/tests/test_citations.py
git commit -m "feat: add citation validator for agentic answers"
```

---

### Task 7: `packages/agents` — pipeline entrypoint with citation retry

**Files:**
- Create: `packages/agents/src/agents/pipeline.py`
- Create: `packages/agents/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `build_initial_messages`, `run_agent_loop` (Task 5); `validate_citations` (Task 6).
- Produces: `MAX_CITATION_RETRIES = 3`; `async def run_agentic_search(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict` returning `{"ok": True, "answer": str, "doc_ids": list[str]}` on success or `{"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": list[str]}` after retries exhaust. Consumed by `retrieval_api/ws.py` (Task 9) and `retrieval_api/retrieval_eval.py` (Task 10).

- [ ] **Step 1: Write the failing tests**

Create `packages/agents/tests/test_pipeline.py`:

```python
import pytest

from agents.pipeline import MAX_CITATION_RETRIES, run_agentic_search


@pytest.mark.asyncio
async def test_pipeline_returns_ok_when_first_answer_is_fully_cited(monkeypatch):
    import agents.pipeline as pipeline_module

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        return {"answer": "See [d1].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)

    result = await run_agentic_search(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}


@pytest.mark.asyncio
async def test_pipeline_retries_on_invalid_citation_then_succeeds(monkeypatch):
    import agents.pipeline as pipeline_module

    calls = {"n": 0}

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"answer": "See [d999].", "seen_doc_ids": {"d1"}, "messages": messages}
        return {"answer": "See [d1].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)

    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await run_agentic_search(
        gateway=object(), es_client=object(), milvus_client=object(), query="q", on_step=on_step,
    )

    assert result == {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}
    assert calls["n"] == 2
    assert steps[0] == ("agent_citation_rejected", {"invalid_doc_ids": ["d999"], "attempt": 1})
    assert steps[1][0] == "agent_answer"


@pytest.mark.asyncio
async def test_pipeline_returns_unverifiable_after_max_retries(monkeypatch):
    import agents.pipeline as pipeline_module

    async def fake_run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=None):
        return {"answer": "See [d999].", "seen_doc_ids": {"d1"}, "messages": messages}

    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)
    calls = {"n": 0}
    original = pipeline_module.run_agent_loop

    async def counting_run_agent_loop(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_agent_loop", counting_run_agent_loop)

    result = await run_agentic_search(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["d999"]}
    assert calls["n"] == MAX_CITATION_RETRIES
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/agents/tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.pipeline'`.

- [ ] **Step 3: Implement**

Create `packages/agents/src/agents/pipeline.py`:

```python
from agents.citations import validate_citations
from agents.loop import build_initial_messages, run_agent_loop

MAX_CITATION_RETRIES = 3


async def run_agentic_search(gateway, es_client, milvus_client, query: str, on_step=None) -> dict:
    messages = build_initial_messages(query)
    seen_doc_ids: set[str] = set()

    for attempt in range(1, MAX_CITATION_RETRIES + 1):
        loop_result = await run_agent_loop(gateway, es_client, milvus_client, messages, seen_doc_ids, on_step=on_step)
        messages = loop_result["messages"]
        seen_doc_ids = loop_result["seen_doc_ids"]
        invalid = validate_citations(loop_result["answer"], seen_doc_ids)

        if not invalid:
            if on_step:
                await on_step("agent_answer", {"answer": loop_result["answer"], "doc_ids": sorted(seen_doc_ids)})
            return {"ok": True, "answer": loop_result["answer"], "doc_ids": sorted(seen_doc_ids)}

        if on_step:
            await on_step("agent_citation_rejected", {"invalid_doc_ids": invalid, "attempt": attempt})

        if attempt == MAX_CITATION_RETRIES:
            return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": invalid}

        messages = messages + [{
            "role": "user",
            "content": (
                f"Your answer cited doc_id(s) not retrieved by any tool call this session: {invalid}. "
                "Revise your answer to cite only doc_ids you actually retrieved."
            ),
        }]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/agents/tests/test_pipeline.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Run the full agents package test suite**

Run: `uv run pytest packages/agents/tests -v`
Expected: PASS (all tests across tools/loop/citations/pipeline).

- [ ] **Step 6: Commit**

```bash
git add packages/agents/src/agents/pipeline.py packages/agents/tests/test_pipeline.py
git commit -m "feat: add agentic search pipeline entrypoint with citation retry"
```

---

### Task 8: retrieval-api — depend on `agents`

**Files:**
- Modify: `packages/retrieval-api/pyproject.toml`

**Interfaces:**
- Consumes: `agents` package (Tasks 4-7).
- Produces: `retrieval_api` can now `import agents`.

- [ ] **Step 1: Add the dependency**

In `packages/retrieval-api/pyproject.toml`, add `"agents"` to `dependencies`:

```toml
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "langchain-core>=0.3",
  "langfuse>=4.14",
  "common",
  "agents",
]
```

- [ ] **Step 2: Re-sync the workspace**

Run: `uv sync --all-packages`
Expected: completes without error.

- [ ] **Step 3: Verify retrieval-api's existing suite is unaffected**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS (no behavior changed yet, just a new importable dependency).

- [ ] **Step 4: Commit**

```bash
git add packages/retrieval-api/pyproject.toml
git commit -m "chore: retrieval-api depends on agents package"
```

---

### Task 9: retrieval-api — `/ws/agent` route

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py`
- Modify: `packages/retrieval-api/tests/test_ws_integration.py`

**Interfaces:**
- Consumes: `agents.pipeline.run_agentic_search` (Task 7); reuses `_emit_trace_step`, `get_gateway_client`, `get_es_client`, `get_milvus_client` already defined in `ws.py`.
- Produces: `WS /ws/agent` — incoming `{"query": str}`; outgoing `{"type": "ai_mode_trace", "step", "data"}` per tool step, then exactly one of `{"type": "agent_done", "answer", "doc_ids"}`, `{"type": "agent_unverifiable", "invalid_doc_ids"}`, or `{"type": "agent_error", "error"}`.

- [ ] **Step 1: Read the existing WS test file to match its exact fixture/mocking style**

Run: `cat packages/retrieval-api/tests/test_ws_integration.py` (already read during planning — reuse the `monkeypatch.setattr(ws_module, ...)` pattern and `TestClient(app).websocket_connect(...)`).

- [ ] **Step 2: Write the failing tests**

Append to `packages/retrieval-api/tests/test_ws_integration.py`:

```python
def test_ws_agent_sends_trace_then_done(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        if on_step:
            import asyncio
            asyncio.get_event_loop()
        return {"ok": True, "answer": "See [d1].", "doc_ids": ["d1"]}

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        done = websocket.receive_json()

    assert done == {"type": "agent_done", "answer": "See [d1].", "doc_ids": ["d1"]}


def test_ws_agent_sends_unverifiable_when_citations_fail(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["d999"]}

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        message = websocket.receive_json()

    assert message == {"type": "agent_unverifiable", "invalid_doc_ids": ["d999"]}


def test_ws_agent_sends_error_on_pipeline_exception(monkeypatch):
    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(ws_module, "run_agentic_search", fake_run_agentic_search)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as websocket:
        websocket.send_json({"query": "gst rate"})
        message = websocket.receive_json()

    assert message == {"type": "agent_error", "error": "RuntimeError: gateway down"}
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v`
Expected: FAIL — `starlette.websockets.WebSocketDisconnect` or 404 (`/ws/agent` doesn't exist yet).

- [ ] **Step 4: Implement**

In `packages/retrieval-api/src/retrieval_api/ws.py`, add the import and the new route (after the existing `search` route):

```python
from agents.pipeline import run_agentic_search
```

```python
@router.websocket("/ws/agent")
async def agent_search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]

    settings = get_settings()
    es_client = get_es_client(settings)
    gateway = get_gateway_client(settings)
    try:
        milvus_client = get_milvus_client(settings)
    except Exception:
        milvus_client = None

    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def emit_trace_step(step: str, data: dict) -> None:
        await _emit_trace_step(send, step, data)

    try:
        result = await run_agentic_search(gateway, es_client, milvus_client, query, on_step=emit_trace_step)
        if result["ok"]:
            await send({"type": "agent_done", "answer": result["answer"], "doc_ids": result["doc_ids"]})
        else:
            await send({"type": "agent_unverifiable", "invalid_doc_ids": result["invalid_doc_ids"]})
    except Exception as exc:
        await send({"type": "agent_error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        await websocket.close()
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v`
Expected: PASS (all tests, including the pre-existing `/ws/search` ones — untouched).

- [ ] **Step 6: Run the full retrieval-api suite**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ws.py packages/retrieval-api/tests/test_ws_integration.py
git commit -m "feat: add /ws/agent route streaming the agentic pipeline"
```

---

### Task 10: retrieval-api — eval CLI agentic mode

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`
- Test: `packages/retrieval-api/tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `agents.pipeline.run_agentic_search` (Task 7).
- Produces: `evaluate_case(...)`'s returned `ranks` dict gains an `"agentic"` key (binary hit: `1` if any gold `doc_id` is in the agent's cited `doc_ids`, else `None`); `errors` dict gains `"agentic"` on pipeline exception or on an `ok: False` unverifiable result; `_print_summary`'s `stages` list gains `"agentic"`.

- [ ] **Step 1: Read the existing eval test file to match its exact fixture style**

Run: `cat packages/retrieval-api/tests/test_retrieval_eval.py`

Match whatever fake `gateway`/`es_client`/`milvus_client` fixtures that file already defines for `evaluate_case` — do not invent new fixture shapes if compatible ones exist.

- [ ] **Step 2: Write the failing tests**

Append to `packages/retrieval-api/tests/test_retrieval_eval.py` (adjust fixture names to match what Step 1 found):

```python
@pytest.mark.asyncio
async def test_evaluate_case_records_agentic_hit_when_gold_doc_cited(monkeypatch):
    import retrieval_api.retrieval_eval as eval_module

    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "See [gold-doc].", "doc_ids": ["gold-doc", "other-doc"]}

    monkeypatch.setattr(eval_module, "run_agentic_search", fake_run_agentic_search)

    case = {
        "id": "Q1", "class": "direct", "query": "q", "gold_doc_ids": ["gold-doc"],
        "expected_collections": ["metadata"], "pass_at": 5,
    }
    result = await eval_module.evaluate_case(
        case, gateway=FAKE_GATEWAY, es_client=FAKE_ES_CLIENT, milvus_client=FAKE_MILVUS_CLIENT, langfuse_enabled=False,
    )

    assert result["ranks"]["agentic"] == 1
    assert "agentic" not in result["errors"]


@pytest.mark.asyncio
async def test_evaluate_case_records_agentic_miss_when_unverifiable(monkeypatch):
    import retrieval_api.retrieval_eval as eval_module

    async def fake_run_agentic_search(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "unverifiable_answer", "invalid_doc_ids": ["bad-doc"]}

    monkeypatch.setattr(eval_module, "run_agentic_search", fake_run_agentic_search)

    case = {
        "id": "Q2", "class": "direct", "query": "q", "gold_doc_ids": ["gold-doc"],
        "expected_collections": ["metadata"], "pass_at": 5,
    }
    result = await eval_module.evaluate_case(
        case, gateway=FAKE_GATEWAY, es_client=FAKE_ES_CLIENT, milvus_client=FAKE_MILVUS_CLIENT, langfuse_enabled=False,
    )

    assert result["ranks"]["agentic"] is None
    assert "unverifiable_answer" in result["errors"]["agentic"]
```

Note: replace `FAKE_GATEWAY`, `FAKE_ES_CLIENT`, `FAKE_MILVUS_CLIENT` with whatever fixtures/fakes Step 1 finds already used by the other `evaluate_case` tests in that file (they must already stub `gateway.embed`, ES, and Milvus calls for the other stages to not error).

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: FAIL — `AttributeError: <module 'retrieval_api.retrieval_eval'> does not have the attribute 'run_agentic_search'`.

- [ ] **Step 4: Implement**

In `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`:

Add the import near the top (with the other `retrieval_api.ai_mode` imports):

```python
from agents.pipeline import run_agentic_search
```

Add a helper function above `evaluate_case`:

```python
def _agentic_hit_rank(doc_ids: list[str] | None, gold: set[str]) -> int | None:
    if not doc_ids:
        return None
    return 1 if gold & set(doc_ids) else None
```

Inside `evaluate_case`, after the existing `reranked = reranked or []` line and before the `ranks = {...}` block, add:

```python
        agentic_result = await measured("agentic", run_agentic_search(gateway, es_client, milvus_client, query))
        agentic_doc_ids = None
        if agentic_result is not None:
            if agentic_result.get("ok"):
                agentic_doc_ids = agentic_result.get("doc_ids")
            else:
                errors["agentic"] = f"unverifiable_answer: {agentic_result.get('invalid_doc_ids')}"
```

Then update the `ranks` dict literal to add one key:

```python
        ranks = {
            "es": doc_rank(es_rows, gold),
            "raw_dense": doc_rank(_flatten(raw_dense), gold),
            "raw_sparse": doc_rank(_flatten(raw_sparse), gold),
            "rewritten_dense": doc_rank(dense_flat, gold),
            "rewritten_sparse": doc_rank(sparse_flat, gold),
            "rrf": doc_rank(merged, gold),
            "reranker": doc_rank(reranked, gold),
            "agentic": _agentic_hit_rank(agentic_doc_ids, gold),
        }
```

Finally, in `_print_summary`, add `"agentic"` to the `stages` list:

```python
    stages = ["es", "raw_dense", "raw_sparse", "rewritten_dense", "rewritten_sparse", "rrf", "reranker", "agentic"]
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full retrieval-api suite**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/retrieval_eval.py packages/retrieval-api/tests/test_retrieval_eval.py
git commit -m "feat: add agentic mode to the retrieval eval CLI"
```

---

### Task 11: web — TracePanel agent step types

**Files:**
- Modify: `packages/web/src/components/TracePanel.tsx`
- Modify: `packages/web/src/components/TracePanel.test.tsx`

**Interfaces:**
- Consumes: trace step literals emitted by `/ws/agent` (Task 9): `agent_tool_call` (`{name, arguments}`), `agent_tool_result` (`{name, result: {rows?, citation?, error?}}`), `agent_citation_rejected` (`{invalid_doc_ids, attempt}`), `agent_answer` (`{answer, doc_ids}`).
- Produces: `TracePanel` renders these four new step types with the same card/summary/body structure as existing steps.

- [ ] **Step 1: Write the failing tests**

Append to `packages/web/src/components/TracePanel.test.tsx` (inside the existing `describe('TracePanel', ...)` block, or add a new `describe` — match whatever the file already does):

```tsx
it('renders an agent_tool_call step with its name and arguments', () => {
  render(<TracePanel steps={[{ step: 'agent_tool_call', data: { name: 'search_es', arguments: { query: 'gst rate' } } }]} />)
  expect(screen.getByText('Agent tool call')).toBeInTheDocument()
  expect(screen.getByText(/search_es/)).toBeInTheDocument()
  expect(screen.getByText(/gst rate/)).toBeInTheDocument()
})

it('renders an agent_tool_result step showing hit count', () => {
  render(<TracePanel steps={[{
    step: 'agent_tool_result',
    data: { name: 'search_es', result: { rows: [{ doc_id: 'd1', score: 1, heading: 'H' }] } },
  }]} />)
  expect(screen.getByText('Agent tool result')).toBeInTheDocument()
  expect(screen.getByText(/1 row/)).toBeInTheDocument()
})

it('renders an agent_tool_result error without crashing', () => {
  render(<TracePanel steps={[{ step: 'agent_tool_result', data: { name: 'search_es', result: { error: 'ES timed out' } } }]} />)
  expect(screen.getByText(/error: ES timed out/)).toBeInTheDocument()
})

it('renders an agent_citation_rejected step with attempt and invalid ids', () => {
  render(<TracePanel steps={[{ step: 'agent_citation_rejected', data: { invalid_doc_ids: ['d999'], attempt: 1 } }]} />)
  expect(screen.getByText('Citation rejected — retrying')).toBeInTheDocument()
  expect(screen.getByText(/attempt 1/)).toBeInTheDocument()
  expect(screen.getByText(/d999/)).toBeInTheDocument()
})

it('renders an agent_answer step with cited doc count', () => {
  render(<TracePanel steps={[{ step: 'agent_answer', data: { answer: 'See [d1].', doc_ids: ['d1'] } }]} />)
  expect(screen.getByText('Agent answer')).toBeInTheDocument()
  expect(screen.getByText(/1 doc/)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/web && npx vitest run src/components/TracePanel.test.tsx`
Expected: FAIL — labels/summaries render as the raw step name or empty string (new cases not handled).

- [ ] **Step 3: Implement**

In `packages/web/src/components/TracePanel.tsx`:

Extend `STEP_LABELS`:

```tsx
const STEP_LABELS: Record<string, string> = {
  intent: 'Intent',
  filters_resolved: 'Filters resolved',
  es_search: 'ES search',
  milvus_dense: 'Milvus dense search',
  milvus_sparse: 'Milvus sparse search',
  rrf_merge: 'RRF merge',
  rerank: 'Rerank',
  synthesis_prompt: 'Synthesis prompt',
  agent_tool_call: 'Agent tool call',
  agent_tool_result: 'Agent tool result',
  agent_citation_rejected: 'Citation rejected — retrying',
  agent_answer: 'Agent answer',
}
```

Extend `summarize()`'s switch, adding cases before `default`:

```tsx
    case 'agent_tool_call':
      return `${d.name}(${JSON.stringify(d.arguments)})`
    case 'agent_tool_result': {
      if (d.result?.error) return `error: ${d.result.error}`
      if (d.result?.citation !== undefined) return d.result.citation ? 'citation found' : 'citation not found'
      const rows = d.result?.rows ?? []
      return `${rows.length} row(s)`
    }
    case 'agent_citation_rejected':
      return `attempt ${d.attempt}: invalid doc_id(s) ${(d.invalid_doc_ids ?? []).join(', ')}`
    case 'agent_answer':
      return `${(d.doc_ids ?? []).length} doc(s) cited`
```

Extend `StepBody()`, adding a case before the final `return null`:

```tsx
  if (step.step === 'agent_tool_result' && d.result?.rows) {
    return <TruncatedHitList hits={d.result.rows} onOpenDocument={onOpenDocument} />
  }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/web && npx vitest run src/components/TracePanel.test.tsx`
Expected: PASS (all tests, including pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/components/TracePanel.tsx packages/web/src/components/TracePanel.test.tsx
git commit -m "feat: render agentic pipeline trace steps in TracePanel"
```

---

### Task 12: web — agent WS config + `useAgentSearch` hook

**Files:**
- Modify: `packages/web/src/lib/config.ts`
- Modify: `packages/web/src/env.d.ts`
- Create: `packages/web/src/api/useAgentSearch.ts`
- Create: `packages/web/src/api/useAgentSearch.test.ts`

**Interfaces:**
- Consumes: `TraceStep` type from `./useSearch`; `/ws/agent` message protocol (Task 9).
- Produces: `resolveAgentWsUrl(): string`; `useAgentSearch(wsUrl: string): AgentSearchState & { search: (query: string) => void }` where `AgentSearchState = { loading: boolean, traceSteps: TraceStep[], result: AgentResult | null, wsError: string | null }` and `AgentResult = { ok: true, answer: string, docIds: string[] } | { ok: false, error: string }`. Consumed by the new `AgentPage` (Task 13).

- [ ] **Step 1: Extend the env type and config helper**

In `packages/web/src/env.d.ts`, widen the inline type:

```ts
declare global {
  interface Window {
    __ENV__?: { WS_URL?: string; AGENT_WS_URL?: string }
  }
}
```

In `packages/web/src/lib/config.ts`, add a new function (leave `resolveWsUrl`/`resolveApiBaseUrl` untouched):

```ts
export function resolveAgentWsUrl(): string {
  const fromEnv = window.__ENV__?.AGENT_WS_URL
  return fromEnv && fromEnv.length > 0 ? fromEnv : 'ws://localhost:8010/ws/agent'
}
```

- [ ] **Step 2: Write the failing hook tests**

Create `packages/web/src/api/useAgentSearch.test.ts`:

```ts
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useAgentSearch } from './useAgentSearch'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  sent: string[] = []
  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }
  addEventListener(type: string, handler: any) {
    if (type === 'open') this.onopen = handler
    if (type === 'message') this.onmessage = handler
    if (type === 'error') this.onerror = handler
    if (type === 'close') this.onclose = handler
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
}

beforeEach(() => {
  FakeWebSocket.instances = []
  // @ts-expect-error test stub
  global.WebSocket = FakeWebSocket
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useAgentSearch', () => {
  it('sends the query and reports the final answer', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))

    act(() => result.current.search('gst rate'))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0])).toEqual({ query: 'gst rate' })

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'ai_mode_trace', step: 'agent_tool_call', data: { name: 'search_es', arguments: {} } }) })
    })
    expect(result.current.traceSteps).toEqual([{ step: 'agent_tool_call', data: { name: 'search_es', arguments: {} } }])

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_done', answer: 'See [d1].', doc_ids: ['d1'] }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result).toEqual({ ok: true, answer: 'See [d1].', docIds: ['d1'] })
  })

  it('reports an unverifiable answer as a non-ok result', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))
    act(() => result.current.search('q'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_unverifiable', invalid_doc_ids: ['d999'] }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result?.ok).toBe(false)
    expect((result.current.result as { ok: false; error: string }).error).toContain('d999')
  })

  it('reports a pipeline error as a non-ok result', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))
    act(() => result.current.search('q'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_error', error: 'RuntimeError: gateway down' }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result).toEqual({ ok: false, error: 'RuntimeError: gateway down' })
  })
})
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd packages/web && npx vitest run src/api/useAgentSearch.test.ts`
Expected: FAIL — `Failed to resolve import "./useAgentSearch"`.

- [ ] **Step 4: Implement**

Create `packages/web/src/api/useAgentSearch.ts`:

```ts
import { useCallback, useRef, useState } from 'react'
import type { TraceStep } from './useSearch'

export type AgentResult =
  | { ok: true; answer: string; docIds: string[] }
  | { ok: false; error: string }

export interface AgentSearchState {
  loading: boolean
  traceSteps: TraceStep[]
  result: AgentResult | null
  wsError: string | null
}

const INITIAL_STATE: AgentSearchState = { loading: false, traceSteps: [], result: null, wsError: null }

export function useAgentSearch(wsUrl: string): AgentSearchState & { search: (query: string) => void } {
  const [state, setState] = useState<AgentSearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (query: string) => {
      socketRef.current?.close()
      setState({ loading: true, traceSteps: [], result: null, wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ query }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'ai_mode_trace') {
          setState((prev) => ({ ...prev, traceSteps: [...prev.traceSteps, { step: message.step, data: message.data }] }))
        } else if (message.type === 'agent_done') {
          setState((prev) => ({ ...prev, loading: false, result: { ok: true, answer: message.answer, docIds: message.doc_ids ?? [] } }))
        } else if (message.type === 'agent_unverifiable') {
          setState((prev) => ({
            ...prev,
            loading: false,
            result: {
              ok: false,
              error: `Could not produce a fully cited answer. Invalid doc_id(s): ${(message.invalid_doc_ids ?? []).join(', ')}`,
            },
          }))
        } else if (message.type === 'agent_error') {
          setState((prev) => ({ ...prev, loading: false, result: { ok: false, error: message.error } }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the agent service failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.loading ? { ...prev, loading: false } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, search }
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd packages/web && npx vitest run src/api/useAgentSearch.test.ts`
Expected: PASS (all 3 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/web/src/lib/config.ts packages/web/src/env.d.ts packages/web/src/api/useAgentSearch.ts packages/web/src/api/useAgentSearch.test.ts
git commit -m "feat: add useAgentSearch hook and agent WS config"
```

---

### Task 13: web — standalone `/agent` page + nav link

**Files:**
- Create: `packages/web/src/AgentPage.tsx`
- Create: `packages/web/src/AgentPage.test.tsx`
- Modify: `packages/web/src/main.tsx`
- Modify: `packages/web/src/App.tsx`

**Interfaces:**
- Consumes: `useAgentSearch` (Task 12), `TracePanel` (Task 11), `SearchBar` (existing component, same as `DebugPage` uses).
- Produces: route `/agent` rendering `AgentPage`; a nav link from the main page to `/agent`.

- [ ] **Step 1: Write the failing test**

Create `packages/web/src/AgentPage.test.tsx`, mirroring `DebugPage.test.tsx`'s structure:

```tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import AgentPage from './AgentPage'
import { useAgentSearch } from './api/useAgentSearch'

vi.mock('./api/useAgentSearch', () => ({
  useAgentSearch: vi.fn(() => ({
    loading: false,
    traceSteps: [],
    result: null,
    wsError: null,
    search: vi.fn(),
  })),
}))

describe('AgentPage', () => {
  it('renders a heading, a link back to search, and the trace panel placeholder', () => {
    render(<AgentPage />, { wrapper: MemoryRouter })
    expect(screen.getByText(/Agentic Search/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /back to search/i })).toBeInTheDocument()
    expect(screen.getByText(/no trace yet/i)).toBeInTheDocument()
  })

  it('calls search with the typed query', async () => {
    const search = vi.fn()
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: null, wsError: null, search,
    })
    const { default: userEvent } = await import('@testing-library/user-event')
    const user = userEvent.setup()
    render(<AgentPage />, { wrapper: MemoryRouter })

    await user.type(screen.getByRole('textbox'), 'gst rate{enter}')

    expect(search).toHaveBeenCalledWith('gst rate')
  })

  it('renders a successful cited answer with its doc_ids', () => {
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: { ok: true, answer: 'See [d1].', docIds: ['d1'] }, wsError: null, search: vi.fn(),
    })
    render(<AgentPage />, { wrapper: MemoryRouter })

    expect(screen.getByText(/See \[d1\]\./)).toBeInTheDocument()
    expect(screen.getByText(/d1/)).toBeInTheDocument()
  })

  it('renders an unverifiable/error result distinctly from a successful answer', () => {
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: { ok: false, error: 'Could not produce a fully cited answer.' }, wsError: null, search: vi.fn(),
    })
    render(<AgentPage />, { wrapper: MemoryRouter })

    expect(screen.getByText(/Could not produce a fully cited answer\./)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd packages/web && npx vitest run src/AgentPage.test.tsx`
Expected: FAIL — `Failed to resolve import "./AgentPage"`.

- [ ] **Step 3: Implement the page**

Create `packages/web/src/AgentPage.tsx`:

```tsx
// src/AgentPage.tsx
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import TracePanel from './components/TracePanel'
import { useAgentSearch } from './api/useAgentSearch'
import { resolveAgentWsUrl } from './lib/config'
import styles from './App.module.css'

export default function AgentPage() {
  const wsUrl = resolveAgentWsUrl()
  const { traceSteps, loading, result, wsError, search } = useAgentSearch(wsUrl)

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Agentic Search — tool-calling agent, cited answers only</h1>
        <Link to="/">Back to search</Link>
      </header>
      <SearchBar onSearch={(query) => search(query)} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      {result && result.ok && (
        <section>
          <p>{result.answer}</p>
          <p>Cited: {result.docIds.join(', ')}</p>
        </section>
      )}
      {result && !result.ok && <p className={styles.wsError}>{result.error}</p>}
      <TracePanel steps={traceSteps} />
    </div>
  )
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd packages/web && npx vitest run src/AgentPage.test.tsx`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Wire the route and nav link**

In `packages/web/src/main.tsx`:

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import DebugPage from './DebugPage'
import AgentPage from './AgentPage'

const container = document.getElementById('root')
if (!container) {
  throw new Error('Root container #root not found')
}

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/debug" element={<DebugPage />} />
        <Route path="/agent" element={<AgentPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
```

In `packages/web/src/App.tsx`, add a nav link next to the existing `/debug` link inside `.headerActions`:

```tsx
<div className={styles.headerActions}>
  <Link to="/debug">Retrieval debug</Link>
  <Link to="/agent">Agentic search</Link>
  <DevModeToggle devMode={devMode} onToggle={setDevMode} />
</div>
```

- [ ] **Step 6: Run the full web test suite**

Run: `cd packages/web && npx vitest run`
Expected: PASS (all tests, including `App.test.tsx` — check it doesn't assert an exact link count/list that the new link would break; if it does, add the new link to that assertion).

- [ ] **Step 7: Commit**

```bash
git add packages/web/src/AgentPage.tsx packages/web/src/AgentPage.test.tsx packages/web/src/main.tsx packages/web/src/App.tsx
git commit -m "feat: add standalone /agent page for the agentic search pipeline"
```

---

## Post-plan verification

- [ ] Run `uv sync --all-packages` from repo root, then `uv run pytest` — expect all backend tests (previous suite + new agents/model-gateway/retrieval-api tests) passing.
- [ ] Run `cd packages/web && npx vitest run` — expect all frontend tests passing.
- [ ] Manually run `retrieval-eval --query <some-id>` against a live stack and confirm the summary table now prints an `agentic` column alongside `es`/`rrf`/`reranker`.
- [ ] Manually open `/agent` in the browser, run a query, and confirm: live trace steps stream in, a final cited answer or an explicit unverifiable/error message appears, and the doc_ids shown as "Cited" match what appeared in the trace's tool results.
