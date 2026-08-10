# Intent Extraction Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AI Mode's `slm` extraction stage (`intent.py`) produce schema-correct, provider-enforced JSON instead of prompt-only free text, extract the full set of ES-filterable fields, and validate a `google/gemma-4-E4B-it` model swap — all without touching retrieval (RRF weighting, collection routing, or any other `intent`-consuming behavior stays out of scope; see design spec's Phasing note).

**Architecture:** Plumb an optional `response_format` parameter through the existing `retrieval-api` → `model-gateway` → DeepInfra chat call chain (mirrors the existing `model` override plumbing already present at every layer). `intent.py` passes `{"type": "json_object"}` and drops its regex-based brace-extraction fallback — DeepInfra guarantees the response is valid JSON, so the existing content-level guards (`_safe_rewrite`, `_sanitize_filters`) become the only remaining layer of defense, exactly as designed. Filters gain `bench`/`judge`; `intent` gains a fixed 4-value enum (classified, still unconsumed downstream). A new small eval dataset with hand-authored gold filter expectations plus a cheap prompt-only CLI checker (no ES/Milvus) validates both the mechanism and the model candidate.

**Tech Stack:** Python 3.11, FastAPI (`model-gateway`), httpx/respx for adapter tests, pytest-asyncio.

## Global Constraints

- Python 3.11, not 3.14 (`pymilvus`'s `grpcio` has no 3.14 wheel).
- No retrieval-side changes in this plan: no RRF weighting, no `intent` consumer, no full `retrieval_eval.py` run. (Design spec, Phasing/Non-goals.)
- No intent-based Milvus collection routing, ever (CLAUDE.md hard rule — not touched by this plan, called out here because a future reader of `intent.py` could otherwise be tempted).
- `query_embed` role stays Voyage-only — not touched by this plan.
- `_safe_rewrite`'s and `_sanitize_filters`'s anti-hallucination logic is unchanged — schema mode guarantees JSON *shape*, not truthfulness.
- Run `uv run pytest` from repo root after each task; it aggregates all packages.

---

### Task 1: `response_format` plumbing (model-gateway adapters, routes, GatewayClient)

**Files:**
- Modify: `packages/model-gateway/src/model_gateway/adapters/base.py`
- Modify: `packages/model-gateway/src/model_gateway/adapters/deepinfra.py`
- Modify: `packages/model-gateway/src/model_gateway/adapters/voyage.py`
- Modify: `packages/model-gateway/src/model_gateway/routes.py`
- Modify: `packages/retrieval-api/src/retrieval_api/gateway_client.py`
- Test: `packages/model-gateway/tests/test_deepinfra_adapter.py`
- Test: `packages/model-gateway/tests/test_routes.py`
- Test: `packages/retrieval-api/tests/test_gateway_client.py`

**Interfaces:**
- Produces: `DeepInfraAdapter.chat(model, messages, tools=None, tool_choice=None, response_format=None)` — unchanged return type `tuple[str | None, dict[str, int], str | None, list[dict] | None]`.
- Produces: `GatewayClient.chat(role, messages, model=None, response_format=None) -> str` and `GatewayClient.chat_with_reasoning(role, messages, model=None, response_format=None) -> tuple[str, str | None]` — Task 2 (`intent.py`) calls these with `response_format={"type": "json_object"}`.

- [ ] **Step 1: Write the failing adapter tests**

Add to `packages/model-gateway/tests/test_deepinfra_adapter.py`:

```python
@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_response_format_when_given():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    await adapter.chat(
        "some-model", [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_response_format_key_when_not_given():
    route = respx.post("https://api.deepinfra.com/v1/openai/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    )
    adapter = DeepInfraAdapter(api_key="k")

    await adapter.chat("some-model", [{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "response_format" not in sent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/model-gateway/tests/test_deepinfra_adapter.py -v`
Expected: the two new tests FAIL with `TypeError: chat() got an unexpected keyword argument 'response_format'`.

- [ ] **Step 3: Implement adapter/protocol changes**

In `packages/model-gateway/src/model_gateway/adapters/base.py`, change the `chat` signature:

```python
class ModelAdapter(Protocol):
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None,
        tool_choice: str | None = None, response_format: dict | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]: ...
    async def embed(self, model: str, text: str) -> list[float]: ...
    async def rerank(self, model: str, query: str, documents: list[str]) -> list[float]: ...
```

In `packages/model-gateway/src/model_gateway/adapters/deepinfra.py`, change `chat`:

```python
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None,
        tool_choice: str | None = None, response_format: dict | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]:
        payload = {"model": model, "messages": messages, "max_tokens": _CHAT_MAX_TOKENS}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format
        async with httpx.AsyncClient(timeout=60.0) as client:
```
(rest of the method body is unchanged)

In `packages/model-gateway/src/model_gateway/adapters/voyage.py`, update the stub signature to match the protocol (still raises):

```python
    async def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None,
        tool_choice: str | None = None, response_format: dict | None = None,
    ) -> tuple[str | None, dict[str, int], str | None, list[dict] | None]:
        raise NotImplementedError("VoyageAdapter does not support chat")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/model-gateway/tests/test_deepinfra_adapter.py -v`
Expected: PASS (all tests including the two new ones).

- [ ] **Step 5: Write the failing routes test, update the existing assertion**

In `packages/model-gateway/tests/test_routes.py`, update the existing assertion (it currently asserts a 2-positional-arg call, which will break once a 3rd optional arg is threaded through) and add a new test:

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
    fake_adapter.chat.assert_awaited_once_with("big-model", [{"role": "user", "content": "hi"}], None, None, None)


def test_chat_route_forwards_response_format_to_adapter(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("{}", {}, None, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "small-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    client.post("/v1/chat", json={
        "role": "slm", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })

    fake_adapter.chat.assert_awaited_once_with(
        "small-model", [{"role": "user", "content": "hi"}], None, None, {"type": "json_object"},
    )
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest packages/model-gateway/tests/test_routes.py -v`
Expected: both tests FAIL — the existing one now expects 5 positional args but the route only passes 4; the new one fails on the same mismatch plus a `response_format` field the `ChatRequest` model doesn't know about (extra fields are ignored by default in pydantic, so this one just fails on the assert).

- [ ] **Step 7: Implement the route change**

In `packages/model-gateway/src/model_gateway/routes.py`, add the field to `ChatRequest` and pass it through:

```python
class ChatRequest(BaseModel):
    role: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | None = None
    model: str | None = None
    response_format: dict | None = None
```

```python
@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    default_model, provider = _resolve(req.role)
    model = req.model or default_model
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
            model, req.messages, req.tools, req.tool_choice, req.response_format,
        )
        generation.update(output=content if content is not None else {"tool_calls": tool_calls}, usage_details=usage_details)
        if reasoning:
            generation.update(metadata={"reasoning": reasoning})
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest packages/model-gateway/tests/test_routes.py -v`
Expected: PASS.

- [ ] **Step 9: Write the failing GatewayClient test**

Add to `packages/retrieval-api/tests/test_gateway_client.py`:

```python
@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_response_format_when_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "{}"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(
        role="slm", messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )

    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_response_format_key_when_not_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "response_format" not in sent
```

- [ ] **Step 10: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: FAIL with `TypeError: chat() got an unexpected keyword argument 'response_format'`.

- [ ] **Step 11: Implement the GatewayClient change**

In `packages/retrieval-api/src/retrieval_api/gateway_client.py`:

```python
    async def chat(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        content, _reasoning = await self.chat_with_reasoning(role, messages, model=model, response_format=response_format)
        return content

    async def chat_with_reasoning(
        self, role: str, messages: list[dict], model: str | None = None,
        response_format: dict | None = None,
    ) -> tuple[str, str | None]:
        body = {"role": role, "messages": messages}
        if model is not None:
            body["model"] = model
        if response_format is not None:
            body["response_format"] = response_format
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat", json=body, headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data["content"], data.get("reasoning")
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: PASS.

- [ ] **Step 13: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests pass (existing count + 6 new tests across the three files).

```bash
git add packages/model-gateway/src/model_gateway/adapters/base.py \
        packages/model-gateway/src/model_gateway/adapters/deepinfra.py \
        packages/model-gateway/src/model_gateway/adapters/voyage.py \
        packages/model-gateway/src/model_gateway/routes.py \
        packages/retrieval-api/src/retrieval_api/gateway_client.py \
        packages/model-gateway/tests/test_deepinfra_adapter.py \
        packages/model-gateway/tests/test_routes.py \
        packages/retrieval-api/tests/test_gateway_client.py
git commit -m "feat: plumb response_format through model-gateway chat path"
```

---

### Task 2: Schema-enforced extraction + filter/intent expansion in `intent.py`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Modify: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Consumes: `GatewayClient.chat(role, messages, model=None, response_format=None)` from Task 1.
- Produces: `extract_intent(gateway, query, on_step=None, model=None) -> dict` — same signature as before. Return shape's `filters` dict may now additionally contain `"bench"`/`"judge"` keys; `intent` is now always one of `{"citation_lookup", "provision_lookup", "conceptual", "unknown"}`.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `packages/retrieval-api/tests/test_ai_mode_intent.py` with:

```python
import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.intent import extract_intent


@pytest.mark.asyncio
async def test_extract_intent_parses_json_object_response():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "BNS section 103 murder punishment",
        "intent": "provision_lookup",
        "filters": {"act": "BNS"},
    })

    result = await extract_intent(gateway, "IPC 302 punishment")

    assert result == {
        "rewritten_query": "IPC 302 punishment",
        "intent": "provision_lookup",
        "filters": {},
    }
    gateway.chat.assert_awaited_once()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_requests_json_object_response_format():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_response_is_wrapped_in_markdown_fence():
    """json_object response_format should prevent this from a compliant model,
    but if a model still wraps its output, fall back rather than regex-guessing
    the JSON out of prose - that guesswork is exactly what schema mode replaces."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = "```json\n" + json.dumps({
        "rewritten_query": "capital gains set off against carried forward business losses",
        "intent": "conceptual",
        "filters": {},
    }) + "\n```"

    result = await extract_intent(gateway, "set off capital gains against brought forward business losses")

    assert result == {
        "rewritten_query": "set off capital gains against brought forward business losses",
        "intent": "unknown",
        "filters": {},
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_to_plain_search_on_unparseable_response():
    """Covers SLM refusals too (e.g. Llama declining a named-party query) -
    AI Mode should degrade to plain semantic search, not fail outright."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = "I cannot provide case law for that person."

    result = await extract_intent(gateway, "some query")

    assert result == {"rewritten_query": "some query", "intent": "unknown", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_system_prompt_includes_schema_context_and_new_fields():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "q", "intent": "conceptual", "filters": {},
    })

    await extract_intent(gateway, "some query")

    system_message = gateway.chat.await_args.kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "facts" in system_message["content"]
    assert "Supreme Court" in system_message["content"]
    assert '"section"' in system_message["content"]
    assert '"gte"' in system_message["content"]
    assert '"lte"' in system_message["content"]
    assert '"bench"' in system_message["content"]
    assert '"judge"' in system_message["content"]
    assert "citation_lookup" in system_message["content"]
    assert "provision_lookup" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_emits_intent_step_when_on_step_given():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "original query normalized",
        "intent": "conceptual",
        "filters": {"act": "CGST Act"},
    })
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await extract_intent(gateway, "original query", on_step=on_step)

    assert result == {"rewritten_query": "original query normalized", "intent": "conceptual", "filters": {}}
    assert steps == [("intent", {
        "query": "original query",
        "rewritten_query": "original query normalized",
        "intent": "conceptual",
        "filters": {},
    })]


@pytest.mark.asyncio
async def test_extract_intent_skips_on_step_when_none():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    result = await extract_intent(gateway, "q")  # no on_step passed

    assert result == {"rewritten_query": "q", "intent": "conceptual", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_act_and_preserves_legal_identifier():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "case law for Bharatiya Nyaya Sanhita about scrap sale",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "80HH scrap sale yes useless drum sale no")

    assert result["rewritten_query"] == "80HH scrap sale yes useless drum sale no"


@pytest.mark.asyncio
async def test_extract_intent_rejects_expansion_of_ambiguous_acronym():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "software royalty Profit and Excess India USA DTAA",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "software royalty PE India USA DTAA")

    assert result["rewritten_query"] == "software royalty PE India USA DTAA"


@pytest.mark.asyncio
async def test_extract_intent_drops_unknown_null_and_empty_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "trade training takeover Kolkata",
        "intent": "conceptual",
        "filters": {"city": "Kolkata", "act": None, "court": "", "section": "37(1)"},
    })

    result = await extract_intent(gateway, "trade training takeover Kolkata section 37(1)")

    assert result["filters"] == {"section": "37(1)"}


@pytest.mark.asyncio
async def test_extract_intent_extracts_bench_and_judge_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
        "intent": "conceptual",
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    })

    result = await extract_intent(
        gateway,
        "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
    )

    assert result["filters"] == {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"}


@pytest.mark.asyncio
async def test_extract_intent_drops_bench_and_judge_when_not_literally_in_query():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "Modvat credit ruling",
        "intent": "conceptual",
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    })

    result = await extract_intent(gateway, "Modvat credit ruling")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_coerces_unrecognized_intent_label_to_unknown():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "q", "intent": "section_lookup", "filters": {},
    })

    result = await extract_intent(gateway, "q")

    assert result["intent"] == "unknown"


@pytest.mark.asyncio
async def test_extract_intent_accepts_each_allowed_intent_label():
    for label in ["citation_lookup", "provision_lookup", "conceptual", "unknown"]:
        gateway = AsyncMock()
        gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": label, "filters": {}})

        result = await extract_intent(gateway, "q")

        assert result["intent"] == label


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_shape_is_invalid():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": None,
        "intent": ["tax"],
        "filters": "none",
    })

    result = await extract_intent(gateway, "original")

    assert result == {"rewritten_query": "original", "intent": "unknown", "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_year_and_court():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax deduction in 2024 decided by Delhi High Court",
        "intent": "conceptual",
        "filters": {},
    })

    result = await extract_intent(gateway, "income tax deduction")

    assert result["rewritten_query"] == "income tax deduction"


@pytest.mark.asyncio
async def test_extract_intent_preserves_user_supplied_year_and_section_numbers():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax section 80HH deduction in 1985",
        "intent": "provision_lookup",
        "filters": {"section": "80HH"},
    })

    result = await extract_intent(gateway, "1985 income tax section 80HH deduction")

    assert result["rewritten_query"] == "income tax section 80HH deduction in 1985"
    assert result["filters"] == {"section": "80HH"}


@pytest.mark.asyncio
async def test_extract_intent_rejects_lossy_rewrite():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "sale of scrap under 80HH",
        "intent": "conceptual",
        "filters": {},
    })

    query = "80HH scrap sale yes useless drum sale no metallic wire factory"
    result = await extract_intent(gateway, query)

    assert result["rewritten_query"] == query


@pytest.mark.asyncio
async def test_extract_intent_requests_model_for_slm_role():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    gateway.get_model.assert_awaited_once_with(role="slm")


@pytest.mark.asyncio
async def test_extract_intent_uses_llama_tuned_prompt_for_llama_model():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    system_message = gateway.chat.await_args.kwargs["messages"][0]
    assert "Forbidden rewrites" in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_warns_when_model_has_no_tuned_prompt(monkeypatch):
    import retrieval_api.ai_mode.intent as intent_module

    captured = {}
    monkeypatch.setattr(intent_module, "get_client", lambda: type(
        "FakeLangfuseClient", (), {"update_current_span": staticmethod(lambda **kw: captured.update(kw))},
    )())
    gateway = AsyncMock()
    gateway.get_model.return_value = "some-brand-new-model"
    gateway.chat.return_value = json.dumps({"rewritten_query": "q", "intent": "conceptual", "filters": {}})

    await extract_intent(gateway, "q")

    assert captured.get("level") == "WARNING"


@pytest.mark.asyncio
async def test_extract_intent_drops_non_iso_or_invented_date_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "income tax cases",
        "intent": "conceptual",
        "filters": {"date_range": {"gte": "not specified", "lte": "2024-12-31"}},
    })

    result = await extract_intent(gateway, "income tax cases")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_forwards_model_override_and_skips_get_model():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "candidate model test",
        "intent": "conceptual",
        "filters": {},
    })

    await extract_intent(gateway, "candidate model test", model="google/gemma-4-E4B-it")

    gateway.get_model.assert_not_awaited()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["model"] == "google/gemma-4-E4B-it"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: FAIL — several tests fail because `intent` values like `"section_lookup"` no longer pass through unchanged (not yet coerced), `bench`/`judge` aren't in `_ALLOWED_FILTERS` yet, `response_format` isn't sent, and the markdown-fence test fails because the regex extraction still rescues it.

- [ ] **Step 3: Implement the changes in `intent.py`**

Replace the full contents of `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`:

```python
import json
from typing import Awaitable, Callable

from langfuse import get_client

from common.schema_context import KNOWN_COURTS, build_schema_context
from retrieval_api.gateway_client import GatewayClient

# Invariant: on_step implementations must not raise. The current only caller
# (ws.py's emit_trace_step / _emit_trace_step) guarantees this by swallowing
# any exception from sending a trace frame. A future caller that passes a
# raising callback would have that exception propagate into run_ai_mode's
# blanket `except Exception`, incorrectly turning a successful pipeline run
# into an ai_mode_error.
OnStep = Callable[[str, dict], Awaitable[None]]

# DeepInfra's json_object response_format mode guarantees the response is a
# valid JSON object with no surrounding prose/markdown fence - see
# https://docs.deepinfra.com/chat/structured-outputs. This replaces a former
# regex-based brace-extraction fallback: if a model still doesn't comply
# despite the mode being requested, that's treated as a hard failure
# (json.loads raises, _fallback_intent kicks in) rather than guessed at.
_RESPONSE_FORMAT = {"type": "json_object"}


def _fallback_intent(query: str) -> dict:
    """Used when the SLM refuses or returns unparseable output (e.g. Llama's
    safety training treating "case law for X vs. Y" as a request for private
    info about a named person) - degrade to a plain semantic search instead
    of failing the whole AI Mode request."""
    return {"rewritten_query": query, "intent": "unknown", "filters": {}}


_LLAMA_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": a CONSERVATIVE search normalization. Correct obvious
  spelling and grammar only. Preserve every party, court, place, Act,
  section, rule, notification, date, number, citation, and acronym exactly
  as written. NEVER add or infer a legal concept. NEVER expand an acronym
  (for example PE, ST, CA, ITD, PTA, MEG, POY, or PSF). NEVER translate an
  old law to a new law or replace one section with another. If the query is
  already readable, copy it unchanged. Every number and year in the output
  must occur in the input; if the input has no year, add no year.
- "intent": exactly one of "citation_lookup" (the query is anchored on a
  party name or case citation), "provision_lookup" (anchored on a
  section/act/rule number), "conceptual" (an open legal question with no
  strong lexical anchor), or "unknown" (none of the above fit confidently).
  Never output any other value.
- "filters": an object with any of "court", "act", "section", "date_range",
  "party", "bench", "judge" - ONLY include a key if its value is LITERALLY
  written in the query. Never guess, infer, or fill in a plausible-sounding
  court, act, section, bench, judge, or date range that the query does not
  state - a wrong filter silently excludes the correct document from the
  search entirely, which is worse than no filter. If the query names a
  person or company (very often written as "X vs. Y" or "X v. Y"), put that
  name under "party" - never under "section" or any other key. If nothing
  is explicitly stated, "filters" should be an empty object. Never output
  null or empty filter values. Never output any other filter key such as
  city, state, topic, or citation. "date_range" MUST be an object with ISO
  date strings, e.g. {"gte": "2020-01-01", "lte": "2022-01-01"} - either key
  may be omitted, but never output "date_range" as a plain string or year
  number, and never invent one when no date was mentioned.

Example: query "case law for Ramesh Gupta vs. Income-tax Officer" mentions
no court, act, section, or date - only a party name - so filters must be
exactly {"party": "Ramesh Gupta"} and intent is "citation_lookup".

Forbidden rewrites:
- "80HH scrap sale" must not mention BNS or any other Act.
- "software royalty PE" must retain "PE" without guessing its expansion.
- "69C diamond cash sale" must not add CGST Act or replace section 69C.
- "59/98-ST certification" must not add Customs Act.

""" + build_schema_context()


def _system_prompt_for_model(model: str) -> str:
    """Different models need different prompt shapes to follow instructions
    reliably (see docs/superpowers/specs/2026-08-06-agentic-search-pipeline-design.md's
    note on agent_chat) - the Llama-tuned prompt above was written and
    eval-validated against Llama-3.1-8B-Instruct's specific tendency to
    over-generalize open-ended rewrite instructions. Fall back to it for any
    other model too, but surface a warning so a future model swap doesn't
    silently inherit a prompt shape nobody has tuned or evaluated for it."""
    if "llama" in model.lower():
        return _LLAMA_SYSTEM_PROMPT
    get_client().update_current_span(
        level="WARNING",
        status_message=f"No prompt shape has been tuned/evaluated for model {model!r} - "
                        "falling back to the Llama-tuned prompt, which may not fit its "
                        "instruction-following style.",
    )
    return _LLAMA_SYSTEM_PROMPT


_ALLOWED_FILTERS = {"court", "act", "section", "date_range", "party", "bench", "judge"}
_ALLOWED_INTENTS = {"citation_lookup", "provision_lookup", "conceptual", "unknown"}
_LEGAL_MARKERS = {
    "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
    "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
    "income tax act", "cgst act", "igst act", "customs act",
    "code of criminal procedure", "indian evidence act",
}


def _protected_identifiers(text: str) -> set[str]:
    import re
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9()/-]*\b", text)
    return {
        token.upper() for token in tokens
        if (token.isupper() and len(token) >= 2)
        or (any(c.isupper() for c in token) and any(c.isdigit() for c in token))
    }


def _safe_rewrite(query: str, rewritten: str) -> str:
    import re
    query_lower, rewritten_lower = query.casefold(), rewritten.casefold()
    if any(marker in rewritten_lower and marker not in query_lower for marker in _LEGAL_MARKERS):
        return query
    if any(court.casefold() in rewritten_lower and court.casefold() not in query_lower for court in KNOWN_COURTS):
        return query
    if set(re.findall(r"\d+", query)) != set(re.findall(r"\d+", rewritten)):
        return query
    if not _protected_identifiers(query).issubset(_protected_identifiers(rewritten)):
        return query
    query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
    rewritten_tokens = set(re.findall(r"[a-z0-9]+", rewritten_lower))
    if query_tokens and len(query_tokens & rewritten_tokens) / len(query_tokens) < 0.6:
        return query
    return rewritten


def _sanitize_filters(query: str, filters) -> dict:
    import re
    if not isinstance(filters, dict):
        return {}
    clean = {}
    for key, value in filters.items():
        if key not in _ALLOWED_FILTERS:
            continue
        if key == "date_range":
            if isinstance(value, dict):
                date_range = {
                    bound: date for bound, date in value.items()
                    if bound in {"gte", "lte"}
                    and isinstance(date, str)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                    and date[:4] in query
                }
                if date_range:
                    clean[key] = date_range
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if value.casefold() not in query.casefold():
            continue
        clean[key] = value
    return clean


def _validate_result(query: str, result) -> dict:
    if not isinstance(result, dict):
        return _fallback_intent(query)
    rewritten, intent = result.get("rewritten_query"), result.get("intent")
    if not isinstance(rewritten, str) or not rewritten.strip() or not isinstance(intent, str):
        return _fallback_intent(query)
    return {
        "rewritten_query": _safe_rewrite(query, rewritten.strip()),
        "intent": intent if intent in _ALLOWED_INTENTS else "unknown",
        "filters": _sanitize_filters(query, result.get("filters")),
    }


async def extract_intent(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, model: str | None = None,
) -> dict:
    resolved_model = model or await gateway.get_model(role="slm")
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _system_prompt_for_model(resolved_model)},
            {"role": "user", "content": query},
        ],
        model=model,
        response_format=_RESPONSE_FORMAT,
    )
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        get_client().update_current_span(
            level="WARNING", status_message=f"SLM did not return valid JSON, falling back to plain search: {response!r}",
        )
        result = _fallback_intent(query)
    else:
        result = _validate_result(query, result)

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
```

Note: `re` is now imported locally inside `_protected_identifiers`, `_safe_rewrite`, and `_sanitize_filters` instead of at module scope, since the top-level `_extract_json_object` (the only other regex user) is deleted. Equivalently valid to keep `import re` at module scope — either is fine; local imports shown here just make each function's dependency explicit given the module has fewer regex users now. If your editor/linter prefers module-scope imports, put `import re` back at the top instead — behavior is identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: PASS (all tests, including the new bench/judge and intent-enum-coercion tests).

- [ ] **Step 5: Check other callers still pass**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS. `test_ai_mode_pipeline.py` and `test_retrieval_eval.py` both call `extract_intent` — confirm their mocked `gateway.chat.return_value`s are still valid JSON strings (they are; `extract_intent`'s public signature is unchanged) and that any hardcoded `"intent"` value they assert on is one of the 4 allowed labels or intentionally testing fallback-to-`"unknown"`. If either file asserts on an old-style free-text intent label (e.g. `"taxation"`), update it to `"conceptual"` (or whichever label fits the test's actual intent) the same way Step 1 did for `test_ai_mode_intent.py`.

- [ ] **Step 6: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests pass.

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat: schema-enforced JSON extraction, bench/judge filters, 4-way intent enum"
```

---

### Task 3: `bench`/`judge` filter fields in `es_client.py` and `schema_context.py`

**Files:**
- Modify: `packages/common/src/common/schema_context.py`
- Modify: `packages/common/src/common/es_client.py`
- Modify: `packages/common/tests/test_schema_context.py`
- Modify: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Produces: `KNOWN_FILTER_FIELDS` now includes `"bench"`, `"judge"`.
- Produces: `_TERM_FILTER_FIELDS` (used by `resolve_doc_id_allowlist`) now includes `bench` → `masterinfo.info.bench`, `judge` → `otherinfo.judge` — these are the exact field paths already used for citation display in `fetch_citations`/`MASTERINFO_CITATION_FIELDS`, so no new ES mapping assumption is introduced.

- [ ] **Step 1: Write the failing schema_context test**

Add to `packages/common/tests/test_schema_context.py`:

```python
def test_build_schema_context_lists_bench_and_judge_filter_fields():
    context = build_schema_context()

    assert "bench" in context
    assert "judge" in context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_schema_context.py -v`
Expected: FAIL — `bench`/`judge` aren't in `KNOWN_FILTER_FIELDS` yet.

- [ ] **Step 3: Implement the schema_context change**

In `packages/common/src/common/schema_context.py`, change:

```python
KNOWN_FILTER_FIELDS = ["court", "act", "section", "party", "date_range"]
```

to:

```python
KNOWN_FILTER_FIELDS = ["court", "act", "section", "party", "date_range", "bench", "judge"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_schema_context.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing es_client test**

Add to `packages/common/tests/test_es_client.py` (reuse the `FakeAsyncES` class already defined in this file):

```python
@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_masterinfo_bench_term():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"bench": "Principal Bench"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"masterinfo.info.bench.keyword": "Principal Bench"}}]}
    }


@pytest.mark.asyncio
async def test_resolve_doc_id_allowlist_queries_otherinfo_judge_term():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}}])

    result = await resolve_doc_id_allowlist(client, {"judge": "D.Y. Chandrachud"})

    assert result == ["d1"]
    assert client.search_calls[0] == {
        "bool": {"must": [{"term": {"otherinfo.judge.keyword": "D.Y. Chandrachud"}}]}
    }
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_es_client.py -v`
Expected: the two new tests FAIL with `ValueError: No recognized filter keys in {...}` (since `bench`/`judge` aren't in `_TERM_FILTER_FIELDS` yet, `_build_filter_must` returns an empty list and `resolve_doc_id_allowlist` raises).

- [ ] **Step 7: Implement the es_client change**

In `packages/common/src/common/es_client.py`, change:

```python
_TERM_FILTER_FIELDS = {
    "court": "masterinfo.info.court.name",
    "act": "masterinfo.info.act.name",
    "section": "masterinfo.info.section.name",
}
```

to:

```python
_TERM_FILTER_FIELDS = {
    "court": "masterinfo.info.court.name",
    "act": "masterinfo.info.act.name",
    "section": "masterinfo.info.section.name",
    "bench": "masterinfo.info.bench",
    "judge": "otherinfo.judge",
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_es_client.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests pass.

```bash
git add packages/common/src/common/schema_context.py packages/common/src/common/es_client.py \
        packages/common/tests/test_schema_context.py packages/common/tests/test_es_client.py
git commit -m "feat: add bench/judge filter fields to schema context and ES term filters"
```

---

### Task 4: Gold-filter eval dataset + prompt-only checker

**Files:**
- Create: `evals/intent_filter_cases.json`
- Create: `packages/retrieval-api/src/retrieval_api/intent_eval.py`
- Test: `packages/retrieval-api/tests/test_intent_eval.py`

**Interfaces:**
- Consumes: `extract_intent(gateway, query, model=None)` from Task 2; `GatewayClient` from `retrieval_api.gateway_client`.
- Produces: `load_intent_cases(path) -> list[dict]` (each case: `{"id": str, "query": str, "expected_filters": dict}`); `check_intent_case(expected_filters: dict, actual_filters: dict) -> bool` (exact dict equality); a CLI (`python -m retrieval_api.intent_eval`) with `--gateway-url` (default `http://localhost:8011`) and `--model` (optional override) that prints a per-case PASS/FAIL and a final `N/total passed` summary.

- [ ] **Step 1: Write the gold-filter dataset**

Create `evals/intent_filter_cases.json`:

```json
[
  {
    "id": "F01",
    "query": "What did the Bombay High Court decide about input tax credit under Rule 6(3)(c)?",
    "expected_filters": {"court": "Bombay High Court"}
  },
  {
    "id": "F02",
    "query": "Recent rulings under the CGST Act on provisional attachment of bank accounts",
    "expected_filters": {"act": "CGST Act"}
  },
  {
    "id": "F03",
    "query": "Explain the scope of section 80HH deduction for backward area undertakings",
    "expected_filters": {"section": "80HH"}
  },
  {
    "id": "F04",
    "query": "case law for Ramesh Gupta vs. Income-tax Officer",
    "expected_filters": {"party": "Ramesh Gupta"}
  },
  {
    "id": "F05",
    "query": "Judgments delivered between 2020-01-01 and 2022-01-01 on transfer pricing comparables",
    "expected_filters": {"date_range": {"gte": "2020-01-01", "lte": "2022-01-01"}}
  },
  {
    "id": "F06",
    "query": "Ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit",
    "expected_filters": {"bench": "Principal Bench"}
  },
  {
    "id": "F07",
    "query": "Judgment authored by Judge D.Y. Chandrachud on constitutional validity of section 271(1)(c)",
    "expected_filters": {"judge": "D.Y. Chandrachud", "section": "271(1)(c)"}
  },
  {
    "id": "F08",
    "query": "Delhi High Court ruling under the Income-tax Act, 1961 on capital gains exemption",
    "expected_filters": {"court": "Delhi High Court", "act": "Income-tax Act, 1961"}
  },
  {
    "id": "F09",
    "query": "Gujarat High Court judgment interpreting section 271(1)(c) as unconstitutional",
    "expected_filters": {"court": "Gujarat High Court", "section": "271(1)(c)"}
  },
  {
    "id": "F10",
    "query": "Can a court order investigators to trace and seize a company's property when investors were defrauded?",
    "expected_filters": {}
  },
  {
    "id": "F11",
    "query": "80HH scrap sale yes useless drum sale no metallic wire factory",
    "expected_filters": {}
  },
  {
    "id": "F12",
    "query": "Alka Khandu Avhad case on cheque dishonour liability under section 138",
    "expected_filters": {"party": "Alka Khandu Avhad", "section": "138"}
  }
]
```

- [ ] **Step 2: Write the failing dataset-loader test**

Create `packages/retrieval-api/tests/test_intent_eval.py`:

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from retrieval_api.intent_eval import check_intent_case, load_intent_cases


def test_repository_intent_filter_dataset_has_twelve_cases_and_unique_ids():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12


def test_load_intent_cases_validates_required_keys(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "F1", "query": "q"}]))

    with pytest.raises(ValueError, match="missing"):
        load_intent_cases(path)


def test_load_intent_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "F1", "query": "q1", "expected_filters": {}},
        {"id": "F1", "query": "q2", "expected_filters": {}},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_intent_cases(path)


def test_check_intent_case_matches_exact_filters():
    assert check_intent_case({"court": "Bombay High Court"}, {"court": "Bombay High Court"}) is True


def test_check_intent_case_flags_mismatch():
    assert check_intent_case({"court": "Bombay High Court"}, {"court": "Delhi High Court"}) is False
    assert check_intent_case({"court": "Bombay High Court"}, {}) is False
    assert check_intent_case({}, {"court": "Bombay High Court"}) is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_intent_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.intent_eval'`.

- [ ] **Step 4: Implement `intent_eval.py`**

Create `packages/retrieval-api/src/retrieval_api/intent_eval.py`:

```python
import argparse
import asyncio
import json
from pathlib import Path

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient


def load_intent_cases(path: str | Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("intent filter eval dataset must be a non-empty JSON array")
    seen: set[str] = set()
    for case in cases:
        required = {"id", "query", "expected_filters"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_intent_case(expected_filters: dict, actual_filters: dict) -> bool:
    return expected_filters == actual_filters


async def run(gateway_url: str, model: str | None, dataset_path: str | Path) -> None:
    cases = load_intent_cases(dataset_path)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0
    for case in cases:
        result = await extract_intent(gateway, case["query"], model=model)
        ok = check_intent_case(case["expected_filters"], result["filters"])
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status} {case['id']}: expected={case['expected_filters']} actual={result['filters']}")
    print(f"\n{passed}/{len(cases)} passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only intent/filter extraction accuracy check")
    parser.add_argument("--gateway-url", default="http://localhost:8011")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/intent_filter_cases.json")
    args = parser.parse_args()
    asyncio.run(run(args.gateway_url, args.model, args.dataset))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_intent_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests pass.

```bash
git add evals/intent_filter_cases.json packages/retrieval-api/src/retrieval_api/intent_eval.py packages/retrieval-api/tests/test_intent_eval.py
git commit -m "feat: add gold-filter eval dataset and prompt-only intent-extraction checker"
```

---

### Task 5: Evaluate `google/gemma-4-E4B-it` and decide

**Files:**
- Modify (only if adopted): `.env.example`

**Interfaces:**
- Consumes: `intent_eval.py`'s CLI from Task 4.

This task is a validation/decision step, not a code change — it only touches `.env.example` if the candidate is adopted.

- [ ] **Step 1: Start a local model-gateway**

```bash
cd packages/model-gateway && uv run uvicorn model_gateway.main:app --port 8011
```

Leave this running in a separate terminal/background process for the remaining steps.

- [ ] **Step 2: Run the baseline check (current default, `Qwen/Qwen3-30B-A3B`)**

```bash
uv run python -m retrieval_api.intent_eval --gateway-url http://localhost:8011
```

Record the `N/12 passed` line and which case IDs failed, if any.

- [ ] **Step 3: Run the candidate check (`google/gemma-4-E4B-it`)**

```bash
uv run python -m retrieval_api.intent_eval --gateway-url http://localhost:8011 --model google/gemma-4-E4B-it
```

Record the `N/12 passed` line and which case IDs failed, if any.

- [ ] **Step 4: Decide**

Compare the two pass counts and, for any case both runs got wrong, whether the failure mode looks structurally the same (e.g. both miss `bench`) or different.

- If `google/gemma-4-E4B-it` matches or beats the `Qwen/Qwen3-30B-A3B` pass count: adopt it. Edit `.env.example`:

  ```
  DEEPINFRA_CHAT_MODEL_SLM=google/gemma-4-E4B-it  # confirmed via evals/intent_filter_cases.json prompt-only check: matches/beats Qwen3-30B-A3B baseline, see docs/small-model-eval-results.md for the retrieval-rank pass this model has NOT yet had (deferred, Phase 2)
  ```

  Then also update your local `.env` (not committed) with the same value if you're running the stack locally, since `.env.example` is a template, not the live config.

- If it's worse: keep `Qwen/Qwen3-30B-A3B` — do not edit `.env.example`. Note the result in your task report so it's not silently re-tried later without cause.

- [ ] **Step 5: Commit if `.env.example` changed**

```bash
git add .env.example
git commit -m "chore: adopt google/gemma-4-E4B-it for slm role per intent_filter_cases.json check"
```

If nothing changed (candidate rejected), skip this step — there's nothing to commit.

- [ ] **Step 6: Stop the local gateway**

Stop the `uvicorn` process started in Step 1.

---

## Self-review notes

- **Spec coverage:** Item 1 (schema enforcement) → Tasks 1-2. Item 2 (filter expansion) → Tasks 2-3. Item 3 (intent taxonomy, classification only) → Task 2. Item 4 (gold-filter eval dataset) → Task 4. Item 5 (model candidate, prompt-only) → Task 5. Non-goals (no retrieval changes) → respected throughout; no task touches `retrieve.py` or `pipeline.py`.
- **Type consistency:** `extract_intent`'s signature (`gateway, query, on_step=None, model=None`) is unchanged end to end across Tasks 2, 4, 5. `GatewayClient.chat`/`chat_with_reasoning`'s new `response_format` parameter name matches `DeepInfraAdapter.chat`'s and `ChatRequest`'s field name exactly, so it round-trips without renaming at any layer.
- **No placeholders:** every step above includes literal code/commands; the only deferred item (RRF weighting / any `intent` consumer) is explicitly named as Phase 2, not left as a vague TODO inside this plan.
