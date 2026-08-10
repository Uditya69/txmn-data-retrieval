# Small-Model Eval Harness (AI Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a single eval run swap the model used for AI Mode's `slm`, `reranker`, and `synthesis` roles via CLI flags, add a rule-based citation-validity check to the eval, and add a script that diffs multiple eval runs — so 3-6B self-host candidates can be A/B'd against the current DeepInfra defaults on the existing 53-query gold set.

**Architecture:** Add an optional `model` override field to the model-gateway's three request schemas (chat/embed/rerank), threaded from `retrieval_eval.py`'s new CLI flags through `GatewayClient` → the AI Mode pipeline functions (`extract_intent`, `rerank_top_chunks`, `synthesize`) → the gateway's `/v1/chat`, `/v1/rerank` endpoints, which use the override instead of the role's configured default when present. The eval script gains a synthesis stage (previously stopped at reranking) with a rule-based citation check reusing `agents.citations.extract_cited_doc_ids`. A new compare script reads N result JSON files and prints deltas.

**Tech Stack:** Python 3.11, FastAPI (model-gateway), httpx (gateway_client), pytest + pytest-asyncio + respx (tests), existing `uv run pytest` from repo root.

## Global Constraints

- `query_embed` role must not be touched by this work — Voyage-only, hard rule (see CLAUDE.md). No override wiring for embed in this plan.
- Do not change `ROLE_PROVIDER_MAP` — a model override changes which model string is sent to the role's existing provider, never which provider is used.
- Preserve every existing test's exact-equality assertions on JSON request bodies (e.g. `test_chat_with_tools_posts_tools_and_returns_tool_calls`) — only include a `"model"` key in an outgoing payload when an override was actually passed, never as an explicit `null`.
- `agent_chat` role is out of scope this round — no CLI flag, no override wiring for the agentic path.
- Run `uv run pytest` from repo root after every task; all 4 packages must stay green.

---

### Task 1: Gateway per-request model override

**Files:**
- Modify: `packages/model-gateway/src/model_gateway/routes.py`
- Test: `packages/model-gateway/tests/test_routes.py`

**Interfaces:**
- Consumes: existing `ROLE_MODEL_MAP`, `ROLE_PROVIDER_MAP`, `_resolve(role) -> tuple[str, str]`, `get_adapter(provider)` — all unchanged.
- Produces: `ChatRequest`, `EmbedRequest`, `RerankRequest` each gain `model: str | None = None`. `/v1/chat` and `/v1/rerank` use `req.model or default_model` as the model string passed to the adapter. (`/v1/embed` schema gains the field for consistency but the endpoint does not need to use it — embed is out of scope; leave its body as-is aside from the schema field so `EmbedRequest` stays valid.)

- [ ] **Step 1: Write the failing tests**

Add to `packages/model-gateway/tests/test_routes.py`:

```python
def test_chat_route_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "default-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    response = client.post(
        "/v1/chat",
        json={"role": "slm", "messages": [{"role": "user", "content": "hi"}], "model": "candidate-model"},
    )

    assert response.status_code == 200
    fake_adapter.chat.assert_awaited_once_with("candidate-model", [{"role": "user", "content": "hi"}], None, None)


def test_chat_route_falls_back_to_role_default_when_model_omitted(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.chat.return_value = ("the answer", {}, None, None)
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"slm": "default-model"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"slm": "deepinfra"})

    client = TestClient(app)
    client.post("/v1/chat", json={"role": "slm", "messages": [{"role": "user", "content": "hi"}]})

    fake_adapter.chat.assert_awaited_once_with("default-model", [{"role": "user", "content": "hi"}], None, None)


def test_rerank_route_uses_override_model_when_provided(monkeypatch):
    fake_adapter = AsyncMock()
    fake_adapter.rerank.return_value = [0.9, 0.1]
    monkeypatch.setattr(routes_module, "get_adapter", lambda provider: fake_adapter)
    monkeypatch.setattr(routes_module, "ROLE_MODEL_MAP", {"reranker": "default-reranker"})
    monkeypatch.setattr(routes_module, "ROLE_PROVIDER_MAP", {"reranker": "deepinfra"})

    client = TestClient(app)
    client.post(
        "/v1/rerank",
        json={"role": "reranker", "query": "q", "documents": ["a", "b"], "model": "candidate-reranker"},
    )

    fake_adapter.rerank.assert_awaited_once_with("candidate-reranker", "q", ["a", "b"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/model-gateway/tests/test_routes.py -v`
Expected: FAIL — `model` is not a valid field on `ChatRequest`/`RerankRequest` (pydantic ignores unknown fields by default unless configured strict, so the actual failure is the `assert_awaited_once_with` mismatch: adapter still called with `"default-model"`/`"default-reranker"` instead of the override).

- [ ] **Step 3: Implement the override**

In `packages/model-gateway/src/model_gateway/routes.py`, change the three request models and the two endpoints that need it:

```python
class ChatRequest(BaseModel):
    role: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | None = None
    model: str | None = None


class EmbedRequest(BaseModel):
    role: str
    text: str
    model: str | None = None


class RerankRequest(BaseModel):
    role: str
    query: str
    documents: list[str]
    model: str | None = None
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
            model, req.messages, req.tools, req.tool_choice,
        )
        generation.update(output=content if content is not None else {"tool_calls": tool_calls}, usage_details=usage_details)
        if reasoning:
            generation.update(metadata={"reasoning": reasoning})
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}
```

```python
@router.post("/v1/rerank")
async def rerank(req: RerankRequest, request: Request):
    default_model, provider = _resolve(req.role)
    model = req.model or default_model
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"rerank:{req.role}",
        model=model,
        input={"query": req.query, "documents": req.documents},
        metadata={"provider": provider, "num_documents": len(req.documents)},
        trace_context=_trace_context_from_headers(request),
    ) as generation:
        scores = await get_adapter(provider).rerank(model, req.query, req.documents)
        generation.update(output=scores)
    return {"scores": scores}
```

Leave `/v1/embed` body unchanged (still uses `_resolve` directly) — `EmbedRequest.model` exists on the schema but the endpoint ignores it, since `query_embed` overrides are out of scope.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/model-gateway/tests/test_routes.py -v`
Expected: PASS, all tests including the 3 new ones and every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add packages/model-gateway/src/model_gateway/routes.py packages/model-gateway/tests/test_routes.py
git commit -m "feat: let gateway chat/rerank requests override the role's default model"
```

---

### Task 2: GatewayClient model passthrough

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/gateway_client.py`
- Test: `packages/retrieval-api/tests/test_gateway_client.py`

**Interfaces:**
- Consumes: Task 1's gateway `/v1/chat`, `/v1/rerank` accepting an optional `model` field.
- Produces: `GatewayClient.chat(role, messages, model=None)`, `GatewayClient.chat_with_reasoning(role, messages, model=None)`, `GatewayClient.rerank(role, query, documents, model=None)`. When `model` is `None`, the outgoing JSON body has no `"model"` key at all (not `"model": null`) — this is required to keep every existing exact-equality body assertion (e.g. `test_chat_with_tools_posts_tools_and_returns_tool_calls`) passing unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_gateway_client.py`:

```python
@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_model_override_when_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}], model="candidate-model")

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-model"


@pytest.mark.asyncio
@respx.mock
async def test_chat_omits_model_key_when_not_provided():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there"})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat(role="slm", messages=[{"role": "user", "content": "hi"}])

    sent = json.loads(route.calls.last.request.content)
    assert "model" not in sent


@pytest.mark.asyncio
@respx.mock
async def test_chat_with_reasoning_sends_model_override():
    route = respx.post("http://gateway/v1/chat").mock(
        return_value=httpx.Response(200, json={"content": "hi there", "reasoning": None})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.chat_with_reasoning(
        role="synthesis", messages=[{"role": "user", "content": "hi"}], model="candidate-synth-model",
    )

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-synth-model"


@pytest.mark.asyncio
@respx.mock
async def test_rerank_sends_model_override():
    route = respx.post("http://gateway/v1/rerank").mock(
        return_value=httpx.Response(200, json={"scores": [0.5]})
    )
    client = GatewayClient(base_url="http://gateway")

    await client.rerank(role="reranker", query="q", documents=["a"], model="candidate-reranker")

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "candidate-reranker"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: FAIL — `chat()`/`chat_with_reasoning()`/`rerank()` raise `TypeError: unexpected keyword argument 'model'`.

- [ ] **Step 3: Implement the passthrough**

In `packages/retrieval-api/src/retrieval_api/gateway_client.py`:

```python
    async def chat(self, role: str, messages: list[dict], model: str | None = None) -> str:
        content, _reasoning = await self.chat_with_reasoning(role, messages, model=model)
        return content

    async def chat_with_reasoning(
        self, role: str, messages: list[dict], model: str | None = None,
    ) -> tuple[str, str | None]:
        body = {"role": role, "messages": messages}
        if model is not None:
            body["model"] = model
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._base_url}/v1/chat", json=body, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return data["content"], data.get("reasoning")
```

```python
    async def rerank(
        self, role: str, query: str, documents: list[str], model: str | None = None,
    ) -> list[float]:
        body = {"role": role, "query": query, "documents": documents}
        if model is not None:
            body["model"] = model
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._base_url}/v1/rerank", json=body, headers=self._headers())
            response.raise_for_status()
            return response.json()["scores"]
```

Leave `chat_with_tools` and `embed` unchanged — `agent_chat` and `query_embed` overrides are out of scope this round.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_gateway_client.py -v`
Expected: PASS, all tests including the 4 new ones and every pre-existing test (in particular `test_chat_with_tools_posts_tools_and_returns_tool_calls`'s exact-body assertion, still unaffected since `chat_with_tools` wasn't touched).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/gateway_client.py packages/retrieval-api/tests/test_gateway_client.py
git commit -m "feat: let GatewayClient pass a per-call model override for chat/rerank"
```

---

### Task 3: Thread model override through the AI Mode pipeline functions

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/rerank.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_rerank_citations.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_synthesize.py`

**Interfaces:**
- Consumes: Task 2's `GatewayClient.chat(role, messages, model=None)`, `.chat_with_reasoning(role, messages, model=None)`, `.rerank(role, query, documents, model=None)`.
- Produces: `extract_intent(gateway, query, on_step=None, model=None)`, `rerank_top_chunks(gateway, query, candidates, top_n=None, model=None)`, `synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, model=None)` — each accepts the new optional `model` kwarg and forwards it to the underlying gateway call. When `model` is `None`, behavior is byte-for-byte identical to today (this is what every existing test in these three files already exercises, so none of them should need changes).

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py`:

```python
@pytest.mark.asyncio
async def test_extract_intent_forwards_model_override_and_skips_get_model():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "candidate model test",
        "intent": "test",
        "filters": {},
    })

    await extract_intent(gateway, "candidate model test", model="Qwen/Qwen3-4B-Instruct-2507")

    gateway.get_model.assert_not_awaited()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["model"] == "Qwen/Qwen3-4B-Instruct-2507"
```

Add to `packages/retrieval-api/tests/test_ai_mode_rerank_citations.py`:

```python
@pytest.mark.asyncio
async def test_rerank_top_chunks_forwards_model_override():
    gateway = AsyncMock()
    gateway.rerank.return_value = [0.9, 0.1]
    candidates = [
        {"chunk_id": "a", "text": "A", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "B", "rrf_score": 0.02},
    ]

    await rerank_top_chunks(gateway, "query", candidates, top_n=2, model="Qwen/Qwen3-Reranker-0.6B")

    call_kwargs = gateway.rerank.await_args.kwargs
    assert call_kwargs["model"] == "Qwen/Qwen3-Reranker-0.6B"
```

Add to `packages/retrieval-api/tests/test_ai_mode_synthesize.py`:

```python
@pytest.mark.asyncio
async def test_synthesize_forwards_model_override(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", None)

    await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {"masterinfo": {"court": "SC"}}},
        model="Qwen/Qwen3-4B-Thinking-2507",
    )

    call_kwargs = gateway.chat_with_reasoning.await_args.kwargs
    assert call_kwargs["model"] == "Qwen/Qwen3-4B-Thinking-2507"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py packages/retrieval-api/tests/test_ai_mode_rerank_citations.py packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: FAIL — each function raises `TypeError: unexpected keyword argument 'model'`.

- [ ] **Step 3: Implement the threading**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, change `extract_intent`:

```python
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
    )
    cleaned = _extract_json_object(response.strip())
    try:
        result = json.loads(cleaned)
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

In `packages/retrieval-api/src/retrieval_api/ai_mode/rerank.py`, change `rerank_top_chunks`:

```python
async def rerank_top_chunks(
    gateway: GatewayClient, query: str, candidates: list[dict],
    top_n: int | None = None, model: str | None = None,
) -> list[dict]:
    scores = await gateway.rerank(
        role="reranker", query=query, documents=[c["text"] for c in candidates], model=model,
    )
    scored = [{**c, "rerank_score": score} for c, score in zip(candidates, scores)]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    cutoff = top_n if top_n is not None else elbow_cutoff(
        [row["rerank_score"] for row in scored], max_keep=_MAX_CHUNKS,
    )
    return scored[:cutoff]
```

In `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`, change `synthesize`:

```python
async def synthesize(
    gateway, es_client, query: str, top_chunks: list[dict], citations: dict,
    on_step: OnStep | None = None, model: str | None = None,
) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}"

    if on_step is not None:
        await on_step("synthesis_prompt", {"prompt": prompt})

    answer, reasoning = await gateway.chat_with_reasoning(
        role="synthesis",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

    return {"answer": answer, "citations": citations, "reasoning": reasoning}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py packages/retrieval-api/tests/test_ai_mode_rerank_citations.py packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: PASS, all tests including the 3 new ones. Every pre-existing test in these files must still pass unmodified since `model=None` reproduces prior behavior exactly (note: pre-existing tests mock `gateway.get_model`, which is still called when `model` is `None`, exactly as before).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/src/retrieval_api/ai_mode/rerank.py packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py packages/retrieval-api/tests/test_ai_mode_intent.py packages/retrieval-api/tests/test_ai_mode_rerank_citations.py packages/retrieval-api/tests/test_ai_mode_synthesize.py
git commit -m "feat: thread optional model override through intent/rerank/synthesize"
```

---

### Task 4: Extend `retrieval_eval.py` with model flags, a synthesis stage, and a 12-query sample

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`
- Test: `packages/retrieval-api/tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: Task 3's `extract_intent(..., model=None)`, `rerank_top_chunks(..., model=None)`; new `synthesize(..., model=None)`; `retrieval_api.ai_mode.citations.prefetch_citations(es_client, candidates, top_n_docs=20)`; `agents.citations.extract_cited_doc_ids(answer: str) -> set[str]`; `retrieval_api.score_cutoff.elbow_cutoff`.
- Produces: `SAMPLE_12_QUERY_IDS: list[str]` module constant. `evaluate_case(case, gateway, es_client, milvus_client, *, limit=50, langfuse_enabled=True, slm_model=None, reranker_model=None, synthesis_model=None) -> dict` — result dict gains `"synthesis_answer"`, `"citation_valid"`, `"citation_invalid_ids"`, `"citation_count"`, `"gold_cited"` keys alongside the existing ones. New CLI flags `--slm-model`, `--reranker-model`, `--synthesis-model`, `--sample12`.

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_retrieval_eval.py` (it already has `test_evaluate_case_reports_each_retrieval_stage` with fakes for `raw_search`, `hybrid_search`, `extract_intent`, `resolve_allowlist` — read the rest of that test first, then extend the same fake set with a fake `synthesize` and `run_agentic_search`, mirroring its existing monkeypatch style):

```python
def test_sample_12_query_ids_are_a_subset_of_the_full_dataset():
    from retrieval_api.retrieval_eval import SAMPLE_12_QUERY_IDS
    root = Path(__file__).parents[3]
    cases = load_cases(root / "evals" / "retrieval_cases.json")
    all_ids = {case["id"] for case in cases}
    assert len(SAMPLE_12_QUERY_IDS) == 12
    assert len(set(SAMPLE_12_QUERY_IDS)) == 12  # no duplicates
    assert set(SAMPLE_12_QUERY_IDS).issubset(all_ids)
    sampled_classes = {case["class"] for case in cases if case["id"] in SAMPLE_12_QUERY_IDS}
    assert sampled_classes == {"direct", "indirect", "adversarial"}  # every class represented


@pytest.mark.asyncio
async def test_evaluate_case_records_citation_validity_against_reranked_chunks(monkeypatch):
    import retrieval_api.retrieval_eval as module

    async def fake_raw_search(client, query, limit=50):
        return []

    async def fake_hybrid(client, collections, dense_vector, sparse_query_text,
                          doc_id_allowlist=None, limit=50):
        suffix = "dense" if dense_vector is not None else "sparse"
        return {name: [{"doc_id": "gold", "chunk_id": f"gold-{name}-{suffix}",
                        "text": "gold text", "score": 1.0}] for name in collections}

    async def fake_intent(gateway, query, model=None):
        return {"rewritten_query": query, "filters": {}, "intent": "test"}

    async def fake_allowlist(es_client, filters):
        return None

    async def fake_rerank(gateway, query, candidates, top_n=None, model=None):
        return candidates[:1]  # only "gold" survives reranking

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, model=None):
        return {"answer": "The point is settled [gold] and also [not-retrieved].", "citations": {}, "reasoning": None}

    async def fake_agentic(gateway, es_client, milvus_client, query):
        return {"ok": True, "doc_ids": ["gold"]}

    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(module, "extract_intent", fake_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_allowlist)
    monkeypatch.setattr(module, "rerank_top_chunks", fake_rerank)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)
    monkeypatch.setattr(module, "run_agentic_search", fake_agentic)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    case = {"id": "Q01", "class": "direct", "query": "q", "gold_doc_ids": ["gold"],
            "expected_collections": ["facts"], "pass_at": 5}

    result = await evaluate_case(case, gateway, es_client=object(), milvus_client=object(), langfuse_enabled=False)

    assert result["citation_count"] == 2
    assert result["citation_invalid_ids"] == ["not-retrieved"]
    assert result["citation_valid"] is False
    assert result["gold_cited"] is True
    assert result["synthesis_answer"] == "The point is settled [gold] and also [not-retrieved]."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: FAIL — `SAMPLE_12_QUERY_IDS` doesn't exist yet; `evaluate_case` result has no `citation_*`/`gold_cited`/`synthesis_answer` keys yet.

- [ ] **Step 3: Implement**

In `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`, add imports and the sample constant near the top:

```python
from agents.citations import extract_cited_doc_ids
from retrieval_api.ai_mode.citations import prefetch_citations
from retrieval_api.ai_mode.synthesize import synthesize

# One query per class from each era band (1927-1965, 1995-2017, 2025-2026),
# stratified across direct/indirect/adversarial so a fast candidate-model run
# still exercises every query class and every corpus era.
SAMPLE_12_QUERY_IDS = [
    "Q01", "Q15", "Q27", "Q51",  # direct
    "Q02", "Q16", "Q28", "Q52",  # indirect
    "Q23", "Q35", "Q47", "Q53",  # adversarial
]
```

Change `evaluate_case`'s signature and body — add the three model kwargs and the synthesis+citation block right after the existing `reranked` computation:

```python
async def evaluate_case(case: dict, gateway, es_client, milvus_client, *, limit: int = 50,
                        langfuse_enabled: bool = True, slm_model: str | None = None,
                        reranker_model: str | None = None, synthesis_model: str | None = None) -> dict:
    query = case["query"]
    gold = set(case["gold_doc_ids"])
    langfuse = get_client()
    context = langfuse.start_as_current_observation(
        as_type="evaluator", name="retrieval-eval", input={"id": case["id"], "query": query},
        metadata={"class": case["class"], "pair": case.get("pair", "")},
    ) if langfuse_enabled else nullcontext(None)

    with context as span:
        started = time.perf_counter()
        errors: dict[str, str] = {}
        timings: dict[str, float] = {}

        async def measured(name, awaitable):
            stage_started = time.perf_counter()
            try:
                return await awaitable
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                return None
            finally:
                timings[name] = round((time.perf_counter() - stage_started) * 1000, 1)

        es_rows = await measured("es", raw_search(es_client, query, limit=limit)) or []
        raw_vector = await measured("raw_embedding", gateway.embed(role="query_embed", text=query))
        raw_dense = (
            await measured("raw_dense", hybrid_search(
                milvus_client, MILVUS_COLLECTIONS, raw_vector, query, limit=limit,
            )) if raw_vector is not None else None
        ) or {name: [] for name in MILVUS_COLLECTIONS}
        raw_sparse = await measured("raw_sparse", hybrid_search(
            milvus_client, MILVUS_COLLECTIONS, None, query, limit=limit,
        )) or {name: [] for name in MILVUS_COLLECTIONS}

        intent = await measured("intent", extract_intent(gateway, query, model=slm_model))
        rewritten_query = intent.get("rewritten_query", query) if intent else query
        allowlist = await measured("filters", resolve_allowlist(es_client, intent.get("filters", {}))) if intent else None
        rewritten_vector = raw_vector if rewritten_query == query else await measured(
            "rewritten_embedding", gateway.embed(role="query_embed", text=rewritten_query),
        )
        rewritten_dense = (
            await measured("rewritten_dense", hybrid_search(
                milvus_client, MILVUS_COLLECTIONS, rewritten_vector, rewritten_query,
                doc_id_allowlist=allowlist, limit=limit,
            )) if rewritten_vector is not None else None
        ) or {name: [] for name in MILVUS_COLLECTIONS}
        rewritten_sparse = await measured("rewritten_sparse", hybrid_search(
            milvus_client, MILVUS_COLLECTIONS, None, rewritten_query,
            doc_id_allowlist=allowlist, limit=limit,
        )) or {name: [] for name in MILVUS_COLLECTIONS}

        dense_flat = _flatten(rewritten_dense)
        sparse_flat = _flatten(rewritten_sparse)
        merged = rrf_merge(dense_flat, sparse_flat)
        reranked = await measured(
            "reranker", rerank_top_chunks(gateway, query, merged, top_n=len(merged), model=reranker_model),
        ) if merged else []
        reranked = reranked or []

        synthesis_answer = None
        citation_count = 0
        citation_invalid_ids: list[str] = []
        citation_valid = False
        gold_cited = False
        if reranked:
            synthesis_cutoff = elbow_cutoff([row["rerank_score"] for row in reranked], max_keep=5)
            synthesis_chunks = reranked[:synthesis_cutoff]
            citations = await measured("prefetch_citations", prefetch_citations(es_client, merged))
            synth_result = await measured(
                "synthesis", synthesize(
                    gateway, es_client, rewritten_query, synthesis_chunks, citations or {}, model=synthesis_model,
                ),
            )
            if synth_result is not None:
                synthesis_answer = synth_result["answer"]
                seen_doc_ids = {c["doc_id"] for c in synthesis_chunks}
                cited_ids = extract_cited_doc_ids(synthesis_answer)
                citation_count = len(cited_ids)
                citation_invalid_ids = sorted(cited_ids - seen_doc_ids)
                citation_valid = not citation_invalid_ids
                gold_cited = bool(gold & cited_ids)

        agentic_result = await measured("agentic", run_agentic_search(gateway, es_client, milvus_client, query))
        agentic_doc_ids = None
        if agentic_result is not None:
            if agentic_result.get("ok"):
                agentic_doc_ids = agentic_result.get("doc_ids")
            else:
                errors["agentic"] = f"unverifiable_answer: {agentic_result.get('invalid_doc_ids')}"

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
        result = {
            "id": case["id"], "pair": case.get("pair"), "class": case["class"],
            "query": query, "gold_doc_ids": case["gold_doc_ids"],
            "rewritten_query": rewritten_query, "pass_at": case["pass_at"],
            "ranks": ranks,
            "collection_ranks": {
                "raw_dense": _collection_ranks(raw_dense, gold),
                "raw_sparse": _collection_ranks(raw_sparse, gold),
                "rewritten_dense": _collection_ranks(rewritten_dense, gold),
                "rewritten_sparse": _collection_ranks(rewritten_sparse, gold),
            },
            "synthesis_answer": synthesis_answer,
            "citation_count": citation_count,
            "citation_invalid_ids": citation_invalid_ids,
            "citation_valid": citation_valid,
            "gold_cited": gold_cited,
            "errors": errors, "timings_ms": timings,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if langfuse_enabled:
            span.update(output={"ranks": ranks, "errors": errors, "rewritten_query": rewritten_query})
            for stage, rank in ranks.items():
                langfuse.score_current_trace(
                    name=f"{stage}_recall_at_{case['pass_at']}", value=rank is not None and rank <= case["pass_at"],
                    data_type="BOOLEAN",
                )
                if rank is not None:
                    langfuse.score_current_trace(name=f"{stage}_rank", value=float(rank), data_type="NUMERIC")
            langfuse.score_current_trace(name="citation_valid", value=citation_valid, data_type="BOOLEAN")
        return result
```

Also import `elbow_cutoff`:

```python
from retrieval_api.score_cutoff import elbow_cutoff
```

Add the CLI flags in `main()`, right after the existing `--limit` argument:

```python
    parser.add_argument("--slm-model", help="override the DeepInfra model used for the slm role")
    parser.add_argument("--reranker-model", help="override the DeepInfra model used for the reranker role")
    parser.add_argument("--synthesis-model", help="override the DeepInfra model used for the synthesis role")
    parser.add_argument("--sample12", action="store_true", help="scope to the fixed 12-query stratified sample")
```

In `_run(args)`, right after the existing `--query`/`--class` filtering block, add:

```python
    if args.sample12:
        wanted = set(SAMPLE_12_QUERY_IDS)
        cases = [case for case in cases if case["id"] in wanted]
```

And thread the three model args into the `evaluate_case` call:

```python
            result = await evaluate_case(
                case, gateway, es_client, milvus_client, limit=args.limit,
                langfuse_enabled=not args.no_langfuse,
                slm_model=args.slm_model, reranker_model=args.reranker_model, synthesis_model=args.synthesis_model,
            )
```

Finally, extend `payload["parameters"]` in `_run` to record which models were used for this run (needed by the Task 5 compare script):

```python
            "parameters": {
                "limit": args.limit,
                "query_ids": args.query,
                "query_class": args.query_class,
                "langfuse_enabled": not args.no_langfuse,
                "slm_model": args.slm_model,
                "reranker_model": args.reranker_model,
                "synthesis_model": args.synthesis_model,
            },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: PASS, all tests including the 2 new ones. The pre-existing `test_evaluate_case_reports_each_retrieval_stage` test must still pass — check whether it needs a `synthesize`/`run_agentic_search` fake added (it currently only fakes `raw_search`, `hybrid_search`, `extract_intent`, `resolve_allowlist`); if `run_agentic_search`/`synthesize` aren't monkeypatched there, real network calls would be attempted and the test would hang/fail — add fakes for both to that existing test as part of this step if missing, mirroring the pattern in the new test above.

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/retrieval_eval.py packages/retrieval-api/tests/test_retrieval_eval.py
git commit -m "feat: add model-override flags, synthesis stage, and citation check to retrieval_eval"
```

---

### Task 5: Compare-runs script

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/compare_eval_runs.py`
- Test: `packages/retrieval-api/tests/test_compare_eval_runs.py`

**Interfaces:**
- Consumes: the `.eval-results/*.json` payload shape produced by `retrieval_eval.py`'s `_run` (`{"run_name": str, "parameters": {...}, "results": [{"id": str, "pass_at": int, "ranks": {...}, "citation_valid": bool, ...}]}`).
- Produces: `load_run(path: Path) -> dict`, `stage_pass_rate(run: dict, stage: str) -> tuple[int, int]` (passed, total), `citation_pass_rate(run: dict) -> tuple[int, int]`, `build_comparison_table(baseline: dict, candidates: list[dict]) -> list[dict]` (one row per candidate: `{"run_name": str, "stage_deltas": {stage: delta_passed}, "citation_pass_delta": int}`), and a `main()` CLI entry point.

- [ ] **Step 1: Write the failing tests**

Create `packages/retrieval-api/tests/test_compare_eval_runs.py`:

```python
import json
from pathlib import Path

import pytest

from retrieval_api.compare_eval_runs import (
    build_comparison_table, citation_pass_rate, load_run, stage_pass_rate,
)


def _run(name, ranks_list, citation_valids):
    return {
        "run_name": name,
        "parameters": {"slm_model": None, "reranker_model": None, "synthesis_model": None},
        "results": [
            {"id": f"Q{i}", "pass_at": 5, "ranks": ranks, "citation_valid": valid}
            for i, (ranks, valid) in enumerate(zip(ranks_list, citation_valids))
        ],
    }


def test_load_run_reads_json_file(tmp_path):
    path = tmp_path / "run.json"
    payload = _run("baseline", [{"es": 1}], [True])
    path.write_text(json.dumps(payload))

    assert load_run(path) == payload


def test_stage_pass_rate_counts_ranks_within_pass_at():
    run = _run("baseline", [{"es": 1}, {"es": 10}, {"es": None}], [True, True, True])

    passed, total = stage_pass_rate(run, "es")

    assert (passed, total) == (1, 3)


def test_citation_pass_rate_counts_valid_flags():
    run = _run("baseline", [{}, {}, {}], [True, False, True])

    passed, total = citation_pass_rate(run)

    assert (passed, total) == (2, 3)


def test_build_comparison_table_reports_delta_vs_baseline():
    baseline = _run("baseline", [{"es": 1}, {"es": 10}], [True, True])
    candidate = _run("candidate", [{"es": 1}, {"es": None}], [True, False])

    table = build_comparison_table(baseline, [candidate])

    assert table == [{
        "run_name": "candidate",
        "stage_deltas": {"es": -1},
        "citation_pass_delta": -1,
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_compare_eval_runs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.compare_eval_runs'`.

- [ ] **Step 3: Write the implementation**

Create `packages/retrieval-api/src/retrieval_api/compare_eval_runs.py`:

```python
import argparse
import json
from pathlib import Path

_STAGES = [
    "es", "raw_dense", "raw_sparse", "rewritten_dense", "rewritten_sparse",
    "rrf", "reranker", "agentic",
]


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def stage_pass_rate(run: dict, stage: str) -> tuple[int, int]:
    results = run["results"]
    passed = sum(
        1 for r in results
        if r["ranks"].get(stage) is not None and r["ranks"][stage] <= r["pass_at"]
    )
    return passed, len(results)


def citation_pass_rate(run: dict) -> tuple[int, int]:
    results = run["results"]
    passed = sum(1 for r in results if r.get("citation_valid"))
    return passed, len(results)


def build_comparison_table(baseline: dict, candidates: list[dict]) -> list[dict]:
    baseline_stage_passed = {stage: stage_pass_rate(baseline, stage)[0] for stage in _STAGES}
    baseline_citation_passed = citation_pass_rate(baseline)[0]
    table = []
    for candidate in candidates:
        stage_deltas = {
            stage: stage_pass_rate(candidate, stage)[0] - baseline_stage_passed[stage]
            for stage in _STAGES
        }
        citation_delta = citation_pass_rate(candidate)[0] - baseline_citation_passed
        table.append({
            "run_name": candidate["run_name"],
            "stage_deltas": stage_deltas,
            "citation_pass_delta": citation_delta,
        })
    return table


def _print_table(baseline: dict, table: list[dict]) -> None:
    total = len(baseline["results"])
    print(f"Baseline: {baseline['run_name']} ({total} queries)")
    header = "run_name".ljust(28) + "".join(s[:10].rjust(12) for s in _STAGES) + "citation".rjust(12)
    print(header)
    for row in table:
        cells = "".join(f"{row['stage_deltas'][s]:+d}".rjust(12) for s in _STAGES)
        print(row["run_name"].ljust(28) + cells + f"{row['citation_pass_delta']:+d}".rjust(12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval runs against a baseline")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    candidates = [load_run(path) for path in args.candidates]
    table = build_comparison_table(baseline, candidates)
    _print_table(baseline, table)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_compare_eval_runs.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 5: Register the CLI entry point and commit**

Check `packages/retrieval-api/pyproject.toml` for how `retrieval-eval` is registered under `[project.scripts]`, and add a matching entry for the new script (e.g. `compare-eval-runs = "retrieval_api.compare_eval_runs:main"`), following the exact same pattern already used there.

```bash
git add packages/retrieval-api/src/retrieval_api/compare_eval_runs.py packages/retrieval-api/tests/test_compare_eval_runs.py packages/retrieval-api/pyproject.toml
git commit -m "feat: add compare_eval_runs script to diff eval runs against a baseline"
```

---

### Task 6: Run the baseline + 4 candidate evals and record results

This task has no unit tests — it's the actual A/B run using the harness built in Tasks 1-5, against the running Docker stack. Do not skip verifying the stack is up first.

**Files:**
- Create: `docs/small-model-eval-results.md` (results doc, follows the pattern of `docs/agent-model-comparison.md`)

- [ ] **Step 1: Confirm the stack is up**

Run: `docker compose ps`
Expected: `retrieval-api`, `model-gateway`, and dependencies are `Up`. If not, `docker compose up -d --build` first per CLAUDE.md.

- [ ] **Step 2: Confirm each candidate model is actually served by DeepInfra**

Run: `curl -s https://api.deepinfra.com/models/list | python3 -c "import json,sys; names={m['model_name'] for m in json.load(sys.stdin)}; print([n for n in ['Qwen/Qwen3-4B-Instruct-2507','Qwen/Qwen3-Reranker-0.6B','BAAI/bge-reranker-v2-m3','Qwen/Qwen3-4B-Thinking-2507'] if n not in names])"`
Expected: empty list. If any candidate name is missing, find its exact DeepInfra catalog name before proceeding (model names must match exactly what the DeepInfra adapter sends) and swap it into the runs below.

- [ ] **Step 3: Run the baseline**

```bash
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 --sample12 --no-langfuse \
  --run-name small-model-eval-baseline
```

- [ ] **Step 4: Run each candidate**

```bash
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 --sample12 --no-langfuse \
  --run-name small-model-eval-slm-qwen3-4b \
  --slm-model Qwen/Qwen3-4B-Instruct-2507

uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 --sample12 --no-langfuse \
  --run-name small-model-eval-reranker-qwen3-0.6b \
  --reranker-model Qwen/Qwen3-Reranker-0.6B

uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 --sample12 --no-langfuse \
  --run-name small-model-eval-reranker-bge-v2-m3 \
  --reranker-model BAAI/bge-reranker-v2-m3

uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 --sample12 --no-langfuse \
  --run-name small-model-eval-synthesis-qwen3-4b-thinking \
  --synthesis-model Qwen/Qwen3-4B-Thinking-2507
```

- [ ] **Step 5: Compare each candidate against the baseline**

```bash
uv run python -m retrieval_api.compare_eval_runs \
  .eval-results/*small-model-eval-baseline.json \
  .eval-results/*small-model-eval-slm-qwen3-4b.json \
  .eval-results/*small-model-eval-reranker-qwen3-0.6b.json \
  .eval-results/*small-model-eval-reranker-bge-v2-m3.json \
  .eval-results/*small-model-eval-synthesis-qwen3-4b-thinking.json
```

- [ ] **Step 6: Write up the results**

Create `docs/small-model-eval-results.md` following `docs/agent-model-comparison.md`'s structure (candidates table, methodology, results table, conclusion) — fill it in with the actual `compare_eval_runs.py` output and each run's `citation_invalid_ids`/`gold_cited` findings, not placeholder text.

- [ ] **Step 7: Commit**

```bash
git add docs/small-model-eval-results.md
git commit -m "docs: record small-model eval results for AI Mode slm/reranker/synthesis roles"
```
