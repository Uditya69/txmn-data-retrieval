# AI Mode Trace Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream AI Mode's internal pipeline stages (SLM rewrite, filter resolution, dense+sparse Milvus retrieval, RRF merge, rerank, synthesis prompt) to the web UI live, as they happen, in a Dev-Mode-only right-hand panel.

**Architecture:** Thread an optional `on_step(step: str, data: dict)` async callback through every AI Mode pipeline stage function. `ws.py` builds a callback that sends `{"type": "ai_mode_trace", "step": ..., "data": ...}` over the websocket, serialized against `instant_task`'s send via one `asyncio.Lock` (both now send concurrently). Frontend's `useSearch` accumulates these into a `traceSteps` array; a new `TracePanel` component renders them as they arrive; `App.tsx` switches to a 2-column layout when Dev Mode is on and a trace exists.

**Tech Stack:** Python 3.11, FastAPI/Starlette websockets, pytest + pytest-asyncio (backend); React + TypeScript, Vitest + Testing Library (frontend).

## Global Constraints

- Full spec: `docs/superpowers/specs/2026-08-04-ai-mode-trace-panel-design.md`.
- Trace applies to AI Mode only — Instant is unaffected (no trace steps, no layout change for Instant-only queries).
- `on_step` is optional everywhere (`on_step=None` default); when `None`, stages must skip the call entirely — zero cost when unused, and existing non-websocket callers keep working unchanged.
- If `on_step` raises, swallow the exception at the call site — a broken trace channel must never fail the pipeline or the final answer.
- Payloads are capped at the source: `text_preview` = first 200 chars; per-collection Milvus lists = top 5; RRF list = top 15; rerank keeps its existing top 3 (full text, no preview truncation there).
- `query_embed` role stays on Voyage; no change to any embedding/search call semantics — this task only adds observability, not new retrieval calls.
- Follow existing code patterns: `.module.css` per component, `data-testid` for loading/empty states (see `OverviewCard.tsx`'s `data-testid="overview-loading"`), monkeypatch-the-module style used throughout the existing pytest suite (e.g. `import retrieval_api.ai_mode.pipeline as module; monkeypatch.setattr(module, "extract_intent", fake)`).

---

### Task 1: `extract_intent` emits the `intent` trace step

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Produces: `extract_intent(gateway, query: str, on_step: Callable[[str, dict], Awaitable[None]] | None = None) -> dict` — same return shape as today (`{"rewritten_query", "intent", "filters"}`); on success, additionally awaits `on_step("intent", {"query": query, "rewritten_query": ..., "intent": ..., "filters": ...})` before returning.

- [ ] **Step 1: Write the failing test**

Append to `packages/retrieval-api/tests/test_ai_mode_intent.py`:

```python
@pytest.mark.asyncio
async def test_extract_intent_emits_intent_step_when_on_step_given():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "rewritten",
        "intent": "taxation",
        "filters": {"act": "CGST Act"},
    })
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await extract_intent(gateway, "original query", on_step=on_step)

    assert result == {"rewritten_query": "rewritten", "intent": "taxation", "filters": {"act": "CGST Act"}}
    assert steps == [("intent", {
        "query": "original query",
        "rewritten_query": "rewritten",
        "intent": "taxation",
        "filters": {"act": "CGST Act"},
    })]


@pytest.mark.asyncio
async def test_extract_intent_skips_on_step_when_none():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({"rewritten_query": "r", "intent": "x", "filters": {}})

    result = await extract_intent(gateway, "q")  # no on_step passed

    assert result == {"rewritten_query": "r", "intent": "x", "filters": {}}
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: the two new tests FAIL with `TypeError: extract_intent() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Implement**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, change the signature and add the emit:

```python
import json
from typing import Awaitable, Callable

from retrieval_api.gateway_client import GatewayClient

OnStep = Callable[[str, dict], Awaitable[None]]


def _extract_json_object(text: str) -> str:
    """SLMs often wrap JSON in prose and/or a markdown code fence despite
    instructions not to - pull out the outermost {...} object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]

_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
Given a user query, return ONLY a JSON object with exactly these keys:
- "rewritten_query": the query rewritten for search, expanding any old-law
  references to their new-law equivalent (IPC -> BNS, CrPC -> BNSS, Evidence
  Act -> BSA) where applicable.
- "intent": one short intent category label.
- "filters": an object with any of "court", "act", "date_range", "party"
  the query explicitly mentions; omit keys that aren't mentioned.
"""


async def extract_intent(gateway: GatewayClient, query: str, on_step: OnStep | None = None) -> dict:
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    cleaned = _extract_json_object(response.strip())
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SLM did not return valid JSON: {response!r}") from exc

    if on_step is not None:
        await on_step("intent", {"query": query, **result})

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: all PASS (6 tests total: 4 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat(retrieval-api): emit intent trace step from extract_intent"
```

---

### Task 2: `resolve_allowlist` emits the `filters_resolved` trace step

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/filter_resolve.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_filter_resolve.py`

**Interfaces:**
- Produces: `resolve_allowlist(es_client, filters: dict, on_step: OnStep | None = None) -> list[str] | None` — same return as today; additionally awaits `on_step("filters_resolved", {"filters": filters, "doc_id_count": N, "doc_id_sample": [...]})` where `doc_id_sample` is the first 10 of the result (or `[]` if result is `None`), and `doc_id_count` is `len(result)` (or `0` if `None`).

- [ ] **Step 1: Write the failing test**

Append to `packages/retrieval-api/tests/test_ai_mode_filter_resolve.py`:

```python
@pytest.mark.asyncio
async def test_resolve_allowlist_emits_filters_resolved_step_with_no_filters():
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await resolve_allowlist(es_client=object(), filters={}, on_step=on_step)

    assert result is None
    assert steps == [("filters_resolved", {"filters": {}, "doc_id_count": 0, "doc_id_sample": []})]


@pytest.mark.asyncio
async def test_resolve_allowlist_emits_filters_resolved_step_with_matches(monkeypatch):
    import retrieval_api.ai_mode.filter_resolve as module

    async def fake_resolve(client, filters):
        return ["d1", "d2", "d3"]

    monkeypatch.setattr(module, "resolve_doc_id_allowlist", fake_resolve)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await resolve_allowlist(es_client=object(), filters={"court": "Supreme Court"}, on_step=on_step)

    assert result == ["d1", "d2", "d3"]
    assert steps == [("filters_resolved", {
        "filters": {"court": "Supreme Court"}, "doc_id_count": 3, "doc_id_sample": ["d1", "d2", "d3"],
    })]
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_filter_resolve.py -v`
Expected: FAIL with `TypeError: resolve_allowlist() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ai_mode/filter_resolve.py`:

```python
from common.es_client import resolve_doc_id_allowlist
from retrieval_api.ai_mode.intent import OnStep


async def resolve_allowlist(es_client, filters: dict, on_step: OnStep | None = None) -> list[str] | None:
    result = await resolve_doc_id_allowlist(es_client, filters)

    if on_step is not None:
        sample = (result or [])[:10]
        await on_step("filters_resolved", {"filters": filters, "doc_id_count": len(result or []), "doc_id_sample": sample})

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_filter_resolve.py -v`
Expected: all PASS (4 tests total: 2 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/filter_resolve.py packages/retrieval-api/tests/test_ai_mode_filter_resolve.py
git commit -m "feat(retrieval-api): emit filters_resolved trace step from resolve_allowlist"
```

---

### Task 3: `retrieve` emits `milvus_dense`, `milvus_sparse`, `rrf_merge` trace steps

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_retrieve.py`

**Interfaces:**
- Produces: `retrieve(gateway, milvus_client, rewritten_query: str, doc_id_allowlist: list[str] | None, on_step: OnStep | None = None) -> list[dict]` — same return as today (RRF-merged candidate list); additionally emits, in order: `milvus_dense`, `milvus_sparse` (each `{"collections": [{"name", "hit_count", "top_hits": [{"chunk_id", "doc_id", "score", "text_preview"}]}]}`, `top_hits` capped at 5 per collection, `text_preview` = first 200 chars of `text`), then `rrf_merge` (`{"candidate_count": N, "top_candidates": [{"chunk_id", "doc_id", "rrf_score", "text_preview"}]}`, capped at 15).
- Produces (new, module-level helper, also directly tested): `_collection_trace(by_collection: dict[str, list[dict]]) -> dict` — builds the `{"collections": [...]}` shape used by both `milvus_dense` and `milvus_sparse`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.ai_mode.retrieve import _collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = _collection_trace({"ruling": rows})

    assert trace == {
        "collections": [{
            "name": "ruling",
            "hit_count": 7,
            "top_hits": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "score": float(i), "text_preview": "x" * 200}
                for i in range(5)
            ],
        }]
    }


@pytest.mark.asyncio
async def test_retrieve_emits_dense_sparse_and_rrf_merge_steps(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await module.retrieve(gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, on_step=on_step)

    assert steps == ["milvus_dense", "milvus_sparse", "rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: FAIL — `_collection_trace` doesn't exist, and `retrieve()` doesn't accept `on_step`.

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`:

```python
from common.milvus_client import hybrid_search
from common.schemas import MILVUS_COLLECTIONS
from retrieval_api.ai_mode.intent import OnStep
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


def _collection_trace(by_collection: dict[str, list[dict]]) -> dict:
    return {
        "collections": [
            {
                "name": name,
                "hit_count": len(rows),
                "top_hits": [
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "score": row["score"],
                        "text_preview": row["text"][:200],
                    }
                    for row in rows[:5]
                ],
            }
            for name, rows in by_collection.items()
        ]
    }


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    rewritten_query: str,
    doc_id_allowlist: list[str] | None,
    on_step: OnStep | None = None,
) -> list[dict]:
    dense_vector = await gateway.embed(role="query_embed", text=rewritten_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_dense", _collection_trace(dense_by_collection))

    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_sparse", _collection_trace(sparse_by_collection))

    merged = rrf_merge(_flatten(dense_by_collection), _flatten(sparse_by_collection))

    if on_step is not None:
        top_candidates = [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "rrf_score": row["rrf_score"],
                "text_preview": row["text"][:200],
            }
            for row in merged[:15]
        ]
        await on_step("rrf_merge", {"candidate_count": len(merged), "top_candidates": top_candidates})

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: all PASS (6 tests total: 4 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py packages/retrieval-api/tests/test_ai_mode_retrieve.py
git commit -m "feat(retrieval-api): emit milvus_dense/milvus_sparse/rrf_merge trace steps from retrieve"
```

---

### Task 4: `rerank_and_prefetch` emits the `rerank` trace step

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/citations.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_rerank_citations.py`

**Interfaces:**
- Consumes: `rerank_module.rerank_top_chunks(gateway, query, candidates, top_n=3)` (unchanged, from Task-independent existing code) — this task does NOT modify `rerank.py`, it observes `rerank_top_chunks`'s result from `citations.py`.
- Produces: `rerank_and_prefetch(gateway, es_client, query: str, candidates: list[dict], on_step: OnStep | None = None) -> tuple[list[dict], dict[str, dict]]` — same return as today; additionally emits `rerank` (`{"considered_count": len(candidates), "top_chunks": [{"chunk_id", "doc_id", "rerank_score", "text"} for each kept chunk]}`, full text, no truncation) after both concurrent calls finish.

- [ ] **Step 1: Write the failing test**

Append to `packages/retrieval-api/tests/test_ai_mode_rerank_citations.py`:

```python
@pytest.mark.asyncio
async def test_rerank_and_prefetch_emits_rerank_step(monkeypatch):
    import retrieval_api.ai_mode.citations as citations_module
    import retrieval_api.ai_mode.rerank as rerank_module

    async def fake_prefetch(es_client, candidates, top_n_docs=20):
        return {"d1": {"masterinfo": {}}}

    async def fake_rerank_top(gateway, query, candidates, top_n=3):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "chunk text", "rerank_score": 0.95}]

    monkeypatch.setattr(citations_module, "prefetch_citations", fake_prefetch)
    monkeypatch.setattr(rerank_module, "rerank_top_chunks", fake_rerank_top)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await rerank_and_prefetch(
        gateway=object(), es_client=object(), query="q",
        candidates=[{"chunk_id": "a", "doc_id": "d1"}], on_step=on_step,
    )

    assert steps == [("rerank", {
        "considered_count": 1,
        "top_chunks": [{"chunk_id": "a", "doc_id": "d1", "rerank_score": 0.95, "text": "chunk text"}],
    })]
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_rerank_citations.py -v`
Expected: FAIL with `TypeError: rerank_and_prefetch() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ai_mode/citations.py`:

```python
import asyncio

import retrieval_api.ai_mode.rerank as rerank_module
from common.es_client import fetch_citations
from retrieval_api.ai_mode.intent import OnStep


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
    gateway, es_client, query: str, candidates: list[dict], on_step: OnStep | None = None
) -> tuple[list[dict], dict[str, dict]]:
    top_chunks, citations = await asyncio.gather(
        rerank_module.rerank_top_chunks(gateway, query, candidates),
        prefetch_citations(es_client, candidates),
    )

    if on_step is not None:
        trace_chunks = [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "rerank_score": c["rerank_score"], "text": c["text"]}
            for c in top_chunks
        ]
        await on_step("rerank", {"considered_count": len(candidates), "top_chunks": trace_chunks})

    return top_chunks, citations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_rerank_citations.py -v`
Expected: all PASS (4 tests total: 3 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/citations.py packages/retrieval-api/tests/test_ai_mode_rerank_citations.py
git commit -m "feat(retrieval-api): emit rerank trace step from rerank_and_prefetch"
```

---

### Task 5: `synthesize` emits the `synthesis_prompt` trace step

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_synthesize.py`

**Interfaces:**
- Produces: `synthesize(gateway, es_client, query: str, top_chunks: list[dict], citations: dict, on_step: OnStep | None = None) -> dict` — same return as today (`{"answer", "citations"}`); additionally emits `synthesis_prompt` (`{"prompt": <full prompt string>}`) right before the `gateway.chat` call.

- [ ] **Step 1: Write the failing test**

Append to `packages/retrieval-api/tests/test_ai_mode_synthesize.py`:

```python
@pytest.mark.asyncio
async def test_synthesize_emits_synthesis_prompt_step(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module

    async def fake_fetch_citations(client, doc_ids):
        return {}

    monkeypatch.setattr(module, "fetch_citations", fake_fetch_citations)

    gateway = AsyncMock()
    gateway.chat.return_value = "Final answer."
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await synthesize(
        gateway, es_client=object(), query="what is cgst",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}}, on_step=on_step,
    )

    assert len(steps) == 1
    step, data = steps[0]
    assert step == "synthesis_prompt"
    assert "what is cgst" in data["prompt"]
    assert "[d1] chunk text" in data["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: FAIL with `TypeError: synthesize() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`:

```python
from common.es_client import fetch_citations
from retrieval_api.ai_mode.intent import OnStep


async def synthesize(
    gateway, es_client, query: str, top_chunks: list[dict], citations: dict, on_step: OnStep | None = None
) -> dict:
    missing_doc_ids = [c["doc_id"] for c in top_chunks if c["doc_id"] not in citations]
    if missing_doc_ids:
        citations = {**citations, **await fetch_citations(es_client, missing_doc_ids)}

    chunk_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in top_chunks)
    prompt = (
        f"Question: {query}\n\nRelevant excerpts:\n{chunk_block}\n\n"
        "Answer the question citing the doc_id in brackets for each claim."
    )

    if on_step is not None:
        await on_step("synthesis_prompt", {"prompt": prompt})

    answer = await gateway.chat(role="synthesis", messages=[{"role": "user", "content": prompt}])

    return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: all PASS (3 tests total: 2 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py packages/retrieval-api/tests/test_ai_mode_synthesize.py
git commit -m "feat(retrieval-api): emit synthesis_prompt trace step from synthesize"
```

---

### Task 6: `run_ai_mode` threads `on_step` through the whole pipeline

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Consumes: the `on_step` param added to `extract_intent`, `resolve_allowlist`, `retrieve`, `rerank_and_prefetch`, `synthesize` in Tasks 1–5.
- Produces: `run_ai_mode(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict` — same return shape as today (`{"ok": True, "answer", "citations"}` or `{"ok": False, "error"}`); forwards `on_step` to every stage call.

**Note:** the three existing tests in `test_ai_mode_pipeline.py` monkeypatch stage functions with fakes that only accept the original positional args (no `on_step`). Since `run_ai_mode` will now always pass `on_step=on_step` as a keyword (even when `None`), those fakes must be updated to accept it or they'll raise `TypeError`. Update all three existing fakes as part of this task (shown in Step 1 below) — this is a required edit, not an optional cleanup.

- [ ] **Step 1: Update existing fakes and add the new test**

Replace the contents of `packages/retrieval-api/tests/test_ai_mode_pipeline.py`:

```python
import pytest

from retrieval_api.ai_mode.pipeline import run_ai_mode


@pytest.mark.asyncio
async def test_run_ai_mode_success_path(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None):
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="original query")

    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}}


@pytest.mark.asyncio
async def test_run_ai_mode_returns_error_on_any_stage_failure(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def failing_extract_intent(gateway, query, on_step=None):
        raise ValueError("SLM did not return valid JSON")

    monkeypatch.setattr(module, "extract_intent", failing_extract_intent)

    result = await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="q")

    assert result == {"ok": False, "error": "SLM did not return valid JSON"}


@pytest.mark.asyncio
async def test_run_ai_mode_succeeds_with_party_only_filter(monkeypatch):
    """Regression test: a party-only filter dict from the SLM must not raise
    ValueError inside resolve_doc_id_allowlist and abort the whole AI Mode run.
    """
    import retrieval_api.ai_mode.pipeline as module

    class FakeESClient:
        index = "test_index"

        async def search(self, index, query, size):
            assert query == {
                "bool": {"must": [{"match": {"otherinfo.partyname.name": "Reliance Industries"}}]}
            }
            return {"hits": {"hits": [{"_source": {"id": "d1"}}]}}

    async def fake_extract_intent(gateway, query, on_step=None):
        return {
            "rewritten_query": "rewritten",
            "intent": "x",
            "filters": {"party": "Reliance Industries"},
        }

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        assert doc_id_allowlist == ["d1"]
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    result = await run_ai_mode(
        gateway=object(), es_client=FakeESClient(), milvus_client=object(), query="original query"
    )

    assert result["ok"] is True
    assert result == {"ok": True, "answer": "final answer", "citations": {"d1": {}}}


@pytest.mark.asyncio
async def test_run_ai_mode_forwards_on_step_to_every_stage(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    received_on_steps = []

    async def fake_extract_intent(gateway, query, on_step=None):
        received_on_steps.append(("extract_intent", on_step))
        return {"rewritten_query": "rewritten", "intent": "x", "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        received_on_steps.append(("resolve_allowlist", on_step))
        return None

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):
        received_on_steps.append(("retrieve", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None):
        received_on_steps.append(("rerank_and_prefetch", on_step))
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t"}], {"d1": {}}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None):
        received_on_steps.append(("synthesize", on_step))
        return {"answer": "final answer", "citations": citations}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    async def on_step(step, data):
        pass

    await run_ai_mode(gateway=object(), es_client=object(), milvus_client=object(), query="q", on_step=on_step)

    assert received_on_steps == [
        ("extract_intent", on_step),
        ("resolve_allowlist", on_step),
        ("retrieve", on_step),
        ("rerank_and_prefetch", on_step),
        ("synthesize", on_step),
    ]
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: `test_run_ai_mode_forwards_on_step_to_every_stage` FAILS (assertion error — `on_step` not forwarded); the three pre-existing tests should still PASS since the fakes now accept `on_step=None` but `run_ai_mode` doesn't pass it yet, which is fine (default applies).

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`:

```python
from retrieval_api.ai_mode.intent import extract_intent, OnStep
from retrieval_api.ai_mode.filter_resolve import resolve_allowlist
from retrieval_api.ai_mode.retrieve import retrieve
from retrieval_api.ai_mode.citations import rerank_and_prefetch
from retrieval_api.ai_mode.synthesize import synthesize


async def run_ai_mode(gateway, es_client, milvus_client, query: str, on_step: OnStep | None = None) -> dict:
    try:
        intent_result = await extract_intent(gateway, query, on_step=on_step)
        doc_id_allowlist = await resolve_allowlist(es_client, intent_result["filters"], on_step=on_step)
        candidates = await retrieve(
            gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist, on_step=on_step
        )
        top_chunks, citations = await rerank_and_prefetch(gateway, es_client, query, candidates, on_step=on_step)
        synthesis = await synthesize(gateway, es_client, query, top_chunks, citations, on_step=on_step)
        return {"ok": True, "answer": synthesis["answer"], "citations": synthesis["citations"]}
    except Exception as exc:  # noqa: BLE001 - AI Mode failure must never crash Instant's result
        return {"ok": False, "error": str(exc)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: all PASS (4 tests total: 3 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py packages/retrieval-api/tests/test_ai_mode_pipeline.py
git commit -m "feat(retrieval-api): thread on_step callback through run_ai_mode"
```

---

### Task 7: `ws.py` streams `ai_mode_trace` messages with lock-serialized sends

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py`
- Test: `packages/retrieval-api/tests/test_ws_integration.py`

**Interfaces:**
- Consumes: `run_ai_mode(gateway, es_client, milvus_client, query, on_step=None)` from Task 6.
- Produces: over the websocket, a new message type sent zero-or-more times per AI-Mode query, interleaved with `instant_result`: `{"type": "ai_mode_trace", "step": <str>, "data": <dict>}`.

**Note:** the five existing tests in `test_ws_integration.py` monkeypatch `ws_module.run_ai_mode` with fakes that only accept `(gateway, es_client, milvus_client, query)`. Since `ws.py` will now always call `run_ai_mode(..., on_step=emit)`, those fakes must accept `on_step=None` too, or they'll raise `TypeError`. Update all five as part of this task.

- [ ] **Step 1: Update existing fakes and add new tests**

Replace the contents of `packages/retrieval-api/tests/test_ws_integration.py`:

```python
# packages/retrieval-api/tests/test_ws_integration.py
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.ws as ws_module


def test_ws_search_sends_instant_then_ai_mode_events(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "final answer", "citations": {"d1": {}}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "tax exemption"})

        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first["type"] == "instant_result"
    assert first["es"] == [{"doc_id": "d1"}]
    assert second == {"type": "ai_mode_done", "answer": "final answer", "citations": {"d1": {}}}


def test_ws_search_streams_ai_mode_trace_steps_before_final_answer(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        await on_step("intent", {"query": query, "rewritten_query": "r", "intent": "x", "filters": {}})
        await on_step("filters_resolved", {"filters": {}, "doc_id_count": 0, "doc_id_sample": []})
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})
        instant_msg = websocket.receive_json()
        trace_1 = websocket.receive_json()
        trace_2 = websocket.receive_json()
        final = websocket.receive_json()

    assert instant_msg["type"] == "instant_result"
    assert trace_1 == {
        "type": "ai_mode_trace", "step": "intent",
        "data": {"query": "q", "rewritten_query": "r", "intent": "x", "filters": {}},
    }
    assert trace_2 == {
        "type": "ai_mode_trace", "step": "filters_resolved",
        "data": {"filters": {}, "doc_id_count": 0, "doc_id_sample": []},
    }
    assert final == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}


@pytest.mark.asyncio
async def test_emit_trace_step_swallows_send_errors():
    from retrieval_api.ws import _emit_trace_step

    async def failing_send(payload):
        raise RuntimeError("connection closed")

    await _emit_trace_step(failing_send, "intent", {"foo": "bar"})  # must not raise


def test_ws_search_instant_mode_does_not_emit_trace_steps(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        raise AssertionError("ai_mode should not run in instant-only mode")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "instant"})
        only = websocket.receive_json()

    assert only["type"] == "instant_result"


def test_ws_search_sends_ai_mode_error_event_on_failure(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": False, "error": "gateway unreachable"}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q"})
        websocket.receive_json()  # instant_result
        second = websocket.receive_json()

    assert second == {"type": "ai_mode_error", "error": "gateway unreachable"}


def test_ws_search_still_answers_when_milvus_client_construction_fails(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        assert milvus_client is None
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": None, "milvus_error": "connection refused"}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        assert milvus_client is None
        return {"ok": False, "error": "connection refused"}

    def raise_milvus_unavailable(*_):
        raise ConnectionError("Milvus unavailable")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", raise_milvus_unavailable)
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "cgst"})
        first = websocket.receive_json()
        second = websocket.receive_json()

    assert first == {
        "type": "instant_result", "es": [{"doc_id": "d1"}], "es_error": None,
        "milvus": None, "milvus_error": "connection refused",
    }
    assert second == {"type": "ai_mode_error", "error": "connection refused"}


def test_ws_search_instant_mode_skips_ai_mode(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        return {"es": [{"doc_id": "d1"}], "es_error": None, "milvus": {}, "milvus_error": None}

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        raise AssertionError("ai_mode should not run in instant-only mode")

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "instant"})
        only = websocket.receive_json()

    assert only["type"] == "instant_result"


def test_ws_search_ai_mode_only_skips_instant(monkeypatch):
    async def fake_run_instant(gateway, es_client, milvus_client, query):
        raise AssertionError("instant should not run in ai_mode-only mode")

    async def fake_run_ai_mode(gateway, es_client, milvus_client, query, on_step=None):
        return {"ok": True, "answer": "final answer", "citations": {}}

    monkeypatch.setattr(ws_module, "run_instant", fake_run_instant)
    monkeypatch.setattr(ws_module, "run_ai_mode", fake_run_ai_mode)
    monkeypatch.setattr(ws_module, "get_settings", lambda: object())
    monkeypatch.setattr(ws_module, "get_es_client", lambda *_: AsyncMock())
    monkeypatch.setattr(ws_module, "get_milvus_client", lambda *_: Mock())
    monkeypatch.setattr(ws_module, "get_gateway_client", lambda *_: AsyncMock())

    client = TestClient(app)
    with client.websocket_connect("/ws/search") as websocket:
        websocket.send_json({"query": "q", "mode": "ai_mode"})
        only = websocket.receive_json()

    assert only == {"type": "ai_mode_done", "answer": "final answer", "citations": {}}
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v`
Expected: `test_ws_search_streams_ai_mode_trace_steps_before_final_answer` FAILS (extra `receive_json()` calls time out / mismatch — `ws.py` doesn't call `on_step` yet). `test_emit_trace_step_swallows_send_errors` FAILS at collection (`ImportError: cannot import name '_emit_trace_step'`). Others should still pass since fakes now default `on_step=None` and `ws.py` doesn't pass it, which is a no-op.

- [ ] **Step 3: Implement**

Replace the contents of `packages/retrieval-api/src/retrieval_api/ws.py`:

```python
import asyncio

from fastapi import APIRouter, WebSocket

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.instant.search import run_instant
from retrieval_api.ai_mode.pipeline import run_ai_mode

router = APIRouter()


def get_gateway_client(settings) -> GatewayClient:
    return GatewayClient(base_url=settings.gateway_url)


async def _emit_trace_step(send, step: str, data: dict) -> None:
    """Swallows any exception from `send` (e.g. the client disconnected
    mid-stream) - a dead trace channel must never fail the AI Mode
    pipeline or its final answer."""
    try:
        await send({"type": "ai_mode_trace", "step": step, "data": data})
    except Exception:
        pass


@router.websocket("/ws/search")
async def search(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    query = message["query"]
    mode = message.get("mode", "both")  # "instant" | "ai_mode" | "both"

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
        instant_task = (
            asyncio.create_task(run_instant(gateway, es_client, milvus_client, query))
            if mode in ("instant", "both") else None
        )
        ai_mode_task = (
            asyncio.create_task(run_ai_mode(gateway, es_client, milvus_client, query, on_step=emit_trace_step))
            if mode in ("ai_mode", "both") else None
        )

        if instant_task is not None:
            instant_result = await instant_task
            await send({"type": "instant_result", **instant_result})

        if ai_mode_task is not None:
            ai_mode_result = await ai_mode_task
            if ai_mode_result["ok"]:
                await send({
                    "type": "ai_mode_done", "answer": ai_mode_result["answer"], "citations": ai_mode_result["citations"],
                })
            else:
                await send({"type": "ai_mode_error", "error": ai_mode_result["error"]})

        await websocket.close()
    finally:
        await es_client.close()
        if milvus_client is not None:
            milvus_client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ws_integration.py -v`
Expected: all PASS (8 tests total: 5 existing + 3 new).

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest -q`
Expected: all tests pass (backend total grows from 57 to 57 + 2 + 2 + 2 + 1 + 1 + 1 + 3 = 69).

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ws.py packages/retrieval-api/tests/test_ws_integration.py
git commit -m "feat(retrieval-api): stream ai_mode_trace messages over the search websocket

Serializes instant_result/ai_mode_trace/ai_mode_done/ai_mode_error sends
behind one asyncio.Lock, since ai_mode's trace steps now fire while
instant_task may still be completing."
```

---

### Task 8: `useSearch` accumulates `ai_mode_trace` messages into `traceSteps`

**Files:**
- Modify: `packages/web/src/api/useSearch.ts`
- Test: `packages/web/src/api/useSearch.test.ts`

**Interfaces:**
- Produces (new exported type): `export interface TraceStep { step: string; data: Record<string, unknown> }`
- Produces (added to `SearchState`): `traceSteps: TraceStep[]` — starts as `[]` on `INITIAL_STATE` and on every new `search()` call; appends one entry per `ai_mode_trace` message, in arrival order; never cleared mid-query.

- [ ] **Step 1: Write the failing test**

Append to `packages/web/src/api/useSearch.test.ts`:

```ts
  it('accumulates ai_mode_trace messages into traceSteps, in arrival order', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'intent', data: { rewritten_query: 'r' } }),
      })
    })
    expect(result.current.traceSteps).toEqual([{ step: 'intent', data: { rewritten_query: 'r' } }])

    act(() => {
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'filters_resolved', data: { doc_id_count: 0 } }),
      })
    })
    expect(result.current.traceSteps).toEqual([
      { step: 'intent', data: { rewritten_query: 'r' } },
      { step: 'filters_resolved', data: { doc_id_count: 0 } },
    ])
  })

  it('resets traceSteps to empty when a new search starts', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('first query')
    })
    let socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'intent', data: {} }),
      })
    })
    expect(result.current.traceSteps).toHaveLength(1)

    act(() => {
      result.current.search('second query')
    })
    expect(result.current.traceSteps).toEqual([])
  })
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts`
Expected: FAIL — `result.current.traceSteps` is `undefined`.

- [ ] **Step 3: Implement**

Replace the contents of `packages/web/src/api/useSearch.ts`:

```ts
import { useCallback, useRef, useState } from 'react'
import type { EsHit, MilvusByCollection } from '../lib/mergeResults'

export type AiModeCitation = Record<string, unknown>

export interface InstantResult {
  es: EsHit[] | null
  es_error: string | null
  milvus: MilvusByCollection | null
  milvus_error: string | null
}

export type AiModeResult =
  | { ok: true; answer: string; citations: Record<string, AiModeCitation> }
  | { ok: false; error: string }

export interface TraceStep {
  step: string
  data: Record<string, unknown>
}

export interface SearchState {
  /** true from search() until ai_mode_done/ai_mode_error/an error/close arrives - tracks AI Mode specifically, not the Documents feed (that only depends on `instant`). */
  loading: boolean
  instant: InstantResult | null
  aiMode: AiModeResult | null
  traceSteps: TraceStep[]
  wsError: string | null
}

const INITIAL_STATE: SearchState = { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null }

export function useSearch(wsUrl: string): SearchState & { search: (query: string) => void } {
  const [state, setState] = useState<SearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (query: string) => {
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
        socket.send(JSON.stringify({ query, mode: 'both' }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'instant_result') {
          setState((prev) => ({
            ...prev,
            instant: {
              es: message.es ?? null,
              es_error: message.es_error ?? null,
              milvus: message.milvus ?? null,
              milvus_error: message.milvus_error ?? null,
            },
          }))
        } else if (message.type === 'ai_mode_trace') {
          setState((prev) => ({
            ...prev,
            traceSteps: [...prev.traceSteps, { step: message.step, data: message.data }],
          }))
        } else if (message.type === 'ai_mode_done') {
          setState((prev) => ({
            ...prev,
            loading: false,
            aiMode: { ok: true, answer: message.answer, citations: message.citations ?? {} },
          }))
        } else if (message.type === 'ai_mode_error') {
          setState((prev) => ({ ...prev, loading: false, aiMode: { ok: false, error: message.error } }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the search service failed.' }))
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts`
Expected: all PASS (5 tests total: 3 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/api/useSearch.ts packages/web/src/api/useSearch.test.ts
git commit -m "feat(web): accumulate ai_mode_trace messages into useSearch's traceSteps"
```

---

### Task 9: `TracePanel` component

**Files:**
- Create: `packages/web/src/components/TracePanel.tsx`
- Create: `packages/web/src/components/TracePanel.module.css`
- Test: `packages/web/src/components/TracePanel.test.tsx`

**Interfaces:**
- Consumes: `TraceStep` from `../api/useSearch` (Task 8).
- Produces: `export default function TracePanel({ steps }: { steps: TraceStep[] }): JSX.Element` — renders nothing but a header when `steps.length === 0`; otherwise one card per step in array order.

Step label and one-line summary are derived per step type by a small lookup — unknown/future step names fall back to just the step name as the label and no summary, so the panel never crashes on an unrecognized step.

- [ ] **Step 1: Write the failing test**

Create `packages/web/src/components/TracePanel.test.tsx`:

```tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import TracePanel from './TracePanel'
import type { TraceStep } from '../api/useSearch'

describe('TracePanel', () => {
  it('shows a placeholder when there are no steps yet', () => {
    render(<TracePanel steps={[]} />)
    expect(screen.getByText(/no trace yet/i)).toBeInTheDocument()
  })

  it('renders one card per step, in arrival order, with a summary line', () => {
    const steps: TraceStep[] = [
      { step: 'intent', data: { query: 'cgst', rewritten_query: 'CGST meaning', intent: 'taxation', filters: {} } },
      { step: 'rrf_merge', data: { candidate_count: 42, top_candidates: [] } },
    ]
    render(<TracePanel steps={steps} />)

    const headers = screen.getAllByRole('heading', { level: 3 })
    expect(headers.map((h) => h.textContent)).toEqual(['Intent', 'RRF merge'])
    expect(screen.getByText(/CGST meaning/)).toBeInTheDocument()
    expect(screen.getByText(/42/)).toBeInTheDocument()
  })

  it('truncates long lists to 5 with a Show more button that reveals the rest locally', async () => {
    const user = userEvent.setup()
    const topHits = Array.from({ length: 8 }, (_, i) => ({
      chunk_id: `c${i}`, doc_id: 'd1', score: 1, text_preview: `preview ${i}`,
    }))
    const steps: TraceStep[] = [
      { step: 'milvus_dense', data: { collections: [{ name: 'ruling', hit_count: 8, top_hits: topHits }] } },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText(/preview 4/)).toBeInTheDocument()
    expect(screen.queryByText(/preview 5/)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show 3 more/i }))

    expect(screen.getByText(/preview 5/)).toBeInTheDocument()
    expect(screen.getByText(/preview 7/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/components/TracePanel.test.tsx`
Expected: FAIL — module `./TracePanel` doesn't exist.

- [ ] **Step 3: Implement**

Create `packages/web/src/components/TracePanel.module.css`:

```css
/* src/components/TracePanel.module.css */
.panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.placeholder {
  color: #777;
  font-size: 13px;
}

.card {
  border: 1px solid #e2e2e2;
  border-radius: 8px;
  padding: 12px 16px;
  background: #fafbff;
}

.card h3 {
  margin: 0 0 4px;
  font-size: 14px;
}

.summary {
  color: #555;
  font-size: 13px;
  margin: 0 0 8px;
}

.hitList {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 12px;
  font-family: ui-monospace, monospace;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.showMore {
  margin-top: 6px;
  background: none;
  border: none;
  color: #1a56db;
  cursor: pointer;
  padding: 0;
  font-size: 12px;
}
```

Create `packages/web/src/components/TracePanel.tsx`:

```tsx
// src/components/TracePanel.tsx
import { useState } from 'react'
import type { TraceStep } from '../api/useSearch'
import styles from './TracePanel.module.css'

const STEP_LABELS: Record<string, string> = {
  intent: 'Intent',
  filters_resolved: 'Filters resolved',
  milvus_dense: 'Milvus dense search',
  milvus_sparse: 'Milvus sparse search',
  rrf_merge: 'RRF merge',
  rerank: 'Rerank',
  synthesis_prompt: 'Synthesis prompt',
}

function summarize(step: TraceStep): string {
  const d = step.data as Record<string, any>
  switch (step.step) {
    case 'intent':
      return `"${d.query}" -> "${d.rewritten_query}" (${d.intent})`
    case 'filters_resolved':
      return `${d.doc_id_count} doc(s) matched`
    case 'milvus_dense':
    case 'milvus_sparse': {
      const collections = d.collections ?? []
      const total = collections.reduce((sum: number, c: any) => sum + c.hit_count, 0)
      return `${collections.length} collections, ${total} hits`
    }
    case 'rrf_merge':
      return `${d.candidate_count} candidates merged`
    case 'rerank':
      return `${d.considered_count} considered, top ${d.top_chunks?.length ?? 0} kept`
    case 'synthesis_prompt':
      return `${(d.prompt ?? '').length} chars`
    default:
      return ''
  }
}

function TruncatedHitList({ hits }: { hits: Array<Record<string, any>> }) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? hits : hits.slice(0, 5)
  const remaining = hits.length - visible.length

  return (
    <>
      <ul className={styles.hitList}>
        {visible.map((hit, i) => (
          <li key={hit.chunk_id ?? i}>
            [{hit.score?.toFixed?.(3) ?? hit.score}] {hit.doc_id}: {hit.text_preview ?? hit.text}
          </li>
        ))}
      </ul>
      {remaining > 0 && (
        <button type="button" className={styles.showMore} onClick={() => setExpanded(true)}>
          Show {remaining} more
        </button>
      )}
    </>
  )
}

function StepBody({ step }: { step: TraceStep }) {
  const d = step.data as Record<string, any>
  if (step.step === 'milvus_dense' || step.step === 'milvus_sparse') {
    return (
      <>
        {(d.collections ?? []).map((c: any) => (
          <div key={c.name}>
            <strong>{c.name}</strong> ({c.hit_count})
            <TruncatedHitList hits={c.top_hits ?? []} />
          </div>
        ))}
      </>
    )
  }
  if (step.step === 'rrf_merge') {
    return <TruncatedHitList hits={d.top_candidates ?? []} />
  }
  if (step.step === 'rerank') {
    return <TruncatedHitList hits={d.top_chunks ?? []} />
  }
  if (step.step === 'synthesis_prompt') {
    return <pre className={styles.hitList}>{d.prompt}</pre>
  }
  return null
}

export interface TracePanelProps {
  steps: TraceStep[]
}

export default function TracePanel({ steps }: TracePanelProps) {
  if (steps.length === 0) {
    return <p className={styles.placeholder}>No trace yet — run an AI Mode query to see it here.</p>
  }

  return (
    <div className={styles.panel}>
      {steps.map((step, index) => (
        <section key={`${step.step}-${index}`} className={styles.card}>
          <h3>{STEP_LABELS[step.step] ?? step.step}</h3>
          <p className={styles.summary}>{summarize(step)}</p>
          <StepBody step={step} />
        </section>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/components/TracePanel.test.tsx`
Expected: all PASS (3 tests).

If `@testing-library/user-event` isn't already a dependency, check first:

Run: `cd packages/web && cat package.json | grep user-event`

If missing, install it:

Run: `cd packages/web && npm install -D @testing-library/user-event`

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/components/TracePanel.tsx packages/web/src/components/TracePanel.module.css packages/web/src/components/TracePanel.test.tsx packages/web/package.json packages/web/package-lock.json
git commit -m "feat(web): add TracePanel component for AI Mode pipeline trace steps"
```

---

### Task 10: Wire `TracePanel` into `App.tsx` behind Dev Mode, 2-column layout

**Files:**
- Modify: `packages/web/src/App.tsx`
- Modify: `packages/web/src/App.module.css`
- Test: `packages/web/src/App.test.tsx`

**Interfaces:**
- Consumes: `traceSteps` from `useSearch` (Task 8), `TracePanel` (Task 9).

Layout rule: the app shows the 2-column trace layout when Dev Mode is on **and** `traceSteps.length > 0` (there is a trace to show). The existing app always sends `mode: "both"` (see `useSearch.ts`) — there's no per-query Instant-vs-AI-Mode selector in this UI — so gating on "a trace exists" is the concrete, testable equivalent of "AI Mode ran" for this codebase, without inventing a mode selector that doesn't exist elsewhere in the app.

- [ ] **Step 1: Write the failing test**

Append to `packages/web/src/App.test.tsx`:

```tsx
import { vi } from 'vitest'

vi.mock('./api/useSearch', () => ({
  useSearch: () => ({
    loading: false,
    instant: null,
    aiMode: null,
    traceSteps: [{ step: 'intent', data: { query: 'q', rewritten_query: 'q', intent: 'x', filters: {} } }],
    wsError: null,
    search: () => {},
  }),
}))

describe('App with a trace', () => {
  it('shows the TracePanel in a two-column layout when dev mode is on', () => {
    window.history.pushState({}, '', '/?dev=1')
    render(<App />)
    expect(screen.getByText(/Intent/)).toBeInTheDocument()
  })
})
```

Note: this test file will need two separate `describe` blocks using different mocking strategies for `useSearch` (the existing test doesn't mock it at all, relying on real `useSearch` with no query yet). Since Vitest hoists `vi.mock` to the top of the file and applies it module-wide, verify in Step 2 whether the existing "renders the page title" test still passes with the mock in place — if `useSearch`'s mock breaks it, adjust the mock's returned `search` to a no-op (already done above) so `SearchBar` and the header render exactly as before.

- [ ] **Step 2: Run tests to verify current state**

Run: `cd packages/web && npx vitest run src/App.test.tsx`
Expected: the existing "renders the page title" test still PASSES (mock doesn't change header rendering); the new "shows the TracePanel" test FAILS (no `TracePanel` wired in yet, so `?dev=1` alone doesn't render "Intent").

- [ ] **Step 3: Implement**

Replace the contents of `packages/web/src/App.tsx`:

```tsx
// src/App.tsx
import { useEffect, useState } from 'react'
import SearchBar from './components/SearchBar'
import OverviewCard from './components/OverviewCard'
import DocumentsFeed from './components/DocumentsFeed'
import DevModeToggle from './components/DevModeToggle'
import TracePanel from './components/TracePanel'
import { useSearch } from './api/useSearch'
import styles from './App.module.css'

function resolveWsUrl(): string {
  const fromEnv = window.__ENV__?.WS_URL
  return fromEnv && fromEnv.length > 0 ? fromEnv : 'ws://localhost:8010/ws/search'
}

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function App() {
  const wsUrl = resolveWsUrl()
  const { instant, aiMode, traceSteps, loading, wsError, search } = useSearch(wsUrl)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [highlightedDocId, setHighlightedDocId] = useState<string | null>(null)

  useEffect(() => {
    if (highlightedDocId === null) return
    const timeout = window.setTimeout(() => setHighlightedDocId(null), 2000)
    return () => window.clearTimeout(timeout)
  }, [highlightedDocId])

  function handleCitationClick(docId: string) {
    setHighlightedDocId(docId)
    document.getElementById(`document-${docId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const showTrace = devMode && traceSteps.length > 0

  const mainContent = (
    <div>
      <OverviewCard aiMode={aiMode} loading={loading} onCitationClick={handleCitationClick} />
      <DocumentsFeed instant={instant} aiMode={aiMode} devMode={devMode} highlightedDocId={highlightedDocId} />
    </div>
  )

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Taxmann Retrieval</h1>
        <DevModeToggle devMode={devMode} onToggle={setDevMode} />
      </header>
      <SearchBar onSearch={search} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      {showTrace ? (
        <div className={styles.splitLayout}>
          {mainContent}
          <aside className={styles.tracePane}>
            <h2>AI Mode trace</h2>
            <TracePanel steps={traceSteps} />
          </aside>
        </div>
      ) : (
        mainContent
      )}
    </div>
  )
}
```

Update `packages/web/src/App.module.css` — add these rules (keep the existing `.page`, `.header`, `.wsError` unchanged):

```css
.splitLayout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

.tracePane {
  position: sticky;
  top: 24px;
  max-height: calc(100vh - 48px);
  overflow-y: auto;
}

.tracePane h2 {
  margin-top: 0;
}
```

Also widen `.page`'s `max-width` so the 2-column layout has room — change `max-width: 800px;` to `max-width: 1400px;` in the existing `.page` rule (single-column view still reads fine narrower, but the trace layout needs the space).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/web && npx vitest run src/App.test.tsx`
Expected: both tests PASS.

- [ ] **Step 5: Run the full frontend and backend suites**

Run: `cd packages/web && npx vitest run`
Expected: all pass (frontend total grows from 28 to 28 + 2 (useSearch) + 3 (TracePanel) + 1 (App) = 34).

Run: `uv run pytest -q` (from repo root)
Expected: 68 passed (per Task 7).

- [ ] **Step 6: Manual verification**

Rebuild and check the live app (Docker stack should already be running per prior session — `docker compose up -d --build` if not):

Run: `docker compose up -d --build web retrieval-api model-gateway`

Navigate to `http://localhost:8501/?dev=1`, submit a query (e.g. "what is cgst"), and confirm:
- Layout splits into two columns once AI Mode's first trace step arrives.
- Trace cards appear one at a time, in order (`intent` first, `synthesis_prompt` last), not all at once.
- Each Milvus step's "Show N more" button reveals more hits without a network request (open browser dev tools Network tab to confirm no new requests fire on click).
- Turning Dev Mode off collapses back to the single-column layout.

- [ ] **Step 7: Commit**

```bash
git add packages/web/src/App.tsx packages/web/src/App.module.css packages/web/src/App.test.tsx
git commit -m "feat(web): show AI Mode trace panel in a 2-column Dev Mode layout"
```

---

## Self-review notes

- **Spec coverage:** every trace step in the spec's table (intent, filters_resolved, milvus_dense, milvus_sparse, rrf_merge, rerank, synthesis_prompt) has a task producing it (Tasks 1–5), a pipeline task forwarding it (Task 6), a transport task streaming it (Task 7), and frontend tasks consuming/rendering it (Tasks 8–10). The `asyncio.Lock` concurrency fix from the spec is in Task 7. The Dev-Mode-gated 2-column layout is in Task 10.
- **Type consistency:** `OnStep` is defined once in `intent.py` (Task 1) and imported everywhere else that needs the type annotation (`filter_resolve.py`, `retrieve.py`, `citations.py`, `synthesize.py`, `pipeline.py`) — no re-definitions. Frontend `TraceStep` is defined once in `useSearch.ts` (Task 8) and imported by `TracePanel.tsx` (Task 9) and `App.tsx` (Task 10).
- **Scope:** single cohesive feature (one pipeline, one panel), not decomposed further — each task is independently testable and committed, but they build toward one deliverable.
