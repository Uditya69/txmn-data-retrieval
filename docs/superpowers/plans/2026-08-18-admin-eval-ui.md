# Admin Eval UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local-only, token-gated `/admin` page in `packages/web` that runs any of the 4 existing eval suites (slm_intent, intent, collection_routing, retrieval) and shows live progress (% + running pass/fail tally) and full per-case results as the run streams in, with zero third-party log noise.

**Architecture:** A new `retrieval_api/admin_eval/` package holds one thin async-generator adapter per suite (each reusing that suite's existing `load_*`/`check_*`/`evaluate_case` functions unmodified) behind a `SUITES` registry, exposed over a single `/ws/admin-eval` WebSocket route (auth + suite id + limit in the first message, matching `/ws/search`'s existing convention) plus a `/admin/api/eval-runs/{suite}` cache-read endpoint. The frontend is a small, router-free `/admin` page (branched on `window.location.pathname` in `main.tsx`, no new dependency) with a login gate, a suite picker, and a live-updating progress bar + table fed by a WS hook mirroring the existing `useSearch`/`useAgentSearch` pattern.

**Tech Stack:** FastAPI WebSocket route (Python 3.11), existing `GatewayClient`/`common.es_client`/`common.milvus_client`; React 18 + Vite + Tailwind (no router library, matches existing app), Vitest + Testing Library for frontend tests, pytest + `TestClient` for backend tests.

**Spec:** `docs/superpowers/specs/2026-08-18-admin-eval-ui-design.md`

## Global Constraints

- `ADMIN_SECRET` unset (default `None`) disables the admin feature entirely — both the WS route and the cache-read endpoint reject unconditionally in that case.
- The WS route takes `token`/`suite`/`limit` from the **first `receive_json()` message**, never a URL query param (matches `/ws/search`'s existing convention — keeps secrets out of URLs/logs).
- No suite's existing `load_*`/`check_*`/`evaluate_case`/pipeline logic is modified — adapters call these functions as-is.
- `retrieval_eval`'s adapter is materially different from the other 3: its own `--limit` CLI flag means per-stage search depth, not case count (the admin "cap to N cases" control does its own `cases[:limit]` slice instead); it needs real `es_client`/`milvus_client`, not just a `GatewayClient`; it defaults to `skip_agentic=True, skip_synthesis=True` for a tractable admin-UI runtime.
- A client disconnect mid-run cancels the run — it does not keep running server-side (no detached background task; see spec's WS route section for why).
- No queueing: a second run request for a suite already running is rejected (`already_running`), not queued.
- No DB, no persisted run history beyond the last completed run per suite, held in memory.

---

## File Structure

Backend (`packages/retrieval-api/src/retrieval_api/`):
- `admin_eval/__init__.py` — empty, marks the package
- `admin_eval/auth.py` — `is_valid_admin_token(token) -> bool`
- `admin_eval/adapters/__init__.py` — empty
- `admin_eval/adapters/slm_intent.py` — adapter for `slm_intent_eval.py`
- `admin_eval/adapters/intent.py` — adapter for `intent_eval.py`
- `admin_eval/adapters/collection_routing.py` — adapter for `collection_routing_eval.py`
- `admin_eval/adapters/retrieval.py` — adapter for `retrieval_eval.py`
- `admin_eval/registry.py` — `SUITES` dict + Langfuse logger silencing
- `admin_eval/router.py` — WS route + cache-read REST endpoint
- `main.py` — modified to include the new router

Backend tests (`packages/retrieval-api/tests/`):
- `test_admin_eval_auth.py`
- `test_admin_eval_adapter_slm_intent.py`
- `test_admin_eval_adapter_intent.py`
- `test_admin_eval_adapter_collection_routing.py`
- `test_admin_eval_adapter_retrieval.py`
- `test_admin_eval_registry.py`
- `test_admin_eval_router.py`

Frontend (`packages/web/src/`):
- `lib/adminAuth.ts` — sessionStorage helpers for the admin token
- `lib/config.ts` — modified, add `resolveAdminWsUrl`
- `admin/useAdminEvalRun.ts` — WS hook
- `admin/AdminLogin.tsx` — token entry form
- `admin/SuiteRunner.tsx` — suite picker + progress bar + live table
- `admin/AdminApp.tsx` — orchestrator (login gate → runner)
- `main.tsx` — modified, branch on pathname

Frontend tests (co-located, matching this repo's existing `*.test.ts(x)` convention):
- `lib/adminAuth.test.ts`
- `lib/config.test.ts` — modified, add cases for `resolveAdminWsUrl`
- `admin/useAdminEvalRun.test.ts`
- `admin/AdminLogin.test.tsx`
- `admin/SuiteRunner.test.tsx`
- `admin/AdminApp.test.tsx`

Also: `common/config.py` — modified, add `admin_secret` field.

---

### Task 1: Config field + admin token check

**Files:**
- Modify: `packages/common/src/common/config.py`
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/__init__.py`
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/auth.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_auth.py`

**Interfaces:**
- Produces: `common.config.Settings.admin_secret: str | None` (default `None`); `retrieval_api.admin_eval.auth.is_valid_admin_token(token: str | None) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_auth.py
import pytest

from retrieval_api.admin_eval.auth import is_valid_admin_token


def test_rejects_when_admin_secret_unset(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": None})())
    assert is_valid_admin_token("anything") is False


def test_rejects_wrong_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token("wrong-secret") is False


def test_accepts_matching_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token("correct-secret") is True


def test_rejects_none_token(monkeypatch):
    import common.config as config_module
    monkeypatch.setattr(config_module, "get_settings", lambda: type("S", (), {"admin_secret": "correct-secret"})())
    assert is_valid_admin_token(None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.admin_eval'`

- [ ] **Step 3: Add the config field**

Edit `packages/common/src/common/config.py`, add inside `Settings`:

```python
    # Gates the local-only admin eval-runner UI (retrieval_api/admin_eval/) - unset
    # (the default) disables that feature entirely, so no deployment needs to think
    # about it unless it opts in.
    admin_secret: str | None = None
```

- [ ] **Step 4: Create the package and auth module**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/__init__.py
```//empty

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/auth.py
from common.config import get_settings


def is_valid_admin_token(token: str | None) -> bool:
    """True only when ADMIN_SECRET is set AND token matches it exactly. A pure
    predicate (not exception-raising) so both the WS route (needs a custom close
    code, not an HTTPException) and the REST cache-read endpoint (needs an
    HTTPException) can each decide their own rejection shape."""
    settings = get_settings()
    return bool(settings.admin_secret) and token == settings.admin_secret
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_auth.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/config.py packages/retrieval-api/src/retrieval_api/admin_eval packages/retrieval-api/tests/test_admin_eval_auth.py
git commit -m "feat(admin-eval): add ADMIN_SECRET config + token check

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: slm_intent adapter

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/adapters/__init__.py`
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/adapters/slm_intent.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_adapter_slm_intent.py`

**Interfaces:**
- Consumes: `retrieval_api.slm_intent_eval.{load_cases, check_categories, check_rewrite, check_filters}` (all pre-existing, unmodified); `retrieval_api.ai_mode.intent.extract_intent(gateway, query, model=None) -> dict` (pre-existing)
- Produces: `retrieval_api.admin_eval.adapters.slm_intent.run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]`, yielding `{"type": "case", "id", "query", "status": "pass"|"fail"|"error", "detail": {...}}`, `{"type": "progress", "done", "total", "percent"}`, `{"type": "done", "summary": {"total", "passed"}}`

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_adapter_slm_intent.py
import pytest

import retrieval_api.admin_eval.adapters.slm_intent as adapter


def _cases():
    return [
        {
            "id": "S01", "query": "case law for Ramesh Gupta vs. Income-tax Officer",
            "expect": "confident", "expected_categories": ["caselaws"],
            "expected_filters": {"party": "Ramesh Gupta"}, "rewrite_must_contain": ["Ramesh Gupta"],
        },
        {
            "id": "S02", "query": "what is this", "expect": "vague",
            "expected_categories": [], "expected_filters": {}, "rewrite_must_contain": [],
        },
    ]


@pytest.mark.asyncio
async def test_run_yields_case_progress_and_done_events(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        if "Ramesh Gupta" in query:
            return {"search_query": "Ramesh Gupta vs. Income-tax Officer", "intent": ["caselaws"], "filters": {"party": "Ramesh Gupta"}}
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]

    case_events = [e for e in events if e["type"] == "case"]
    progress_events = [e for e in events if e["type"] == "progress"]
    done_events = [e for e in events if e["type"] == "done"]

    assert [c["id"] for c in case_events] == ["S01", "S02"]
    assert case_events[0]["status"] == "pass"
    assert case_events[1]["status"] == "pass"  # safe-empty categories
    assert progress_events[-1] == {"type": "progress", "done": 2, "total": 2, "percent": 100}
    assert done_events == [{"type": "done", "summary": {"total": 2, "passed": 2}}]


@pytest.mark.asyncio
async def test_run_respects_limit(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", 1)]
    case_events = [e for e in events if e["type"] == "case"]
    assert [c["id"] for c in case_events] == ["S01"]


@pytest.mark.asyncio
async def test_run_yields_error_status_and_continues(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())

    call_count = {"n": 0}

    async def fake_extract_intent(gateway, query, model=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("gateway unreachable")
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "error"
    assert "gateway unreachable" in case_events[0]["detail"]["error"]
    assert case_events[1]["status"] == "pass"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_slm_intent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.admin_eval.adapters'`

- [ ] **Step 3: Write the adapter**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/adapters/__init__.py
```//empty

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/adapters/slm_intent.py
from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.slm_intent_eval import check_categories, check_filters, check_rewrite, load_cases

DATASET_PATH = Path("evals/slm_intent_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    for i, case in enumerate(cases, start=1):
        try:
            result = await extract_intent(gateway, case["query"])
        except Exception as exc:
            yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
        else:
            cat_status = check_categories(case["expected_categories"], result["intent"])
            rewrite_ok, rewrite_reasons = check_rewrite(case, result["search_query"])
            filters_ok = check_filters(case["expected_filters"], result["filters"])
            case_ok = cat_status != "wrong" and rewrite_ok and filters_ok
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "rewrite": result["search_query"], "rewrite_ok": rewrite_ok, "rewrite_reasons": rewrite_reasons,
                    "categories": {"status": cat_status, "expected": case["expected_categories"], "actual": result["intent"]},
                    "filters": {"ok": filters_ok, "expected": case["expected_filters"], "actual": result["filters"]},
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_slm_intent.py -v`
Expected: PASS (3 tests). If `pytest.mark.asyncio` errors with "unknown marker," check `packages/retrieval-api/pyproject.toml`/repo root `pyproject.toml` for `[tool.pytest.ini_options] asyncio_mode = "auto"` (this repo already runs async tests elsewhere, e.g. `test_ai_mode_intent.py` — match its convention, no new config should be needed).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/adapters packages/retrieval-api/tests/test_admin_eval_adapter_slm_intent.py
git commit -m "feat(admin-eval): add slm_intent suite adapter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: intent adapter

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/adapters/intent.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_adapter_intent.py`

**Interfaces:**
- Consumes: `retrieval_api.intent_eval.{load_intent_cases, check_intent_case}` (pre-existing); `retrieval_api.ai_mode.intent.extract_intent`
- Produces: `retrieval_api.admin_eval.adapters.intent.run(gateway_url, limit) -> AsyncIterator[dict]`, same event shape as Task 2

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_adapter_intent.py
import pytest

import retrieval_api.admin_eval.adapters.intent as adapter


def _cases():
    return [
        {
            "id": "F01", "query": "What did the Bombay High Court decide about ITC under Rule 6(3)(c)?",
            "expected_filters": {"court": "Bombay High Court"}, "expected_categories": ["rules", "caselaws"],
        },
    ]


@pytest.mark.asyncio
async def test_run_yields_pass_case(monkeypatch):
    monkeypatch.setattr(adapter, "load_intent_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": ["rules", "caselaws"], "filters": {"court": "Bombay High Court"}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "pass"
    assert events[-1] == {"type": "done", "summary": {"total": 1, "passed": 1}}


@pytest.mark.asyncio
async def test_run_yields_fail_case_on_filter_mismatch(monkeypatch):
    monkeypatch.setattr(adapter, "load_intent_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": ["rules", "caselaws"], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "fail"
    assert case_events[0]["detail"]["filters"]["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_intent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the adapter**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/adapters/intent.py
from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.intent_eval import check_intent_case, load_intent_cases

DATASET_PATH = Path("evals/intent_filter_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_intent_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    for i, case in enumerate(cases, start=1):
        try:
            result = await extract_intent(gateway, case["query"])
        except Exception as exc:
            yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
        else:
            filters_ok, categories_ok = check_intent_case(
                case["expected_filters"], result["filters"], case["expected_categories"], result["intent"],
            )
            case_ok = filters_ok and categories_ok
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "filters": {"ok": filters_ok, "expected": case["expected_filters"], "actual": result["filters"]},
                    "categories": {"ok": categories_ok, "expected": case["expected_categories"], "actual": result["intent"]},
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_intent.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/adapters/intent.py packages/retrieval-api/tests/test_admin_eval_adapter_intent.py
git commit -m "feat(admin-eval): add intent suite adapter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: collection_routing adapter

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/adapters/collection_routing.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_adapter_collection_routing.py`

**Interfaces:**
- Consumes: `retrieval_api.collection_routing_eval.{load_routing_cases, check_routing_case}` (pre-existing); `retrieval_api.ai_mode.intent.extract_intent`
- Produces: `retrieval_api.admin_eval.adapters.collection_routing.run(gateway_url, limit) -> AsyncIterator[dict]`, same event shape as Task 2

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_adapter_collection_routing.py
import pytest

import retrieval_api.admin_eval.adapters.collection_routing as adapter


def _cases():
    return [
        {"id": "R01", "expect": "confident", "query": "case law for Ramesh Gupta vs. Income-tax Officer", "expected_categories": ["caselaws"]},
        {"id": "R02", "expect": "vague", "query": "capital gains", "expected_categories": []},
    ]


@pytest.mark.asyncio
async def test_run_marks_wrong_confident_tag_as_fail(monkeypatch):
    monkeypatch.setattr(adapter, "load_routing_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        if query == "capital gains":
            return {"search_query": query, "intent": ["commentary"], "filters": {}}  # wrong non-empty tag
        return {"search_query": query, "intent": ["caselaws"], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "pass"
    assert case_events[0]["detail"]["outcome"] == "exact"
    assert case_events[1]["status"] == "fail"
    assert case_events[1]["detail"]["outcome"] == "wrong"


@pytest.mark.asyncio
async def test_run_treats_safe_empty_as_pass(monkeypatch):
    monkeypatch.setattr(adapter, "load_routing_cases", lambda path: _cases())

    async def fake_extract_intent(gateway, query, model=None):
        return {"search_query": query, "intent": [], "filters": {}}

    monkeypatch.setattr(adapter, "extract_intent", fake_extract_intent)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert all(c["status"] == "pass" for c in case_events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_collection_routing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the adapter**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/adapters/collection_routing.py
from pathlib import Path
from typing import AsyncIterator

from retrieval_api.ai_mode.intent import extract_intent
from retrieval_api.collection_routing_eval import check_routing_case, load_routing_cases
from retrieval_api.gateway_client import GatewayClient

DATASET_PATH = Path("evals/collection_routing_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_routing_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    for i, case in enumerate(cases, start=1):
        try:
            result = await extract_intent(gateway, case["query"])
        except Exception as exc:
            yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
        else:
            outcome = check_routing_case(case["expected_categories"], result["intent"])
            case_ok = outcome != "wrong"
            passed += case_ok
            yield {
                "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                "detail": {
                    "outcome": outcome, "expect": case["expect"],
                    "expected": case["expected_categories"], "actual": result["intent"],
                },
            }
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_collection_routing.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/adapters/collection_routing.py packages/retrieval-api/tests/test_admin_eval_adapter_collection_routing.py
git commit -m "feat(admin-eval): add collection_routing suite adapter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: retrieval adapter

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/adapters/retrieval.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_adapter_retrieval.py`

**Interfaces:**
- Consumes: `retrieval_api.retrieval_eval.{load_cases, evaluate_case}` (pre-existing, unmodified); `common.config.get_settings`; `common.es_client.get_es_client`; `common.milvus_client.get_milvus_client`; `retrieval_api.gateway_client.GatewayClient`
- Produces: `retrieval_api.admin_eval.adapters.retrieval.run(gateway_url, limit) -> AsyncIterator[dict]`, same event shape as Task 2, `detail` carries `{"ranks", "pass_at", "errors", "timings_ms"}`

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_adapter_retrieval.py
from unittest.mock import AsyncMock, Mock

import pytest

import retrieval_api.admin_eval.adapters.retrieval as adapter


def _cases():
    return [
        {"id": "Q01", "class": "direct", "query": "capital gains exemption section 54F", "gold_doc_ids": ["d1"], "expected_collections": [], "pass_at": 5},
        {"id": "Q02", "class": "direct", "query": "GST refund on export", "gold_doc_ids": ["d2"], "expected_collections": [], "pass_at": 5},
    ]


@pytest.mark.asyncio
async def test_run_uses_reranker_rank_for_pass_fail(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases())
    monkeypatch.setattr(adapter, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())
    fake_es = AsyncMock()
    fake_milvus = Mock()
    monkeypatch.setattr(adapter, "get_es_client", lambda settings: fake_es)
    monkeypatch.setattr(adapter, "get_milvus_client", lambda settings: fake_milvus)

    async def fake_evaluate_case(case, gateway, es_client, milvus_client, **kwargs):
        rank = 2 if case["id"] == "Q01" else None  # Q02 misses (rank None)
        return {"ranks": {"reranker": rank}, "errors": {}, "timings_ms": {}}

    monkeypatch.setattr(adapter, "evaluate_case", fake_evaluate_case)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]

    assert case_events[0]["status"] == "pass"   # rank 2 <= pass_at 5
    assert case_events[1]["status"] == "fail"   # rank None
    assert events[-1] == {"type": "done", "summary": {"total": 2, "passed": 1}}
    fake_es.close.assert_awaited_once()
    fake_milvus.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_closes_clients_even_on_case_error(monkeypatch):
    monkeypatch.setattr(adapter, "load_cases", lambda path: _cases()[:1])
    monkeypatch.setattr(adapter, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())
    fake_es = AsyncMock()
    fake_milvus = Mock()
    monkeypatch.setattr(adapter, "get_es_client", lambda settings: fake_es)
    monkeypatch.setattr(adapter, "get_milvus_client", lambda settings: fake_milvus)

    async def failing_evaluate_case(case, gateway, es_client, milvus_client, **kwargs):
        raise RuntimeError("es unreachable")

    monkeypatch.setattr(adapter, "evaluate_case", failing_evaluate_case)

    events = [event async for event in adapter.run("http://gateway", None)]
    case_events = [e for e in events if e["type"] == "case"]
    assert case_events[0]["status"] == "error"
    fake_es.close.assert_awaited_once()
    fake_milvus.close.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the adapter**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/adapters/retrieval.py
from pathlib import Path
from typing import AsyncIterator

from common.config import get_settings
from common.es_client import get_es_client
from common.milvus_client import get_milvus_client
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.retrieval_eval import evaluate_case, load_cases

DATASET_PATH = Path("evals/retrieval_cases.json")


async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_cases(DATASET_PATH)
    if limit:
        cases = cases[:limit]  # first-N slice - NOT the same as evaluate_case's own `limit` kwarg
    total = len(cases)

    settings = get_settings()
    es_client = get_es_client(settings)
    milvus_client = get_milvus_client(settings)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0

    try:
        for i, case in enumerate(cases, start=1):
            try:
                result = await evaluate_case(
                    case, gateway, es_client, milvus_client,
                    langfuse_enabled=False, skip_agentic=True, skip_synthesis=True,
                )
            except Exception as exc:
                yield {"type": "case", "id": case["id"], "query": case["query"], "status": "error", "detail": {"error": str(exc)}}
            else:
                # "reranker" is the final retrieval-pipeline stage a synthesis call
                # actually consumes - used as the single pass/fail headline signal;
                # every stage's own rank is still exposed in detail for full context.
                rank = result["ranks"]["reranker"]
                case_ok = rank is not None and rank <= case["pass_at"]
                passed += case_ok
                yield {
                    "type": "case", "id": case["id"], "query": case["query"], "status": "pass" if case_ok else "fail",
                    "detail": {
                        "ranks": result["ranks"], "pass_at": case["pass_at"],
                        "errors": result["errors"], "timings_ms": result["timings_ms"],
                    },
                }
            yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}
    finally:
        await es_client.close()
        milvus_client.close()

    yield {"type": "done", "summary": {"total": total, "passed": passed}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_adapter_retrieval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/adapters/retrieval.py packages/retrieval-api/tests/test_admin_eval_adapter_retrieval.py
git commit -m "feat(admin-eval): add retrieval suite adapter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Suite registry + Langfuse log silencing

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/registry.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_registry.py`

**Interfaces:**
- Consumes: all 4 adapters' `run` functions from Tasks 2-5
- Produces: `retrieval_api.admin_eval.registry.SUITES: dict[str, dict]` — each value `{"name": str, "run": Callable}`

- [ ] **Step 1: Write the failing test**

```python
# packages/retrieval-api/tests/test_admin_eval_registry.py
import logging

from retrieval_api.admin_eval.registry import SUITES


def test_all_four_suites_registered():
    assert set(SUITES.keys()) == {"slm_intent", "intent", "collection_routing", "retrieval"}


def test_each_suite_has_a_display_name_and_callable_run():
    for suite_id, suite in SUITES.items():
        assert isinstance(suite["name"], str) and suite["name"]
        assert callable(suite["run"])


def test_langfuse_logger_silenced_on_import():
    assert logging.getLogger("langfuse").level == logging.CRITICAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.admin_eval.registry'`

- [ ] **Step 3: Write the registry**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/registry.py
import logging

from retrieval_api.admin_eval.adapters import collection_routing, intent, retrieval, slm_intent

# The Langfuse SDK's own logger emits "Authentication error"/"Context error"
# warnings on every traced call when LANGFUSE_PUBLIC_KEY isn't set - third-party
# log output, not something the eval scripts print themselves. Silenced here
# (only for runs started through the admin path) rather than touching the eval
# scripts or their CLI behavior.
logging.getLogger("langfuse").setLevel(logging.CRITICAL)

SUITES: dict[str, dict] = {
    "slm_intent": {"name": "SLM Intent, Filters & Rewrite", "run": slm_intent.run},
    "intent": {"name": "Intent + Filters (exact-match)", "run": intent.run},
    "collection_routing": {"name": "Collection Routing", "run": collection_routing.run},
    "retrieval": {"name": "Retrieval Pipeline", "run": retrieval.run},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/registry.py packages/retrieval-api/tests/test_admin_eval_registry.py
git commit -m "feat(admin-eval): add suite registry, silence Langfuse log spam

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: WS route + cache-read endpoint + wire into app

**Files:**
- Create: `packages/retrieval-api/src/retrieval_api/admin_eval/router.py`
- Modify: `packages/retrieval-api/src/retrieval_api/main.py`
- Test: `packages/retrieval-api/tests/test_admin_eval_router.py`

**Interfaces:**
- Consumes: `retrieval_api.admin_eval.auth.is_valid_admin_token`, `retrieval_api.admin_eval.registry.SUITES`, `common.config.get_settings`
- Produces: `retrieval_api.admin_eval.router.router: APIRouter` (WS `/ws/admin-eval`, GET `/admin/api/eval-runs/{suite}`); mounted into `retrieval_api.main.app`

- [ ] **Step 1: Write the failing tests**

```python
# packages/retrieval-api/tests/test_admin_eval_router.py
import pytest
from fastapi.testclient import TestClient

from retrieval_api.main import app
import retrieval_api.admin_eval.router as router_module


@pytest.fixture(autouse=True)
def _reset_router_state():
    router_module._running.clear()
    router_module._cache.clear()
    yield
    router_module._running.clear()
    router_module._cache.clear()


async def _fake_suite_run(gateway_url, limit):
    yield {"type": "case", "id": "T1", "query": "q1", "status": "pass", "detail": {}}
    yield {"type": "progress", "done": 1, "total": 1, "percent": 100}
    yield {"type": "done", "summary": {"total": 1, "passed": 1}}


def test_ws_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: False)
    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "slm_intent", "token": "wrong"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "unauthorized"}


def test_ws_rejects_unknown_suite(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "not_a_real_suite", "token": "t"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "unknown_suite"}


def test_ws_rejects_already_running_suite(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _fake_suite_run})
    router_module._running.add("fake_suite")

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        message = ws.receive_json()
        assert message == {"type": "error", "reason": "already_running"}


def test_ws_streams_events_and_populates_cache(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    monkeypatch.setitem(router_module.SUITES, "fake_suite", {"name": "Fake", "run": _fake_suite_run})
    monkeypatch.setattr(router_module, "get_settings", lambda: type("S", (), {"gateway_url": "http://gateway"})())

    client = TestClient(app)
    with client.websocket_connect("/ws/admin-eval") as ws:
        ws.send_json({"suite": "fake_suite", "token": "t"})
        messages = [ws.receive_json() for _ in range(3)]

    assert messages[0]["type"] == "case"
    assert messages[1]["type"] == "progress"
    assert messages[2] == {"type": "done", "summary": {"total": 1, "passed": 1}}
    assert "fake_suite" not in router_module._running
    assert router_module._cache["fake_suite"]["summary"] == {"total": 1, "passed": 1}
    assert router_module._cache["fake_suite"]["cases"] == [messages[0]]


def test_cache_read_returns_null_before_any_run(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "t"})
    assert response.status_code == 200
    assert response.json() is None


def test_cache_read_returns_populated_run(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: True)
    router_module._cache["slm_intent"] = {"summary": {"total": 1, "passed": 1}, "cases": []}
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "t"})
    assert response.status_code == 200
    assert response.json() == {"summary": {"total": 1, "passed": 1}, "cases": []}


def test_cache_read_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(router_module, "is_valid_admin_token", lambda token: False)
    client = TestClient(app)
    response = client.get("/admin/api/eval-runs/slm_intent", headers={"X-Admin-Token": "wrong"})
    assert response.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval_api.admin_eval.router'`

- [ ] **Step 3: Write the router**

```python
# packages/retrieval-api/src/retrieval_api/admin_eval/router.py
from fastapi import APIRouter, Header, HTTPException, WebSocket

from common.config import get_settings
from retrieval_api.admin_eval.auth import is_valid_admin_token
from retrieval_api.admin_eval.registry import SUITES

router = APIRouter()

# Process-local run state - single admin user, single machine (see spec's
# "no queueing" non-goal). _running tracks suite ids currently streaming;
# _cache holds the last completed run per suite for the read-only endpoint.
_running: set[str] = set()
_cache: dict[str, dict] = {}


@router.websocket("/ws/admin-eval")
async def admin_eval(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_json()
    token = message.get("token")
    suite = message.get("suite")
    limit = message.get("limit")

    if not is_valid_admin_token(token):
        await websocket.send_json({"type": "error", "reason": "unauthorized"})
        await websocket.close(code=4403)
        return
    if suite not in SUITES:
        await websocket.send_json({"type": "error", "reason": "unknown_suite"})
        await websocket.close(code=4404)
        return
    if suite in _running:
        await websocket.send_json({"type": "error", "reason": "already_running"})
        await websocket.close(code=4409)
        return

    _running.add(suite)
    gateway_url = get_settings().gateway_url
    cases: list[dict] = []
    try:
        async for event in SUITES[suite]["run"](gateway_url, limit):
            if event["type"] == "case":
                cases.append(event)
            await websocket.send_json(event)
            if event["type"] == "done":
                _cache[suite] = {"summary": event["summary"], "cases": cases}
    finally:
        _running.discard(suite)


@router.get("/admin/api/eval-runs/{suite}")
def get_eval_run(suite: str, x_admin_token: str | None = Header(default=None)):
    if not is_valid_admin_token(x_admin_token):
        raise HTTPException(status_code=403)
    if suite not in SUITES:
        raise HTTPException(status_code=404)
    return _cache.get(suite)
```

- [ ] **Step 4: Wire the router into the app**

Edit `packages/retrieval-api/src/retrieval_api/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.router import router as auth_router
from retrieval_api.admin_eval.router import router as admin_eval_router
from retrieval_api.ws import router
from retrieval_api.documents import router as documents_router
from retrieval_api.query_analysis import router as query_analysis_router
from retrieval_api.intent_analysis import router as intent_analysis_router
from retrieval_api.ai_mode_analysis import router as ai_mode_analysis_router

app = FastAPI(title="retrieval-api")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)
app.include_router(router)
app.include_router(documents_router)
app.include_router(query_analysis_router)
app.include_router(intent_analysis_router)
app.include_router(ai_mode_analysis_router)
app.include_router(auth_router)
app.include_router(admin_eval_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_admin_eval_router.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Run the full backend suite to check nothing else broke**

Run: `uv run pytest packages/retrieval-api/tests -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/admin_eval/router.py packages/retrieval-api/src/retrieval_api/main.py packages/retrieval-api/tests/test_admin_eval_router.py
git commit -m "feat(admin-eval): add WS route + cache-read endpoint, wire into app

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Frontend — admin token storage

**Files:**
- Create: `packages/web/src/lib/adminAuth.ts`
- Test: `packages/web/src/lib/adminAuth.test.ts`

**Interfaces:**
- Produces: `getStoredAdminToken(): string | null`, `setStoredAdminToken(token: string): void`, `clearStoredAdminToken(): void`

- [ ] **Step 1: Write the failing test**

```typescript
// packages/web/src/lib/adminAuth.test.ts
import { describe, expect, it, beforeEach } from 'vitest'
import { getStoredAdminToken, setStoredAdminToken, clearStoredAdminToken } from './adminAuth'

describe('adminAuth', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('returns null when nothing stored', () => {
    expect(getStoredAdminToken()).toBeNull()
  })

  it('stores and retrieves a token', () => {
    setStoredAdminToken('secret-123')
    expect(getStoredAdminToken()).toBe('secret-123')
  })

  it('clears a stored token', () => {
    setStoredAdminToken('secret-123')
    clearStoredAdminToken()
    expect(getStoredAdminToken()).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/lib/adminAuth.test.ts`
Expected: FAIL - cannot find module `./adminAuth`

- [ ] **Step 3: Write the implementation**

```typescript
// packages/web/src/lib/adminAuth.ts
const ADMIN_TOKEN_KEY = 'taxmann-admin-token'

// sessionStorage (not localStorage) deliberately - the admin token shouldn't
// outlive the browser tab; each new admin session re-enters it.
export function getStoredAdminToken(): string | null {
  return sessionStorage.getItem(ADMIN_TOKEN_KEY)
}

export function setStoredAdminToken(token: string): void {
  sessionStorage.setItem(ADMIN_TOKEN_KEY, token)
}

export function clearStoredAdminToken(): void {
  sessionStorage.removeItem(ADMIN_TOKEN_KEY)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/lib/adminAuth.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/lib/adminAuth.ts packages/web/src/lib/adminAuth.test.ts
git commit -m "feat(admin-eval): add admin token sessionStorage helpers

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Frontend — admin WS URL resolution

**Files:**
- Modify: `packages/web/src/lib/config.ts`
- Modify: `packages/web/src/lib/config.test.ts` (create if it doesn't already exist)

**Interfaces:**
- Produces: `resolveAdminWsUrl(apiBaseUrl: string): string`

- [ ] **Step 1: Check for an existing config test file**

Run: `ls packages/web/src/lib/config.test.ts 2>/dev/null || echo "no existing test file"`

If it exists, read it first and add the new test alongside the existing ones, keeping its existing style. If not, create it fresh as below.

- [ ] **Step 2: Write the failing test**

```typescript
// packages/web/src/lib/config.test.ts (add this describe block; keep any existing tests in the file)
import { describe, expect, it } from 'vitest'
import { resolveAdminWsUrl } from './config'

describe('resolveAdminWsUrl', () => {
  it('derives the admin WS url from an http api base url', () => {
    expect(resolveAdminWsUrl('http://localhost:8010')).toBe('ws://localhost:8010/ws/admin-eval')
  })

  it('derives the admin WS url from an https api base url', () => {
    expect(resolveAdminWsUrl('https://api.example.com')).toBe('wss://api.example.com/ws/admin-eval')
  })
})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/lib/config.test.ts`
Expected: FAIL - `resolveAdminWsUrl` is not exported

- [ ] **Step 4: Add the implementation**

Append to `packages/web/src/lib/config.ts`:

```typescript
export function resolveAdminWsUrl(apiBaseUrl: string): string {
  return `${apiBaseUrl.replace(/^http/, 'ws')}/ws/admin-eval`
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/lib/config.test.ts`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/web/src/lib/config.ts packages/web/src/lib/config.test.ts
git commit -m "feat(admin-eval): add resolveAdminWsUrl

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: Frontend — WS run hook

**Files:**
- Create: `packages/web/src/admin/useAdminEvalRun.ts`
- Test: `packages/web/src/admin/useAdminEvalRun.test.ts`

**Interfaces:**
- Produces: `useAdminEvalRun(wsUrl: string): AdminEvalState & { run: (suite: string, token: string, limit?: number) => void; loadCached: (apiBaseUrl: string, suite: string, token: string) => Promise<void> }` where `AdminEvalState = { running: boolean; percent: number; total: number; passed: number; cases: CaseEvent[]; error: string | null }` and `CaseEvent = { type: 'case'; id: string; query: string; status: 'pass' | 'fail' | 'error'; detail: Record<string, unknown> }`. `loadCached` hits `GET {apiBaseUrl}/admin/api/eval-runs/{suite}` (spec's cache-read endpoint) and hydrates `cases`/`percent`/`total`/`passed` from a prior completed run **without** setting `running: true` — this is what makes a completed run survive a page refresh or a suite switch, per the spec's "Survive a page refresh" goal.

- [ ] **Step 1: Write the failing test**

```typescript
// packages/web/src/admin/useAdminEvalRun.test.ts
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useAdminEvalRun } from './useAdminEvalRun'

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
    if (type === 'open') {
      this.onopen = handler
      handler()
    }
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

describe('useAdminEvalRun', () => {
  it('sends suite/token/limit and accumulates case/progress events', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))

    act(() => result.current.run('slm_intent', 'tok', 10))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0])).toEqual({ suite: 'slm_intent', token: 'tok', limit: 10 })
    expect(result.current.running).toBe(true)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'case', id: 'S01', query: 'q', status: 'pass', detail: {} }) })
    })
    expect(result.current.cases).toEqual([{ type: 'case', id: 'S01', query: 'q', status: 'pass', detail: {} }])
    expect(result.current.passed).toBe(1)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'progress', done: 1, total: 2, percent: 50 }) })
    })
    expect(result.current.percent).toBe(50)
    expect(result.current.total).toBe(2)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'done', summary: { total: 2, passed: 1 } }) })
    })
    await waitFor(() => expect(result.current.running).toBe(false))
  })

  it('surfaces a server error event and stops running', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    act(() => result.current.run('slm_intent', 'bad-token'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })
    })
    await waitFor(() => expect(result.current.running).toBe(false))
    expect(result.current.error).toBe('unauthorized')
  })

  it('marks the run interrupted if the socket closes while still running', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    act(() => result.current.run('slm_intent', 'tok'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onclose?.()
    })
    await waitFor(() => expect(result.current.running).toBe(false))
    expect(result.current.error).toBe('Run interrupted.')
  })

  it('loadCached hydrates state from a prior run without setting running', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        summary: { total: 2, passed: 1 },
        cases: [
          { type: 'case', id: 'S01', query: 'q1', status: 'pass', detail: {} },
          { type: 'case', id: 'S02', query: 'q2', status: 'fail', detail: {} },
        ],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    await act(async () => {
      await result.current.loadCached('http://x', 'slm_intent', 'tok')
    })

    expect(fetchMock).toHaveBeenCalledWith('http://x/admin/api/eval-runs/slm_intent', {
      headers: { 'X-Admin-Token': 'tok' },
    })
    expect(result.current.running).toBe(false)
    expect(result.current.cases).toHaveLength(2)
    expect(result.current.total).toBe(2)
    expect(result.current.passed).toBe(1)
    expect(result.current.percent).toBe(100)
  })

  it('loadCached leaves state untouched when no run has completed yet (null response)', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => null })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    await act(async () => {
      await result.current.loadCached('http://x', 'slm_intent', 'tok')
    })

    expect(result.current.cases).toEqual([])
    expect(result.current.total).toBe(0)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/admin/useAdminEvalRun.test.ts`
Expected: FAIL - cannot find module `./useAdminEvalRun`

- [ ] **Step 3: Write the hook**

```typescript
// packages/web/src/admin/useAdminEvalRun.ts
import { useCallback, useRef, useState } from 'react'

export interface CaseEvent {
  type: 'case'
  id: string
  query: string
  status: 'pass' | 'fail' | 'error'
  detail: Record<string, unknown>
}

export interface AdminEvalState {
  running: boolean
  percent: number
  total: number
  passed: number
  cases: CaseEvent[]
  error: string | null
}

const INITIAL_STATE: AdminEvalState = { running: false, percent: 0, total: 0, passed: 0, cases: [], error: null }

export function useAdminEvalRun(
  wsUrl: string,
): AdminEvalState & {
  run: (suite: string, token: string, limit?: number) => void
  loadCached: (apiBaseUrl: string, suite: string, token: string) => Promise<void>
} {
  const [state, setState] = useState<AdminEvalState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  // Hydrates from GET /admin/api/eval-runs/{suite} (the spec's cache-read
  // endpoint) - lets a completed run survive a page refresh or a suite switch
  // without starting a new WS run. Deliberately does not touch `running`.
  const loadCached = useCallback(async (apiBaseUrl: string, suite: string, token: string) => {
    const response = await fetch(`${apiBaseUrl}/admin/api/eval-runs/${suite}`, {
      headers: { 'X-Admin-Token': token },
    })
    if (!response.ok) return
    const cached = await response.json()
    if (!cached) return
    setState((prev) => ({
      ...prev,
      cases: cached.cases,
      total: cached.summary.total,
      passed: cached.summary.passed,
      percent: 100,
    }))
  }, [])

  const run = useCallback(
    (suite: string, token: string, limit?: number) => {
      socketRef.current?.close()
      setState({ ...INITIAL_STATE, running: true })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, running: false, error: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ suite, token, limit }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'case') {
          setState((prev) => ({
            ...prev,
            cases: [...prev.cases, message as CaseEvent],
            passed: prev.passed + (message.status === 'pass' ? 1 : 0),
          }))
        } else if (message.type === 'progress') {
          setState((prev) => ({ ...prev, percent: message.percent, total: message.total }))
        } else if (message.type === 'done') {
          setState((prev) => ({ ...prev, running: false }))
        } else if (message.type === 'error') {
          setState((prev) => ({ ...prev, running: false, error: message.reason }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, running: false, error: 'Connection failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.running ? { ...prev, running: false, error: 'Run interrupted.' } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, run, loadCached }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/admin/useAdminEvalRun.test.ts`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/admin/useAdminEvalRun.ts packages/web/src/admin/useAdminEvalRun.test.ts
git commit -m "feat(admin-eval): add useAdminEvalRun WS hook

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: Frontend — login gate component

**Files:**
- Create: `packages/web/src/admin/AdminLogin.tsx`
- Test: `packages/web/src/admin/AdminLogin.test.tsx`

**Interfaces:**
- Consumes: nothing new
- Produces: `AdminLogin({ onSubmit: (token: string) => void; error: string | null }): JSX.Element`

- [ ] **Step 1: Write the failing test**

```tsx
// packages/web/src/admin/AdminLogin.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AdminLogin from './AdminLogin'

describe('AdminLogin', () => {
  it('submits the entered token', async () => {
    const onSubmit = vi.fn()
    render(<AdminLogin onSubmit={onSubmit} error={null} />)

    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'my-secret')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    expect(onSubmit).toHaveBeenCalledWith('my-secret')
  })

  it('shows an error message when provided', () => {
    render(<AdminLogin onSubmit={vi.fn()} error="Invalid token." />)
    expect(screen.getByText('Invalid token.')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/admin/AdminLogin.test.tsx`
Expected: FAIL - cannot find module `./AdminLogin`

- [ ] **Step 3: Write the component**

```tsx
// packages/web/src/admin/AdminLogin.tsx
import { useState } from 'react'

interface AdminLoginProps {
  onSubmit: (token: string) => void
  error: string | null
}

export default function AdminLogin({ onSubmit, error }: AdminLoginProps) {
  const [token, setToken] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onSubmit(token)
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-sm mx-auto mt-24 flex flex-col gap-3">
      <h1 className="text-lg font-semibold">Admin</h1>
      <input
        type="password"
        value={token}
        onChange={(e) => setToken(e.target.value)}
        placeholder="Admin token"
        className="border rounded px-3 py-2"
        autoFocus
      />
      {error && <p className="text-sm" style={{ color: 'crimson' }}>{error}</p>}
      <button type="submit" className="border rounded px-3 py-2 font-medium">
        Enter
      </button>
    </form>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/admin/AdminLogin.test.tsx`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/admin/AdminLogin.tsx packages/web/src/admin/AdminLogin.test.tsx
git commit -m "feat(admin-eval): add AdminLogin component

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 12: Frontend — suite runner (picker + progress + table)

**Files:**
- Create: `packages/web/src/admin/SuiteRunner.tsx`
- Test: `packages/web/src/admin/SuiteRunner.test.tsx`

**Interfaces:**
- Consumes: `useAdminEvalRun` from Task 10 (including `loadCached`)
- Produces: `SuiteRunner({ wsUrl: string; apiBaseUrl: string; token: string; onUnauthorized: () => void }): JSX.Element`. On mount and whenever the selected suite changes, calls `loadCached(apiBaseUrl, suite, token)` before any "Run" click, so a previously-completed run for that suite shows immediately (spec's "survive a page refresh" goal).

- [ ] **Step 1: Write the failing test**

```tsx
// packages/web/src/admin/SuiteRunner.test.tsx
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SuiteRunner from './SuiteRunner'

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
    if (type === 'open') {
      this.onopen = handler
      handler()
    }
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
  // Mount always calls loadCached() first (see spec's "survive a page refresh"
  // goal) - stub a no-prior-run response by default so every test's mount step
  // doesn't need its own fetch mock; the cache-hydration test below overrides this.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => null }))
})

describe('SuiteRunner', () => {
  it('runs the selected suite and renders a case row as it streams in', async () => {
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0]).suite).toBe('slm_intent')

    socket.onmessage?.({
      data: JSON.stringify({ type: 'case', id: 'S01', query: 'case law for X', status: 'pass', detail: { rewrite: 'X' } }),
    })

    expect(await screen.findByText('S01')).toBeInTheDocument()
    expect(screen.getByText('case law for X')).toBeInTheDocument()
    expect(screen.getByText('pass')).toBeInTheDocument()
  })

  it('calls onUnauthorized when the server reports an unauthorized error', async () => {
    const onUnauthorized = vi.fn()
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="bad" onUnauthorized={onUnauthorized} />)

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })

    expect(onUnauthorized).toHaveBeenCalled()
  })

  it('hydrates from a cached run on mount, without opening a WS connection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          summary: { total: 1, passed: 1 },
          cases: [{ type: 'case', id: 'S01', query: 'cached query', status: 'pass', detail: {} }],
        }),
      }),
    )

    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    expect(await screen.findByText('cached query')).toBeInTheDocument()
    expect(FakeWebSocket.instances).toHaveLength(0)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/admin/SuiteRunner.test.tsx`
Expected: FAIL - cannot find module `./SuiteRunner`

- [ ] **Step 3: Write the component**

```tsx
// packages/web/src/admin/SuiteRunner.tsx
import { useEffect, useState } from 'react'
import { useAdminEvalRun } from './useAdminEvalRun'

const SUITES: { id: string; name: string }[] = [
  { id: 'slm_intent', name: 'SLM Intent, Filters & Rewrite' },
  { id: 'intent', name: 'Intent + Filters (exact-match)' },
  { id: 'collection_routing', name: 'Collection Routing' },
  { id: 'retrieval', name: 'Retrieval Pipeline' },
]

interface SuiteRunnerProps {
  wsUrl: string
  apiBaseUrl: string
  token: string
  onUnauthorized: () => void
}

export default function SuiteRunner({ wsUrl, apiBaseUrl, token, onUnauthorized }: SuiteRunnerProps) {
  const evalRun = useAdminEvalRun(wsUrl)
  const [selected, setSelected] = useState(SUITES[0].id)
  const [limit, setLimit] = useState('')

  useEffect(() => {
    if (evalRun.error === 'unauthorized') onUnauthorized()
  }, [evalRun.error, onUnauthorized])

  // Shows the last completed run for the newly-selected suite immediately,
  // before/without starting a new WS run - survives a page refresh and makes
  // switching suites not look like a blank slate if one already ran.
  useEffect(() => {
    evalRun.loadCached(apiBaseUrl, selected, token)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, apiBaseUrl, token])

  return (
    <div className="max-w-4xl mx-auto py-8 flex flex-col gap-4">
      <div className="flex gap-2 flex-wrap">
        {SUITES.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setSelected(s.id)}
            className="px-3 py-1.5 rounded border text-sm"
            style={{ fontWeight: selected === s.id ? 600 : 400 }}
          >
            {s.name}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <input
          type="number"
          min={1}
          value={limit}
          onChange={(e) => setLimit(e.target.value)}
          placeholder="limit (optional)"
          className="border rounded px-2 py-1 w-40 text-sm"
        />
        <button
          type="button"
          onClick={() => evalRun.run(selected, token, limit ? Number(limit) : undefined)}
          disabled={evalRun.running}
          className="border rounded px-3 py-1.5 text-sm font-medium"
        >
          {evalRun.running ? 'Running…' : 'Run'}
        </button>
        {evalRun.error && evalRun.error !== 'unauthorized' && (
          <span className="text-sm" style={{ color: 'crimson' }}>{evalRun.error}</span>
        )}
      </div>

      {evalRun.total > 0 && (
        <div className="flex flex-col gap-1">
          <div className="h-2 rounded bg-gray-200 overflow-hidden">
            <div className="h-full bg-green-600" style={{ width: `${evalRun.percent}%` }} />
          </div>
          <p className="text-xs text-gray-600">
            {evalRun.percent}% · {evalRun.passed}/{evalRun.cases.length} passed of {evalRun.total}
          </p>
        </div>
      )}

      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left border-b">
            <th className="py-1 pr-2">ID</th>
            <th className="py-1 pr-2">Query</th>
            <th className="py-1 pr-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {evalRun.cases.map((c) => (
            <tr key={c.id} className="border-b align-top">
              <td className="py-1 pr-2 font-mono">{c.id}</td>
              <td className="py-1 pr-2">{c.query}</td>
              <td className="py-1 pr-2">
                <span style={{ color: c.status === 'pass' ? 'green' : 'crimson' }}>{c.status}</span>
                <details className="mt-1">
                  <summary className="cursor-pointer text-xs text-gray-500">detail</summary>
                  <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(c.detail, null, 2)}</pre>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/admin/SuiteRunner.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/admin/SuiteRunner.tsx packages/web/src/admin/SuiteRunner.test.tsx
git commit -m "feat(admin-eval): add SuiteRunner (picker + progress + live table)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 13: Frontend — AdminApp orchestrator

**Files:**
- Create: `packages/web/src/admin/AdminApp.tsx`
- Test: `packages/web/src/admin/AdminApp.test.tsx`

**Interfaces:**
- Consumes: `AdminLogin` (Task 11), `SuiteRunner` (Task 12), `getStoredAdminToken`/`setStoredAdminToken`/`clearStoredAdminToken` (Task 8), `resolveWsUrl`/`resolveApiBaseUrl`/`resolveAdminWsUrl` (Task 9 + existing `config.ts`)
- Produces: `AdminApp(): JSX.Element` (default export)

- [ ] **Step 1: Write the failing test**

```tsx
// packages/web/src/admin/AdminApp.test.tsx
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AdminApp from './AdminApp'
import { getStoredAdminToken } from '../lib/adminAuth'

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
    if (type === 'open') {
      this.onopen = handler
      handler()
    }
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
  sessionStorage.clear()
  FakeWebSocket.instances = []
  // @ts-expect-error test stub
  global.WebSocket = FakeWebSocket
  // SuiteRunner (rendered once logged in) always calls loadCached() on mount -
  // stub a no-prior-run response so that doesn't need its own setup per test.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => null }))
})

describe('AdminApp', () => {
  it('shows the login form when no token is stored', () => {
    render(<AdminApp />)
    expect(screen.getByPlaceholderText('Admin token')).toBeInTheDocument()
  })

  it('stores the token and shows the runner after login', async () => {
    render(<AdminApp />)
    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'my-secret')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    expect(getStoredAdminToken()).toBe('my-secret')
    expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
  })

  it('returns to the login form and clears the token when the server reports unauthorized', async () => {
    render(<AdminApp />)
    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'stale-token')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })

    expect(await screen.findByPlaceholderText('Admin token')).toBeInTheDocument()
    expect(getStoredAdminToken()).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/web && npx vitest run src/admin/AdminApp.test.tsx`
Expected: FAIL - cannot find module `./AdminApp`

- [ ] **Step 3: Write the component**

```tsx
// packages/web/src/admin/AdminApp.tsx
import { useState } from 'react'
import AdminLogin from './AdminLogin'
import SuiteRunner from './SuiteRunner'
import { getStoredAdminToken, setStoredAdminToken, clearStoredAdminToken } from '../lib/adminAuth'
import { resolveWsUrl, resolveApiBaseUrl, resolveAdminWsUrl } from '../lib/config'

export default function AdminApp() {
  const [token, setToken] = useState<string | null>(getStoredAdminToken)
  const [loginError, setLoginError] = useState<string | null>(null)

  const apiBaseUrl = resolveApiBaseUrl(resolveWsUrl())
  const adminWsUrl = resolveAdminWsUrl(apiBaseUrl)

  function handleLogin(candidate: string) {
    setStoredAdminToken(candidate)
    setToken(candidate)
    setLoginError(null)
  }

  function handleUnauthorized() {
    clearStoredAdminToken()
    setToken(null)
    setLoginError('Invalid token.')
  }

  if (!token) {
    return <AdminLogin onSubmit={handleLogin} error={loginError} />
  }

  return <SuiteRunner wsUrl={adminWsUrl} apiBaseUrl={apiBaseUrl} token={token} onUnauthorized={handleUnauthorized} />
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/web && npx vitest run src/admin/AdminApp.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/admin/AdminApp.tsx packages/web/src/admin/AdminApp.test.tsx
git commit -m "feat(admin-eval): add AdminApp orchestrator (login gate -> runner)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 14: Wire `/admin` route into main.tsx

**Files:**
- Modify: `packages/web/src/main.tsx`

**Interfaces:**
- Consumes: `AdminApp` from Task 13, existing `App` default export

No new test file — `main.tsx` is a bootstrap entrypoint with no existing test coverage in this repo (confirm: `ls packages/web/src/main.test.tsx` should not exist). Verified manually per Step 3 below instead.

- [ ] **Step 1: Modify main.tsx**

```tsx
// packages/web/src/main.tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import AdminApp from './admin/AdminApp'
import { ErrorBoundary } from './components/ErrorBoundary'
import './index.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('Root container #root not found')
}

// No router dependency in this app (see App.tsx's own URLSearchParams-based
// dev-mode flag for precedent) - a plain pathname branch is enough for one
// extra page.
const isAdminRoute = window.location.pathname.startsWith('/admin')

createRoot(container).render(
  <StrictMode>
    <ErrorBoundary>
      {isAdminRoute ? <AdminApp /> : <App />}
    </ErrorBoundary>
  </StrictMode>,
)
```

- [ ] **Step 2: Run the full frontend test suite to confirm nothing broke**

Run: `cd packages/web && npm test`
Expected: all tests pass (including all admin/* and lib/* tests added in Tasks 8-13)

- [ ] **Step 3: Manual verification**

Run: `cd packages/web && npm run dev`, then in a browser visit `http://localhost:5173/admin` (adjust port to whatever Vite prints) and confirm the login form renders. Visit `http://localhost:5173/` and confirm the normal chat app still renders unchanged.

- [ ] **Step 4: Commit**

```bash
git add packages/web/src/main.tsx
git commit -m "feat(admin-eval): mount AdminApp at /admin

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 15: End-to-end smoke check

No new files - this task verifies the whole feature works together against a real local stack before calling it done.

- [ ] **Step 1: Set ADMIN_SECRET and start the stack**

Add `ADMIN_SECRET=local-dev-secret` to `.env`, then:

```bash
docker compose up -d --build model-gateway retrieval-api
cd packages/web && npm run dev
```

- [ ] **Step 2: Full-suite backend check**

Run: `uv run pytest packages/retrieval-api/tests -q`
Expected: all tests pass, including every `test_admin_eval_*.py` file added in Tasks 1-7

- [ ] **Step 3: Manual browser check**

Visit `/admin`, enter `local-dev-secret`, pick "SLM Intent, Filters & Rewrite", set limit to `5`, click Run. Confirm: progress bar advances, 5 rows appear with pass/fail pills, expanding a row's "detail" shows the full JSON, no console errors, and refreshing the page then re-entering the token shows the login form again (sessionStorage is empty on a hard refresh only if the tab itself was closed - a same-tab refresh should still show the runner immediately, confirm that's the actual behavior observed).

- [ ] **Step 4: Confirm log cleanliness**

While the run from Step 3 is in progress, check the `retrieval-api` container logs (`docker compose logs -f retrieval-api`) - confirm no "Authentication error" / "Context error" Langfuse lines appear.

- [ ] **Step 5: Commit (if Step 1's .env change should be tracked)**

If `.env.example` exists in the repo, add `ADMIN_SECRET=` to it (empty - a real value stays local-only):

```bash
git add .env.example
git commit -m "docs: document ADMIN_SECRET in .env.example

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

If there's no `.env.example` in this repo, skip this step - confirm with `ls .env.example` first.
