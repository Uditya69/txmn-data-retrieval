# AI Mode Chunk-Context Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give AI Mode's `extract_intent` SLM call pre-parsed structural signal (citation/section/court_city/quoted spans) about the query, by calling `chunk_query` and appending a trimmed JSON projection of its output to the user message.

**Architecture:** One new private helper, `_build_chunk_context(query: str) -> str | None`, added to `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`. `extract_intent` calls it once per request and conditionally appends its output to the existing user message content. No other file changes.

**Tech Stack:** Python 3.11, pytest, pytest-asyncio, `common.query_tokenizer.chunk_query` (already implemented, no changes to it).

## Global Constraints

- Design source: `docs/superpowers/specs/2026-08-13-ai-mode-chunk-context-injection-design.md`.
- Drop `proximity` and `alt_text` from every chunk — only `text` and `type` survive the projection.
- Drop any chunk with `type == "text"`.
- Use `chunk_query`'s own type strings verbatim (`citation`, `section`, `court_city`, `quoted`) — no relabeling.
- If the projected list is empty, `_build_chunk_context` returns `None` and the user message is unchanged from today (just the raw query) — never send an empty `[]` block.
- Exact block wording when non-empty (query and JSON block separated by a blank line):
  ```
  {query}

  Structural spans already present in the query above (for reference only — do not add anything not already in the query text):
  {json_block}
  ```
- No change to `chunk_query`, `analyze_query`, `_sanitize_filters`, `_safe_rewrite`, `_validate_result`, `_system_prompt_for_model`, or `_LLAMA_SYSTEM_PROMPT`'s existing content.
- No change to Instant mode.
- No cross-validation/auto-correction logic tying `chunk_query` output to SLM output.

---

### Task 1: `_build_chunk_context` helper + user-message wiring in `extract_intent`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
  - Add import: `from common.query_tokenizer import chunk_query`
  - Add new function `_build_chunk_context` (place after `_fallback_intent`, before `_LLAMA_SYSTEM_PROMPT`, around current line 33)
  - Modify `extract_intent` (currently intent.py:193-219) to build and append the block
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Produces: `_build_chunk_context(query: str) -> str | None`, module-private (leading underscore, same convention as `_sanitize_filters`/`_safe_rewrite` in this file), imported directly by name in the test file per Step 1.
- Consumes (from existing code, unchanged): `common.query_tokenizer.chunk_query(query: str) -> list[dict]`, each dict having keys `text: str`, `type: str` (one of `"text" | "section" | "citation" | "court_city" | "quoted"`), `proximity: int`, `alt_text: str | None`.

- [ ] **Step 1: Write failing unit tests for `_build_chunk_context`**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py` (append at end of file):

```python
from retrieval_api.ai_mode.intent import _build_chunk_context


def test_build_chunk_context_returns_none_for_text_only_query():
    assert _build_chunk_context("capital gains set off business losses") is None


def test_build_chunk_context_projects_and_filters_chunks():
    result = _build_chunk_context('1995 taxmann.com 569 Delhi High Court "capital gains"')

    assert result is not None
    spans = json.loads(result)
    assert isinstance(spans, list)
    for span in spans:
        assert set(span.keys()) == {"text", "type"}
        assert span["type"] != "text"
    types = {span["type"] for span in spans}
    assert "citation" in types
    assert "court_city" in types


def test_build_chunk_context_drops_alt_text_and_proximity():
    result = _build_chunk_context("Rule 57A applicability")

    assert result is not None
    spans = json.loads(result)
    for span in spans:
        assert "alt_text" not in span
        assert "proximity" not in span
```

- [ ] **Step 2: Run new tests, confirm they fail on missing `_build_chunk_context`**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k build_chunk_context -v`
Expected: FAIL — `ImportError: cannot import name '_build_chunk_context'`

- [ ] **Step 3: Implement `_build_chunk_context`**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, add the import at the top (alongside the existing `from common.schema_context import ...` line):

```python
from common.query_tokenizer import chunk_query
```

Add the function after `_fallback_intent` (currently ends at line 32), before `_LLAMA_SYSTEM_PROMPT` starts:

```python
def _build_chunk_context(query: str) -> str | None:
    """Trimmed JSON projection of chunk_query's structural spans, for injection
    into extract_intent's user message. Drops `proximity`/`alt_text` (ES-only,
    and alt_text's normalized form would never literal-match _sanitize_filters'
    substring check against the raw query - see design spec) and any
    type=="text" chunk (a bare word run adds no signal beyond the raw query
    the model already sees). Returns None when nothing structural is found,
    so callers can omit the block entirely rather than send an empty list."""
    spans = [
        {"text": chunk["text"], "type": chunk["type"]}
        for chunk in chunk_query(query)
        if chunk["type"] != "text"
    ]
    if not spans:
        return None
    return json.dumps(spans)
```

- [ ] **Step 4: Run the three new tests, confirm they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k build_chunk_context -v`
Expected: PASS (all 3)

- [ ] **Step 5: Write failing integration tests for `extract_intent`'s user message**

Append to `packages/retrieval-api/tests/test_ai_mode_intent.py`:

```python
@pytest.mark.asyncio
async def test_extract_intent_appends_chunk_context_when_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "1995 taxmann.com 569",
        "intent": "citation_lookup",
        "filters": {},
    })

    await extract_intent(gateway, "1995 taxmann.com 569 Delhi High Court")

    call_kwargs = gateway.chat.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message.startswith("1995 taxmann.com 569 Delhi High Court\n\n")
    assert "Structural spans already present in the query above" in user_message
    assert '"type": "citation"' in user_message
    assert '"type": "court_city"' in user_message


@pytest.mark.asyncio
async def test_extract_intent_user_message_unchanged_when_no_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "rewritten_query": "capital gains treatment",
        "intent": "conceptual",
        "filters": {},
    })

    await extract_intent(gateway, "capital gains treatment")

    call_kwargs = gateway.chat.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message == "capital gains treatment"
```

- [ ] **Step 6: Run the two new integration tests, confirm they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k chunk_context -v`
Expected: the two `extract_intent_*` tests FAIL (user message doesn't yet contain the block); the earlier 3 `_build_chunk_context` tests still PASS.

- [ ] **Step 7: Wire the block into `extract_intent`'s user message**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, modify `extract_intent` (currently):

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
        response_format=_RESPONSE_FORMAT,
    )
```

to:

```python
async def extract_intent(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, model: str | None = None,
) -> dict:
    resolved_model = model or await gateway.get_model(role="slm")
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    response = await gateway.chat(
        role="slm",
        messages=[
            {"role": "system", "content": _system_prompt_for_model(resolved_model)},
            {"role": "user", "content": user_message},
        ],
        model=model,
        response_format=_RESPONSE_FORMAT,
    )
```

- [ ] **Step 8: Run full `test_ai_mode_intent.py` file, confirm all tests pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: PASS — all pre-existing tests plus the 5 new ones (3 unit + 2 integration).

- [ ] **Step 9: Run full repo test suite to confirm no regressions**

Run: `uv run pytest`
Expected: PASS — same pass count as before plus 5, no failures elsewhere (this change only touches `intent.py`'s user-message construction, which no other module reads directly; `run_ai_mode`/`ws.py` callers only consume `extract_intent`'s return dict, unaffected).

- [ ] **Step 10: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat: inject chunk_query structural spans into AI Mode intent extraction

Give extract_intent's SLM call pre-parsed citation/section/court_city/quoted
span data instead of making it re-derive entity boundaries from raw text.
Drops proximity (ES-only) and alt_text (would fail _sanitize_filters'
literal-substring check if the model used it as a filter value) - see
docs/superpowers/specs/2026-08-13-ai-mode-chunk-context-injection-design.md

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Follow-up (not part of this plan)

The design spec's Validation plan (baseline / candidate-A / candidate-B eval runs via `compare_eval_runs.py`) is explicitly out of scope for implementation — it's a separate execution task using the eval harness from `docs/superpowers/specs/2026-08-10-small-model-eval-harness-design.md`, run after this change lands.
