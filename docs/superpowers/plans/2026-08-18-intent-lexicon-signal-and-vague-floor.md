# Intent Lexicon Signal And Vague Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a code-computed "does this query have any recognizable legal anchor" signal to `extract_intent`, feed it to the SLM as a soft hint, and use it as a hard, deterministic floor that forces `intent = []` whenever no anchor is found — regardless of what the SLM tagged.

**Architecture:** Single file (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`) gains one new pure-function detector built from three signals that already exist in the codebase (`chunk_query`'s structural spans, `classify_query_shape`, `expand_query_synonyms` — the same mechanism already powering `/v1/query-analysis`). The detector feeds two consumers: a soft prompt hint (unconditional, always attempted) and a hard post-processing override (`_validate_result`, always wins). Because the hard floor triggers on the same "no anchor" condition that an already-shipped fact-pattern-question fix also relies on, this plan knowingly reverts that fix's benefit for anchor-less queries — documented as an explicit, deliberate ruling, not a bug to fix later.

**Tech Stack:** Python 3.11, pytest + pytest-asyncio, no new dependencies. `common.query_tokenizer`'s `classify_query_shape`/`expand_query_synonyms` are newly imported into `ai_mode/intent.py` (they already exist, already used by `common/es_client.py::build_query_preview`).

**Spec:** `docs/superpowers/specs/2026-08-18-intent-lexicon-signal-and-vague-floor-design.md`

## Global Constraints

- The hard floor (`_too_vague_to_tag`) has **no word-count threshold and no phrasing-shape exemption** (e.g. no "ends in `?`" carve-out) — it fires purely on anchor absence, at any query length. This is an explicit, recorded ruling in the spec, not an oversight — do not add either back in without a new ruling.
- The floor deliberately reverts `extract_intent`'s existing fact-pattern-question `caselaws` signal for any query with zero literal anchor. This is accepted and documented — do not treat a fact-pattern query getting force-emptied as a bug to fix.
- `_sanitize_filters`/`_validate_categories`'s existing behavior is untouched — this plan adds a new check alongside them.
- No change to `collections_for_intent()` (`common/schemas.py`) — its existing empty-intent-searches-all-11-collections behavior is exactly what this plan leans on for safety.
- Every existing test in `test_ai_mode_intent.py` that uses an anchor-free stub query (`"q"`, `"original query normalized"`, etc.) to assert a non-empty `result["intent"]` must be given a real anchor in its query string (not have its assertions weakened to expect `[]`) — preserves what each test actually verifies.

---

### Task 1: Shared anchor detector — `_has_legal_anchor`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py:8` (import line), and add a new function after `_build_chunk_context` (currently ends at line 67, blank line 68-69 before `_LLAMA_SYSTEM_PROMPT` starts at line 70)
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Consumes: `common.query_tokenizer.classify_query_shape(query: str) -> str` (returns `"citation"`, `"provision"`, or `"plain"`), `common.query_tokenizer.expand_query_synonyms(query: str) -> str` (returns the query unchanged if no lexicon term matched, or with expansions appended if one did) — both already exist and are unit-tested in `packages/common/tests/`, not part of this task.
- Produces: `_has_legal_anchor(query: str, chunk_context: str | None) -> bool` — Tasks 2 and 3 both call this.

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py` (add the import near the top, alongside the existing imports):

```python
from retrieval_api.ai_mode.intent import _has_legal_anchor
```

```python
def test_has_legal_anchor_true_when_chunk_context_present():
    assert _has_legal_anchor("Delhi High Court ruling", '[{"text": "Delhi High Court", "type": "court_city"}]') is True


def test_has_legal_anchor_true_when_lexicon_synonym_matches():
    # "ACIT" expands via the legal lexicon (see common/legal_lexicon.py's synonyms table) -
    # expand_query_synonyms appends the expansion, so the returned string differs from the input.
    assert _has_legal_anchor("ACIT order challenged", None) is True


def test_has_legal_anchor_true_when_shape_is_not_plain():
    # "section 80HH" matches SECTION_PATTERN inside classify_query_shape -> shape="provision",
    # even with chunk_context=None passed explicitly (simulating no structural spans found).
    assert _has_legal_anchor("explain section 80HH", None) is True


def test_has_legal_anchor_false_for_bare_topic_words():
    assert _has_legal_anchor("capital gains", None) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k has_legal_anchor -v`
Expected: all 4 FAIL with `ImportError: cannot import name '_has_legal_anchor'`.

- [ ] **Step 3: Add the import and the function to `intent.py`**

Change line 8 (the existing `common.query_tokenizer` import) from:

```python
from common.query_tokenizer import chunk_query
```

to:

```python
from common.query_tokenizer import chunk_query, classify_query_shape, expand_query_synonyms
```

Add this function immediately after `_build_chunk_context` (after its closing `return json.dumps(spans, ensure_ascii=False)` line, before the blank lines that precede `_LLAMA_SYSTEM_PROMPT`):

```python
def _has_legal_anchor(query: str, chunk_context: str | None) -> bool:
    """True when any layer of the existing lexical pipeline (structural chunking, legal
    lexicon, shape classification) recognizes something in this query - a citation,
    section/rule reference, court/party name, date, or known legal abbreviation. False
    means the query is lexically empty of legal content, used both as a soft prompt hint
    (below) and a hard classification floor (_too_vague_to_tag, in _validate_result)."""
    if chunk_context is not None:
        return True  # a structural span (citation/section/court/date/party) was found
    if expand_query_synonyms(query) != query:
        return True  # a legal-lexicon term/abbreviation was recognized
    if classify_query_shape(query) != "plain":
        return True  # provision/citation shape implies an anchor
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k has_legal_anchor -v`
Expected: all 4 PASS.

- [ ] **Step 5: Run the full file to confirm nothing else broke**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: all existing tests still PASS (this task only adds an import and a new unused-so-far function; no existing behavior changes yet).

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat(ai-mode): add shared legal-anchor detector for intent classification

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Lexicon signal — soft prompt hint

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py` — `_LLAMA_SYSTEM_PROMPT` (lines 136-137) and `extract_intent`'s user-message assembly (lines 294-301)
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Consumes: `_has_legal_anchor(query, chunk_context) -> bool` (Task 1).
- Produces: no new function — `extract_intent`'s user message now conditionally includes a `"Lexicon check: ..."` block. Task 3 does not consume this directly (it reads `_has_legal_anchor` itself), but both must agree on wording for the eval/report to read sensibly together.

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py`:

```python
@pytest.mark.asyncio
async def test_extract_intent_appends_lexicon_check_when_no_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "capital gains", "intent": [], "filters": {}})

    await extract_intent(gateway, "capital gains")

    user_message = gateway.chat.await_args.kwargs["messages"][1]["content"]
    assert "Lexicon check:" in user_message
    assert "no known legal term" in user_message


@pytest.mark.asyncio
async def test_extract_intent_omits_lexicon_check_when_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "Delhi High Court ruling", "intent": ["caselaws"], "filters": {},
    })

    await extract_intent(gateway, "Delhi High Court ruling")

    user_message = gateway.chat.await_args.kwargs["messages"][1]["content"]
    assert "Lexicon check:" not in user_message
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k lexicon_check -v`
Expected: `test_extract_intent_appends_lexicon_check_when_no_anchor_found` FAILS (no "Lexicon check:" text in the user message yet). `test_extract_intent_omits_lexicon_check_when_anchor_found` PASSES trivially (nothing appends it yet either way) — that's fine, it becomes a real check once Step 3 lands.

- [ ] **Step 3: Wire the hint into `extract_intent` and add the prompt sentence**

In `_LLAMA_SYSTEM_PROMPT`, change these two lines (currently lines 136-137):

```python
  Output an empty list when no category confidently applies. Never output
  any other value.
```

to:

```python
  Output an empty list when no category confidently applies. Never output
  any other value. If the user message below includes a "Lexicon check" note
  stating no legal term was recognized in the query, treat that as strong
  evidence to abstain (output an empty list) unless the query's own wording -
  not just its general subject - clearly names something concrete.
```

In `extract_intent`, change the body (currently lines 294-301):

```python
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
```

to:

```python
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    has_anchor = _has_legal_anchor(query, chunk_context)
    if not has_anchor:
        user_message += (
            "\n\nLexicon check: no known legal term, Act/section reference, citation, or "
            "party pattern was recognized anywhere in this query."
        )
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
```

(the rest of `extract_intent`'s body is unchanged for this task — leave the `gateway.chat(...)` call and everything after it exactly as it is; `has_anchor` is used again in Task 3).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -k lexicon_check -v`
Expected: both PASS.

- [ ] **Step 5: Fix the one pre-existing test this step's change breaks**

`test_extract_intent_user_message_unchanged_when_no_spans_found` uses an anchor-free query (`"capital gains treatment"`) and asserts `user_message == "capital gains treatment"` exactly — Step 3's change now appends the lexicon-check block to that same query, so this exact-equality assertion fails as of this step, deterministically (not a maybe). Replace the whole test with:

```python
@pytest.mark.asyncio
async def test_extract_intent_user_message_unchanged_when_no_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "capital gains treatment",
        "intent": ["caselaws"],
        "filters": {},
    })

    await extract_intent(gateway, "capital gains treatment")

    call_kwargs = gateway.chat.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message.startswith("capital gains treatment")
    assert "Structural spans already present" not in user_message
    assert "Lexicon check:" in user_message
```

This keeps testing what it originally verified (no structural-span block gets injected when `chunk_query` finds nothing) while accounting for the new lexicon-check block this task adds.

- [ ] **Step 6: Run the full file to confirm nothing else broke**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat(ai-mode): add lexicon-check soft hint to extract_intent's prompt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Too-vague-to-tag hard floor, plus fixing collateral test breakage

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py` — `_validate_result` (lines 275-286) and its one call site inside `extract_intent` (line 319, `result = _validate_result(query, result)`)
- Modify: `packages/retrieval-api/tests/test_ai_mode_intent.py` — give real anchors to every pre-existing test whose query has none but whose assertions depend on a non-empty `intent` or an unmodified user message (see Step 1 below for the exact list)
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py` (new tests) — same file

**Interfaces:**
- Consumes: `_has_legal_anchor(query, chunk_context) -> bool` (Task 1).
- Produces: `_too_vague_to_tag(query: str, chunk_context: str | None) -> bool`. `_validate_result` becomes `_validate_result(query: str, result, chunk_context: str | None) -> dict` — signature change, one call site to update.

**This task must land as one commit** — the floor and its collateral test fixes cannot be split across commits without leaving the suite red in between.

- [ ] **Step 1: Fix the pre-existing tests that will collide with the floor**

These queries currently have no legal anchor and their tests assert a non-empty `result["intent"]` or an exact unmodified `user_message` — both would break once the floor exists. Change only the `query` string passed to `extract_intent` in each (keep every other line of each test, including its mocked `gateway.chat.return_value`, unchanged unless noted) so each query now contains a real anchor a real user might plausibly write, without changing what the test is actually verifying:

1. `test_extract_intent_drops_unrecognized_category_values` — change `await extract_intent(gateway, "q")` to `await extract_intent(gateway, "section 80HH deduction")`.
2. `test_extract_intent_dedupes_category_values` — same change: `"q"` → `"section 80HH deduction"`.
3. `test_extract_intent_accepts_each_allowed_category_label` — this test loops over all 6 category labels reusing `query="q"` each iteration; change the loop body's call to `await extract_intent(gateway, "section 80HH deduction")`.
4. `test_extract_intent_emits_intent_step_when_on_step_given` — this one is trickier: its query (`"original query normalized"`) is also asserted verbatim as `original_query`/`search_query` in the expected dict, and appears again in the `steps` assertion. Replace every occurrence of `"original query normalized"` in this test — there are 7, count them as you go: (1) `gateway.chat.return_value`'s `"search_query"`; (2) the `extract_intent(gateway, ...)` call argument; (3) `"original_query"` and (4) `"search_query"` inside the `assert result == {...}` block; (5) `"query"`, (6) `"original_query"`, and (7) `"search_query"` inside the `assert steps == [...]` block — with `"section 80HH normalized"`. Keep everything else (the `intent`/`filters` values, the `steps` structure, the `on_step` callback) identical.
5. `test_extract_intent_parses_json_object_response` — its query is `"IPC 302 punishment"`. Whether `"IPC"` is a recognized lexicon synonym is not confirmed by this plan (see spec's Non-goals). To avoid depending on unverified lexicon coverage, change the query to unambiguously contain a real anchor: replace `await extract_intent(gateway, "IPC 302 punishment")` with `await extract_intent(gateway, "section 302 punishment")`. In the `assert result == {...}` block right below it, change both the `"original_query"` value and the `"search_query"` value from `"IPC 302 punishment"` to `"section 302 punishment"` (2 replacements total in that dict; leave the trailing `# rewrite rejected: <60% token overlap with input` comment exactly as it is — the rewrite is still rejected on this new query too, since `_safe_rewrite`'s digit-set check fires first: `{"302"} != {"103"}`, independent of token overlap). Leave `gateway.chat.return_value`'s `"search_query": "BNS section 103 murder punishment"` untouched — only the `extract_intent(...)` call argument and the two assertion values change.

- [ ] **Step 2: Write the new failing tests**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py`:

```python
def test_too_vague_to_tag_true_when_no_anchor():
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("capital gains", None) is True


def test_too_vague_to_tag_false_when_anchor_present():
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("explain section 80HH", None) is False


def test_too_vague_to_tag_true_for_anchor_free_fact_pattern_question():
    """Documents the accepted tradeoff from the design spec's "Explicit ruling" section:
    a fact-pattern/scenario question with zero literal anchor is force-emptied by this
    floor even though extract_intent's caselaws category signal would otherwise tag it -
    deliberate, not a bug. A future change to _too_vague_to_tag has to consciously break
    this test to reintroduce fact-pattern tagging."""
    from retrieval_api.ai_mode.intent import _too_vague_to_tag
    assert _too_vague_to_tag("gift from father taxable?", None) is True


@pytest.mark.asyncio
async def test_extract_intent_forces_empty_intent_when_no_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "capital gains", "intent": ["commentary"], "filters": {},
    })

    result = await extract_intent(gateway, "capital gains")

    assert result["intent"] == []


@pytest.mark.asyncio
async def test_extract_intent_keeps_intent_when_anchor_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "section 80HH deduction", "intent": ["acts"], "filters": {},
    })

    result = await extract_intent(gateway, "section 80HH deduction")

    assert result["intent"] == ["acts"]
```

- [ ] **Step 3: Run tests to verify the new ones fail and Step 1's fixed tests fail too (pre-fix)**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: the 4 new tests from Step 2 FAIL (`_too_vague_to_tag` doesn't exist yet; `extract_intent_forces_empty_intent...` fails because nothing forces it empty yet). The 5 tests fixed in Step 1 should already PASS at this point (their queries now have real anchors, which Task 1/2's code doesn't touch their outcome) — if any of those 5 still fail here, re-check the query substitution in Step 1 before proceeding.

- [ ] **Step 4: Implement `_too_vague_to_tag` and wire it into `_validate_result`**

Add this function immediately after `_has_legal_anchor` (from Task 1):

```python
def _too_vague_to_tag(query: str, chunk_context: str | None) -> bool:
    """Deliberately no word-count or phrasing-shape (e.g. "ends in ?") exemption - see
    docs/superpowers/specs/2026-08-18-intent-lexicon-signal-and-vague-floor-design.md's
    "Explicit ruling" section. This knowingly force-empties anchor-free fact-pattern
    questions that extract_intent's caselaws signal would otherwise correctly tag -
    accepted because a guaranteed-safe search-all outcome was judged strictly
    preferable to any residual risk of a wrong-collection search."""
    return not _has_legal_anchor(query, chunk_context)
```

Change `_validate_result` (currently lines 275-286) from:

```python
def _validate_result(query: str, result) -> dict:
    if not isinstance(result, dict):
        return _fallback_intent(query)
    search_query = result.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        return _fallback_intent(query)
    return {
        "original_query": query,
        "search_query": _safe_rewrite(query, search_query.strip()),
        "intent": _validate_categories(result.get("intent")),
        "filters": _sanitize_filters(query, result.get("filters")),
    }
```

to:

```python
def _validate_result(query: str, result, chunk_context: str | None) -> dict:
    if not isinstance(result, dict):
        return _fallback_intent(query)
    search_query = result.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        return _fallback_intent(query)
    return {
        "original_query": query,
        "search_query": _safe_rewrite(query, search_query.strip()),
        "intent": [] if _too_vague_to_tag(query, chunk_context) else _validate_categories(result.get("intent")),
        "filters": _sanitize_filters(query, result.get("filters")),
    }
```

Change the one call site inside `extract_intent` (currently `result = _validate_result(query, result)`, in the `else` branch after the `json.loads(response)` try/except) to:

```python
        result = _validate_result(query, result, chunk_context)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: all tests PASS, including all 6 fixed from Step 1 and all 4 new from Step 2.

- [ ] **Step 6: Run the full monorepo suite**

Run: `uv run pytest`
Expected: all tests pass except the 2 known pre-existing, unrelated `test_settings_load_from_env` flaky failures (`packages/auth/tests/test_config.py`, `packages/persona/tests/test_config.py` — a test-isolation/settings-caching issue predating this plan, not caused by it). If you see any *other* failure, stop and investigate before committing — do not assume it's unrelated.

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat(ai-mode): force intent=[] when no legal anchor is found

Deliberately reverts the fact-pattern-question caselaws signal for
anchor-free queries - see spec's Explicit ruling section. No word-count
or phrasing-shape exemption, by design.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Eval verification against the real model

**Files:**
- None modified — this task runs the existing `collection_routing_eval.py` script (from an earlier plan, already in the repo at `packages/retrieval-api/src/retrieval_api/collection_routing_eval.py`) against the real `model-gateway` service and records the result. No dataset edit needed (see Global Constraints and the spec's Testing section — `"gift from father taxable?"`'s `expected_categories` is already `[]`).

**Interfaces:**
- Consumes: `extract_intent` (Tasks 1-3's changes, already committed), `evals/collection_routing_cases.json` (pre-existing, unmodified by this task).
- Produces: nothing new for other tasks — this is a verification step confirming the shipped behavior matches the spec's intent, not a code change.

This task requires a running `model-gateway` service and is not part of the automated pytest suite - do not attempt to script it as a pytest test.

- [ ] **Step 1: Ensure `model-gateway` is running**

Run: `docker compose up -d model-gateway` (from repo root). Confirm it's reachable — the compose file maps it to host port 8001.

- [ ] **Step 2: Run the eval**

Run: `uv run python -m retrieval_api.collection_routing_eval --gateway-url http://localhost:8001`

- [ ] **Step 3: Confirm the expected outcome changes**

Compare against the pre-fix baseline recorded in the spec's Problem section (R13/R14/R18 all `wrong`). Expected post-fix:
- R13 (`"gift from father taxable?"`) — `safe-empty` (was `wrong`)
- R14 (`"capital gains"`) — `safe-empty` (was `wrong`)
- R18 (`"help with income tax"`) — `safe-empty` (was `wrong`)
- All 10 confident cases (R01-R10) — still `PASS` (either `exact` or `safe-empty` — per the spec's Testing section, a shift from `exact` to `safe-empty` on any of them is an accepted possible outcome of this design, not a regression; a shift to `wrong` would be a real regression)
- Overall tally: no case in the `wrong` bucket among R11-R18 (the vague-labeled cases) except any genuinely new failure you did not expect — investigate any such case before considering this task done.

- [ ] **Step 4: Record the result**

No commit needed for this task (nothing changed in git) — report the eval's final tally and the three flipped cases' outcomes as this task's deliverable. If any confident case (R01-R10) shifted to `wrong` (not just `safe-empty`), stop and report it — do not silently consider the task complete.
