# Persona Context Trust Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate persona context on sample size (query_count >= 20), replace expertise_level/query_style's last-write-wins overwrite with a vote-tally that builds up over time, add a shared relevance-judgment instruction so the SLM ignores a persona hint that doesn't fit the current query, and wire persona context into `extract_intent` (today it only reaches `synthesize`).

**Architecture:** Two packages change. `packages/persona` (`merge.py`, `prompt.py`) owns the tally/gate/instruction-constant logic — pure functions, no I/O. `packages/retrieval-api`'s `ai_mode/intent.py`, `ai_mode/synthesize.py`, `ai_mode/pipeline.py` consume `persona.prompt.RELEVANCE_INSTRUCTION` and thread an existing-but-partially-unused `persona_context` parameter one hop further (`pipeline.py` already receives it; it just doesn't forward it to `extract_intent` yet).

**Tech Stack:** Python 3.11, pytest + pytest-asyncio, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-17-persona-context-trust-gating-design.md`

## Global Constraints

- Trust threshold is `query_count >= 20` (spec §2) — a module-level named constant, not a magic number.
- `record_signal`'s write path is never gated — it keeps writing on every AI-mode query regardless of `query_count` (spec §2). Only `render_persona_context`'s *surfacing* is gated.
- No backfill script — migration to the vote-tally shape happens lazily on the next `record_signal` write for a given doc (spec §1, Non-goals).
- `_sanitize_filters`/`_validate_categories` in `ai_mode/intent.py` are untouched — persona stays a soft prior for `intent` tagging, never a filter source (spec Non-goals).
- The relevance-judgment instruction text must be defined once and imported by both `ai_mode/intent.py` and `ai_mode/synthesize.py` — no copy-pasted duplicate string (spec §3).

---

### Task 1: Vote-tally merge logic (`persona` package)

**Files:**
- Modify: `packages/persona/src/persona/merge.py`
- Test: `packages/persona/tests/test_merge.py`

**Interfaces:**
- Consumes: nothing new — `VALID_EXPERTISE_LEVELS`, `VALID_QUERY_STYLES` already defined in this file.
- Produces: `merge_expertise_patch(existing: dict, patch: dict | None) -> dict` — **signature unchanged**, but return shape changes: result now includes `expertise_votes: dict[str, int]` and `query_style_votes: dict[str, int]` alongside the existing `expertise_level`/`query_style` string fields (the latter always set to the current tally mode). Task 2 (`prompt.py`) and `persona/repository.py`'s `record_signal` (unchanged, already passes `existing` through) depend on this shape.

- [ ] **Step 1: Write the failing tests (replace the existing `merge_expertise_patch` tests)**

Replace `packages/persona/tests/test_merge.py` entirely with:

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


def test_merge_expertise_patch_first_vote_seeds_tally_and_sets_mode():
    result = merge_expertise_patch({}, {"expertise_level": "student"})
    assert result["expertise_votes"] == {"student": 1}
    assert result["expertise_level"] == "student"


def test_merge_expertise_patch_accumulates_votes_across_calls():
    first = merge_expertise_patch({}, {"expertise_level": "practitioner"})
    second = merge_expertise_patch(first, {"expertise_level": "practitioner"})
    third = merge_expertise_patch(second, {"expertise_level": "student"})
    assert third["expertise_votes"] == {"practitioner": 2, "student": 1}
    assert third["expertise_level"] == "practitioner"  # still majority


def test_merge_expertise_patch_exact_tie_keeps_previous_mode():
    existing = {"expertise_votes": {"practitioner": 1}, "expertise_level": "practitioner"}
    result = merge_expertise_patch(existing, {"expertise_level": "student"})
    assert result["expertise_votes"] == {"practitioner": 1, "student": 1}
    assert result["expertise_level"] == "practitioner"  # tie: previous mode kept


def test_merge_expertise_patch_new_leader_overtakes_previous_mode():
    existing = {"expertise_votes": {"practitioner": 1, "student": 1}, "expertise_level": "practitioner"}
    result = merge_expertise_patch(existing, {"expertise_level": "student"})
    assert result["expertise_votes"] == {"practitioner": 1, "student": 2}
    assert result["expertise_level"] == "student"  # student now leads outright, no tie


def test_merge_expertise_patch_migrates_old_string_field_as_one_vote():
    existing = {"expertise_level": "practitioner"}  # old shape, predates this change, no tally yet
    result = merge_expertise_patch(existing, {"expertise_level": "student"})
    assert result["expertise_votes"] == {"practitioner": 1, "student": 1}
    assert result["expertise_level"] == "practitioner"  # tie: previous (migrated) mode kept


def test_merge_expertise_patch_handles_empty_existing():
    result = merge_expertise_patch({}, {"expertise_level": "student"})
    assert result["expertise_level"] == "student"
    assert result["expertise_votes"] == {"student": 1}


def test_merge_expertise_patch_drops_invalid_expertise_level():
    existing = {"expertise_level": "practitioner", "expertise_votes": {"practitioner": 1}}
    result = merge_expertise_patch(existing, {"expertise_level": "omniscient"})
    assert result == existing


def test_merge_expertise_patch_drops_invalid_query_style():
    existing = {"query_style": "broad", "query_style_votes": {"broad": 1}}
    result = merge_expertise_patch(existing, {"query_style": "essay-length"})
    assert result == existing


def test_merge_expertise_patch_strips_extraneous_keys():
    result = merge_expertise_patch({}, {"expertise_level": "student", "injected": "malicious"})
    assert result["expertise_level"] == "student"
    assert "injected" not in result


def test_merge_expertise_patch_returns_existing_unchanged_when_all_values_invalid():
    existing = {"expertise_level": "practitioner", "query_style": "precise-citation"}
    result = merge_expertise_patch(existing, {"expertise_level": "bogus", "query_style": "bogus"})
    assert result == existing


def test_merge_expertise_patch_both_fields_merge_independently_in_one_call():
    existing = {
        "expertise_level": "practitioner", "expertise_votes": {"practitioner": 1},
        "query_style": "broad", "query_style_votes": {"broad": 1},
    }
    result = merge_expertise_patch(existing, {"expertise_level": "expert", "query_style": "precise-citation"})
    assert result["expertise_votes"] == {"practitioner": 1, "expert": 1}
    assert result["query_style_votes"] == {"broad": 1, "precise-citation": 1}
    assert result["expertise_level"] == "practitioner"  # tie kept
    assert result["query_style"] == "broad"  # tie kept
```

- [ ] **Step 2: Run tests to verify the expertise/query_style ones fail**

Run: `uv run pytest packages/persona/tests/test_merge.py -v`
Expected: the `merge_category_affinity` tests PASS unchanged; every `merge_expertise_patch` test FAILS (old code still does last-write-wins overwrite, so e.g. `result["expertise_votes"]` raises `KeyError`).

- [ ] **Step 3: Replace `merge_expertise_patch` in `merge.py` with vote-tally logic**

Replace the existing `merge_expertise_patch` function (keep `KNOWN_CATEGORIES`, `VALID_EXPERTISE_LEVELS`, `VALID_QUERY_STYLES`, `merge_category_affinity` exactly as they are) with:

```python
def _resolve_mode(votes: dict[str, int], previous: str | None) -> str:
    max_count = max(votes.values())
    leaders = sorted(value for value, count in votes.items() if count == max_count)
    # Tie: keep the previous mode rather than churn to an arbitrary leader -
    # see docs/superpowers/specs/2026-08-17-persona-context-trust-gating-design.md §1.
    return previous if previous in leaders else leaders[0]


def _merge_vote_field(existing: dict, new_value: str, votes_key: str, value_key: str) -> tuple[dict, str]:
    votes = existing.get(votes_key)
    if votes is None:
        # Migration: an old doc has the plain string field but no tally yet -
        # seed the tally with that value as one vote before adding this one.
        old_value = existing.get(value_key)
        votes = {old_value: 1} if old_value else {}
    else:
        votes = dict(votes)
    votes[new_value] = votes.get(new_value, 0) + 1
    return votes, _resolve_mode(votes, existing.get(value_key))


def merge_expertise_patch(existing: dict, patch: dict | None) -> dict:
    if not patch:
        return existing

    # Never merge unvalidated SLM output verbatim - render_persona_context
    # interpolates expertise_level/query_style straight into the synthesis
    # system prompt on every future request for this user, so an
    # out-of-enum or extraneous key here is a same-account prompt-injection
    # vector. Drop anything that isn't one of the design spec's enumerated
    # values, and drop any key other than expertise_level/query_style entirely.
    filtered = {}
    expertise_level = patch.get("expertise_level")
    if expertise_level in VALID_EXPERTISE_LEVELS:
        filtered["expertise_level"] = expertise_level
    query_style = patch.get("query_style")
    if query_style in VALID_QUERY_STYLES:
        filtered["query_style"] = query_style

    if not filtered:
        return existing

    result = dict(existing)
    if "expertise_level" in filtered:
        votes, mode = _merge_vote_field(existing, filtered["expertise_level"], "expertise_votes", "expertise_level")
        result["expertise_votes"] = votes
        result["expertise_level"] = mode
    if "query_style" in filtered:
        votes, mode = _merge_vote_field(existing, filtered["query_style"], "query_style_votes", "query_style")
        result["query_style_votes"] = votes
        result["query_style"] = mode
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/persona/tests/test_merge.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/persona/src/persona/merge.py packages/persona/tests/test_merge.py
git commit -m "feat(persona): replace expertise/query_style overwrite with vote tally

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Trust gate + shared relevance instruction (`persona` package)

**Files:**
- Modify: `packages/persona/src/persona/prompt.py`
- Test: `packages/persona/tests/test_prompt.py`

**Interfaces:**
- Consumes: `persona.merge.KNOWN_CATEGORIES` (already imported today).
- Produces: `render_persona_context(persona: dict | None) -> str` — **signature unchanged**, now returns `""` whenever `persona.get("query_count", 0) < 20`. New module-level `RELEVANCE_INSTRUCTION: str` constant — Task 3 (`ai_mode/intent.py`) and Task 4 (`ai_mode/synthesize.py`) import this directly from `persona.prompt`.

Note: `render_persona_context` does **not** need to read `expertise_votes`/`query_style_votes` directly — Task 1's `merge_expertise_patch` always keeps the plain `expertise_level`/`query_style` fields set to the current tally mode, so the existing field reads already work for both old-shape and migrated docs.

- [ ] **Step 1: Write the failing tests (replace the existing file)**

Replace `packages/persona/tests/test_prompt.py` entirely with:

```python
from persona.prompt import RELEVANCE_INSTRUCTION, render_persona_context


def test_render_persona_context_returns_empty_string_for_none():
    assert render_persona_context(None) == ""


def test_render_persona_context_returns_empty_string_for_no_signal_yet():
    assert render_persona_context({"user_id": "u1", "query_count": 0, "category_affinity": {}}) == ""


def test_render_persona_context_returns_empty_string_below_trust_threshold():
    persona = {
        "query_count": 19,
        "category_affinity": {"acts": 0.1, "caselaws": 0.8, "commentary": 0.1, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "expertise_level": "practitioner",
    }
    assert render_persona_context(persona) == ""


def test_render_persona_context_names_top_category_and_expertise_at_threshold():
    persona = {
        "query_count": 20,
        "category_affinity": {"acts": 0.1, "caselaws": 0.8, "commentary": 0.1, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "expertise_level": "practitioner",
    }
    context = render_persona_context(persona)
    assert "caselaws" in context
    assert "practitioner" in context


def test_render_persona_context_includes_query_style_when_present():
    persona = {
        "query_count": 25,
        "category_affinity": {"acts": 0.9, "caselaws": 0.0, "commentary": 0.0, "rules": 0.0, "articles": 0.0, "tariff": 0.0},
        "query_style": "precise-citation",
    }
    context = render_persona_context(persona)
    assert "precise-citation" in context


def test_relevance_instruction_is_a_nonempty_stable_string():
    assert isinstance(RELEVANCE_INSTRUCTION, str)
    assert "ignore" in RELEVANCE_INSTRUCTION.lower()
    assert len(RELEVANCE_INSTRUCTION) > 0
```

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `uv run pytest packages/persona/tests/test_prompt.py -v`
Expected: `test_render_persona_context_returns_empty_string_below_trust_threshold` FAILS (no gate yet, old code renders at count 19); `test_relevance_instruction_is_a_nonempty_stable_string` FAILS with `ImportError` (`RELEVANCE_INSTRUCTION` doesn't exist yet); the two count-20/25 tests currently pass by coincidence under old code (no gate) but confirm they still pass after the gate is added in the next step.

- [ ] **Step 3: Add the gate and the shared instruction constant to `prompt.py`**

```python
from persona.merge import KNOWN_CATEGORIES

_TRUST_THRESHOLD = 20

RELEVANCE_INSTRUCTION = (
    "The note above is a prior about this user's typical usage, not a "
    "fact about this query. Use it only if this query is genuinely "
    "ambiguous on its own. If the query's own content conflicts with or "
    "is unrelated to the note, ignore the note and rely on the query "
    "alone."
)


def render_persona_context(persona: dict | None) -> str:
    if not persona or persona.get("query_count", 0) < _TRUST_THRESHOLD:
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

Run: `uv run pytest packages/persona/tests/test_prompt.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/persona/src/persona/prompt.py packages/persona/tests/test_prompt.py
git commit -m "feat(persona): gate render_persona_context on query_count>=20, add RELEVANCE_INSTRUCTION

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Wire persona context into `extract_intent`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py`

**Interfaces:**
- Consumes: `persona.prompt.RELEVANCE_INSTRUCTION` (Task 2).
- Produces: `extract_intent(gateway, query, on_step=None, model=None, persona_context="") -> dict` — new `persona_context` keyword param, default `""` (backward compatible with every existing call site). Task 5 (`pipeline.py`) will start passing this.

- [ ] **Step 1: Write the failing tests**

Add to `packages/retrieval-api/tests/test_ai_mode_intent.py` (add the import alongside the existing ones at the top, and these two tests anywhere in the file):

```python
from persona.prompt import RELEVANCE_INSTRUCTION
```

```python
@pytest.mark.asyncio
async def test_extract_intent_includes_persona_context_in_user_message():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    await extract_intent(gateway, "q", persona_context="This user frequently asks about caselaws.")

    user_message = gateway.chat.await_args.kwargs["messages"][1]["content"]
    assert "This user frequently asks about caselaws." in user_message
    assert RELEVANCE_INSTRUCTION in user_message


@pytest.mark.asyncio
async def test_extract_intent_omits_persona_block_when_context_empty():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    await extract_intent(gateway, "q")

    user_message = gateway.chat.await_args.kwargs["messages"][1]["content"]
    assert RELEVANCE_INSTRUCTION not in user_message
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: `test_extract_intent_includes_persona_context_in_user_message` FAILS with `TypeError: extract_intent() got an unexpected keyword argument 'persona_context'`. The omits-block test passes trivially (no persona_context support yet, so the instruction is never in the message) — that's fine, it becomes a real check once Step 3 lands.

- [ ] **Step 3: Add `persona_context` param to `extract_intent`**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, add the import near the top (with the other local imports):

```python
from persona.prompt import RELEVANCE_INSTRUCTION
```

Change the function signature and body (`intent.py:275-284` currently):

```python
async def extract_intent(
    gateway: GatewayClient, query: str, on_step: OnStep | None = None, model: str | None = None,
    persona_context: str = "",
) -> dict:
    resolved_model = model or await gateway.get_model(role="slm")
    chunk_context = _build_chunk_context(query)
    user_message = query if chunk_context is None else (
        f"{query}\n\n"
        "Structural spans already present in the query above (for reference "
        f"only — do not add anything not already in the query text):\n{chunk_context}"
    )
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
    response = await gateway.chat(
```

(the rest of the function body is unchanged — leave the `gateway.chat(...)` call and everything after it exactly as it is).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py packages/retrieval-api/tests/test_ai_mode_intent.py
git commit -m "feat(ai-mode): wire persona_context into extract_intent

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Add relevance instruction to `synthesize`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_synthesize.py`

**Interfaces:**
- Consumes: `persona.prompt.RELEVANCE_INSTRUCTION` (Task 2).
- Produces: `synthesize(...)` — signature unchanged (`persona_context` param already existed); only the assembled `system_prompt` content changes when `persona_context` is non-empty.

- [ ] **Step 1: Write the failing test**

Add to `packages/retrieval-api/tests/test_ai_mode_synthesize.py`:

```python
@pytest.mark.asyncio
async def test_synthesize_appends_relevance_instruction_when_persona_context_present(monkeypatch):
    import retrieval_api.ai_mode.synthesize as module
    from persona.prompt import RELEVANCE_INSTRUCTION

    monkeypatch.setattr(module, "fetch_citations", AsyncMock(return_value={}))

    gateway = AsyncMock()
    gateway.chat_with_reasoning.return_value = ("Answer.", None)

    await synthesize(
        gateway, es_client=object(), query="q",
        top_chunks=[{"chunk_id": "a", "doc_id": "d1", "text": "chunk text"}],
        citations={"d1": {}},
        persona_context="This user frequently asks about caselaws.",
    )

    system_prompt = gateway.chat_with_reasoning.await_args.kwargs["messages"][0]["content"]
    assert "This user frequently asks about caselaws." in system_prompt
    assert RELEVANCE_INSTRUCTION in system_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: FAILS — `RELEVANCE_INSTRUCTION` is not in `system_prompt` yet (today's code appends only `persona_context`, no instruction).

- [ ] **Step 3: Update `synthesize.py` to append the instruction**

Add the import near the top of `packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py`:

```python
from persona.prompt import RELEVANCE_INSTRUCTION
```

Replace this line (`synthesize.py:39`):

```python
    system_prompt = _SYSTEM_PROMPT if not persona_context else f"{_SYSTEM_PROMPT}\n{persona_context}"
```

with:

```python
    system_prompt = _SYSTEM_PROMPT if not persona_context else f"{_SYSTEM_PROMPT}\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_synthesize.py -v`
Expected: all PASS (including the pre-existing tests, which don't pass `persona_context` and so are unaffected).

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/synthesize.py packages/retrieval-api/tests/test_ai_mode_synthesize.py
git commit -m "feat(ai-mode): append RELEVANCE_INSTRUCTION to synthesize's persona context

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Forward `persona_context` from `pipeline.py` to `extract_intent`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Consumes: `extract_intent(gateway, query, on_step=None, model=None, persona_context="")` (Task 3).
- Produces: `run_ai_mode(...)` — signature unchanged; `persona_context` (already a param) now reaches both `extract_intent` and `synthesize` instead of only `synthesize`.

- [ ] **Step 1: Update existing fakes and write the failing test**

In `packages/retrieval-api/tests/test_ai_mode_pipeline.py`, update the two existing `fake_extract_intent` definitions (in `test_run_ai_mode_success_path` and `test_run_ai_mode_forwards_rerank_enabled_setting_from_env`) to accept the new kwarg — change:

```python
    async def fake_extract_intent(gateway, query, on_step=None):
```

to:

```python
    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
```

in both places (this is required regardless of the new test below — once `pipeline.py` starts passing `persona_context=` unconditionally in Step 3, these fakes would otherwise raise `TypeError: unexpected keyword argument`).

Then add this new test to the same file:

```python
@pytest.mark.asyncio
async def test_run_ai_mode_forwards_persona_context_to_extract_intent_and_synthesize(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    received = {}

    async def fake_extract_intent(gateway, query, on_step=None, persona_context=""):
        received["intent_persona"] = persona_context
        return {"original_query": query, "search_query": "rewritten", "intent": [], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    async def fake_retrieve(gateway, milvus_client, search_query, doc_id_allowlist, intent, on_step=None):
        return []

    async def fake_rerank_and_prefetch(gateway, es_client, query, candidates, on_step=None, rerank_enabled=True):
        return [], {}

    async def fake_synthesize(gateway, es_client, query, top_chunks, citations, on_step=None, persona_context=""):
        received["synth_persona"] = persona_context
        return {"answer": "a", "citations": {}}

    monkeypatch.setattr(module, "extract_intent", fake_extract_intent)
    monkeypatch.setattr(module, "resolve_allowlist", fake_resolve_allowlist)
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(module, "rerank_and_prefetch", fake_rerank_and_prefetch)
    monkeypatch.setattr(module, "synthesize", fake_synthesize)

    await run_ai_mode(
        gateway=object(), es_client=object(), milvus_client=object(), query="q",
        persona_context="persona-hint",
    )

    assert received["intent_persona"] == "persona-hint"
    assert received["synth_persona"] == "persona-hint"
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: `test_run_ai_mode_forwards_persona_context_to_extract_intent_and_synthesize` FAILS — `received["intent_persona"]` stays `""` (pipeline.py doesn't forward it to `extract_intent` yet). The two pre-existing tests still PASS (their fakes now accept the extra kwarg but pipeline.py isn't sending it yet, which is fine since it defaults to `""`).

- [ ] **Step 3: Forward `persona_context` to `extract_intent` in `pipeline.py`**

In `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`, change:

```python
                intent_result = await extract_intent(gateway, query, on_step=on_step)
```

to:

```python
                intent_result = await extract_intent(gateway, query, on_step=on_step, persona_context=persona_context)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: all tests across all 4 packages PASS (143+ pre-existing plus the new ones from this plan).

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py packages/retrieval-api/tests/test_ai_mode_pipeline.py
git commit -m "feat(ai-mode): forward persona_context from pipeline to extract_intent

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
