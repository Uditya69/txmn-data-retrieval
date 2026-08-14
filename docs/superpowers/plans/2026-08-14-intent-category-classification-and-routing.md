# Intent Category Classification and Collection Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the AI Mode `slm` stage's 4-value intent enum with a multi-label content-category
tag (`acts`/`rules`/`caselaws`/`articles`/`commentary`/`tariff`), bias `search_query` phrasing by
that tag, and route which of the 11 Milvus collections get searched based on it.

**Architecture:** `extract_intent()` (`intent.py`) now returns `{"original_query", "intent":
list[str], "search_query", "filters"}` instead of the old `{"rewritten_query", "intent": str,
"filters"}`. `filters` extraction is untouched. The category list flows through `pipeline.py`
into `retrieve.py`, which uses a new `collections_for_intent()` (`common/schemas.py`) to pick
which Milvus collections `hybrid_search` queries, instead of always querying all 11. RRF fusion
weighting is explicitly **not** touched by category — stays neutral 1.0/1.0 (deletes
`_INTENT_RRF_WEIGHTS` outright, no replacement).

**Tech Stack:** Python 3.11, pytest + pytest-asyncio, existing `GatewayClient`/DeepInfra `slm`
role, Milvus via `pymilvus`.

**Spec:**
- `docs/superpowers/specs/2026-08-13-intent-category-classification-design.md`
- `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md`

## Global Constraints

- `filters` extraction (`court`/`act`/`section`/`date_range`/`party`/`bench`/`judge`) and
  `doc_id_allowlist` resolution stay untouched — category is purely additive alongside them.
- `section` filter key is now **unconditionally dropped** in `_sanitize_filters` (the old
  `intent != "provision_lookup"` gate compared against a value that no longer exists — made
  explicit rather than left as dead/always-true code).
- RRF fusion weight is always neutral `(1.0, 1.0)` — no category-based dense/sparse weighting.
  This was proposed during brainstorming and explicitly rejected; do not reintroduce it.
- `tariff` category has no Milvus collection to route to yet (`tariff_section` not in
  `MILVUS_COLLECTIONS`) — falls back to searching all 11, same as an empty/unrecognized-only
  `intent` list.
- Empty `intent` list (nothing confidently tagged, or the SLM-failure fallback path) → search
  all 11 collections, same as today's always-search-everything behavior.
- `caselaws` category maps to the original 7 collections (`case_summary`, `digest`, `headnotes`,
  `facts`, `held`, `ruling`, `metadata`) — `metadata`'s fields are case-doc-specific
  (`landmark_ruling`, etc.), it is not a fifth cross-category collection.
- `uv run pytest` from repo root must pass after every task.

---

### Task 1: `common/schemas.py` — category-to-collection routing

**Files:**
- Modify: `packages/common/src/common/schemas.py`
- Test: `packages/common/tests/test_schemas.py` (create if it doesn't exist — check first)

**Interfaces:**
- Consumes: existing `MILVUS_COLLECTIONS` list (already has all 11 names, confirmed current
  content: `["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
  "act_section", "rule_section", "article_section", "commentary_section"]`).
- Produces: `CATEGORY_COLLECTIONS: dict[str, list[str]]`, `collections_for_intent(intent:
  list[str]) -> list[str]` — both consumed by Task 4 (`retrieve.py`) and Task 6
  (`retrieval_eval.py`).

- [ ] **Step 1: Check for an existing schemas test file**

Run: `ls packages/common/tests/ 2>/dev/null || dir packages\common\tests`

If `test_schemas.py` exists, read it first and add to it rather than overwriting. If it doesn't
exist, Step 2 creates it fresh.

- [ ] **Step 2: Write the failing tests**

Create/append to `packages/common/tests/test_schemas.py`:

```python
from common.schemas import MILVUS_COLLECTIONS, collections_for_intent


def test_collections_for_intent_empty_list_returns_all_collections():
    assert collections_for_intent([]) == MILVUS_COLLECTIONS


def test_collections_for_intent_single_category_routes_to_its_group():
    assert collections_for_intent(["acts"]) == ["act_section"]
    assert collections_for_intent(["rules"]) == ["rule_section"]
    assert collections_for_intent(["articles"]) == ["article_section"]
    assert collections_for_intent(["commentary"]) == ["commentary_section"]


def test_collections_for_intent_caselaws_routes_to_original_seven():
    assert collections_for_intent(["caselaws"]) == [
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
    ]


def test_collections_for_intent_multi_category_unions_groups():
    result = collections_for_intent(["acts", "caselaws"])

    assert set(result) == {
        "case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata",
        "act_section",
    }


def test_collections_for_intent_result_order_follows_milvus_collections():
    result = collections_for_intent(["acts", "caselaws"])

    assert result == [c for c in MILVUS_COLLECTIONS if c in set(result)]


def test_collections_for_intent_tariff_only_falls_back_to_all_collections():
    """tariff_section isn't in MILVUS_COLLECTIONS yet - a tariff-only tag has
    nothing to route to, so it must fall back to searching everything rather
    than an empty collection list (which would search nothing)."""
    assert collections_for_intent(["tariff"]) == MILVUS_COLLECTIONS


def test_collections_for_intent_unrecognized_tag_only_falls_back_to_all_collections():
    assert collections_for_intent(["not_a_real_category"]) == MILVUS_COLLECTIONS
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_schemas.py -v`
Expected: FAIL with `ImportError: cannot import name 'collections_for_intent'`

- [ ] **Step 4: Implement `CATEGORY_COLLECTIONS` and `collections_for_intent`**

Add to the end of `packages/common/src/common/schemas.py` (after the existing
`MASTERINFO_CITATION_FIELDS` block):

```python
# Maps each intent category tag to the Milvus collection(s) it routes to. "tariff"
# has no entry - tariff_section isn't in MILVUS_COLLECTIONS yet (parked in the
# ingestion pipeline's _disabled_collections, not live) - a tariff-only intent tag
# falls through collections_for_intent's fallback instead. "caselaws" maps to the
# original 7 collections including metadata - its fields (landmark_ruling, doc-level
# heading/subheading) are case-doc-specific, not a generic cross-category collection.
CATEGORY_COLLECTIONS: dict[str, list[str]] = {
    "caselaws": ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"],
    "acts": ["act_section"],
    "rules": ["rule_section"],
    "articles": ["article_section"],
    "commentary": ["commentary_section"],
}


def collections_for_intent(intent: list[str]) -> list[str]:
    """Which Milvus collections to search for a given intent category list.
    Empty/unrecognized-only intent (nothing confidently tagged, a tariff-only
    tag, or a value CATEGORY_COLLECTIONS has no entry for) falls back to
    searching every collection - never worse than the old always-search-
    everything behavior. Multi-category intent unions its groups. Return
    order follows MILVUS_COLLECTIONS's own order, not intent's tag order, so
    trace/log output stays collection-order-stable regardless of tag order.
    """
    if not intent:
        return MILVUS_COLLECTIONS
    routed = {collection for tag in intent for collection in CATEGORY_COLLECTIONS.get(tag, [])}
    return [c for c in MILVUS_COLLECTIONS if c in routed] or MILVUS_COLLECTIONS
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_schemas.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Run the full common package test suite to check for regressions**

Run: `uv run pytest packages/common/tests -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add packages/common/src/common/schemas.py packages/common/tests/test_schemas.py
git commit -m "feat: add category-to-collection routing to common/schemas.py"
```

---

### Task 2: `intent.py` — category taxonomy, output shape, search_query phrasing bias

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`
- Modify: `packages/common/src/common/schema_context.py:38` (wording only: "phrase rewritten_query" → "phrase search_query")
- Test: `packages/retrieval-api/tests/test_ai_mode_intent.py` (rewrite)

**Interfaces:**
- Consumes: `common.schema_context.build_schema_context()` (unchanged signature), `GatewayClient.chat()`/`.get_model()` (unchanged).
- Produces: `extract_intent(gateway, query, on_step=None, model=None) -> dict` returning
  `{"original_query": str, "intent": list[str], "search_query": str, "filters": dict}`. This
  exact shape is consumed by Task 3 (`pipeline.py`) and Task 6 (`retrieval_eval.py`).
  `_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}` is
  the canonical category set, also referenced by Task 1's `CATEGORY_COLLECTIONS` keys (must stay
  in sync — `CATEGORY_COLLECTIONS`'s keys are a subset of `_ALLOWED_CATEGORIES` minus `tariff`).

- [ ] **Step 1: Write the new system prompt (failing test first)**

Replace the entire `test_ai_mode_intent.py` file content with the version below. This is a full
rewrite (not a patch) because nearly every existing test asserts on the old `rewritten_query`
field name and the old string-valued `intent` — mechanically updating each in place is
error-prone at this file's size; a clean rewrite keeps every test's assertions consistent with
the new shape.

```python
import json
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.intent import _build_chunk_context, extract_intent


@pytest.mark.asyncio
async def test_extract_intent_parses_json_object_response():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "BNS section 103 murder punishment",
        "intent": ["acts"],
        "filters": {"act": "BNS"},
    })

    result = await extract_intent(gateway, "IPC 302 punishment")

    assert result == {
        "original_query": "IPC 302 punishment",
        "search_query": "IPC 302 punishment",  # rewrite rejected: <60% token overlap with input
        "intent": ["acts"],
        "filters": {},
    }
    gateway.chat.assert_awaited_once()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["role"] == "slm"


@pytest.mark.asyncio
async def test_extract_intent_requests_json_object_response_format():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

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
        "search_query": "capital gains set off against carried forward business losses",
        "intent": ["caselaws"],
        "filters": {},
    }) + "\n```"

    result = await extract_intent(gateway, "set off capital gains against brought forward business losses")

    assert result == {
        "original_query": "set off capital gains against brought forward business losses",
        "search_query": "set off capital gains against brought forward business losses",
        "intent": [],
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

    assert result == {
        "original_query": "some query", "search_query": "some query", "intent": [], "filters": {},
    }


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_response_is_none():
    """A provider can return a null/empty completion for `content`; json.loads(None)
    raises TypeError (not JSONDecodeError) and must still degrade to the fallback
    instead of propagating an unhandled exception."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = None

    result = await extract_intent(gateway, "some query")

    assert result == {
        "original_query": "some query", "search_query": "some query", "intent": [], "filters": {},
    }


@pytest.mark.asyncio
async def test_extract_intent_system_prompt_includes_schema_context_and_new_fields():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

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
    for category in ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]:
        assert f'"{category}"' in system_message["content"]


@pytest.mark.asyncio
async def test_extract_intent_emits_intent_step_when_on_step_given():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "original query normalized",
        "intent": ["caselaws"],
        "filters": {"act": "CGST Act"},
    })
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await extract_intent(gateway, "original query normalized", on_step=on_step)

    assert result == {
        "original_query": "original query normalized",
        "search_query": "original query normalized",
        "intent": ["caselaws"],
        "filters": {},
    }
    assert steps == [("intent", {
        "query": "original query normalized",
        "original_query": "original query normalized",
        "search_query": "original query normalized",
        "intent": ["caselaws"],
        "filters": {},
    })]


@pytest.mark.asyncio
async def test_extract_intent_skips_on_step_when_none():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    result = await extract_intent(gateway, "q")  # no on_step passed

    assert result == {"original_query": "q", "search_query": "q", "intent": [], "filters": {}}


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_act_and_preserves_legal_identifier():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "case law for Bharatiya Nyaya Sanhita about scrap sale",
        "intent": ["caselaws"],
        "filters": {},
    })

    result = await extract_intent(gateway, "80HH scrap sale yes useless drum sale no")

    assert result["search_query"] == "80HH scrap sale yes useless drum sale no"


@pytest.mark.asyncio
async def test_extract_intent_rejects_expansion_of_ambiguous_acronym():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "software royalty Profit and Excess India USA DTAA",
        "intent": [],
        "filters": {},
    })

    result = await extract_intent(gateway, "software royalty PE India USA DTAA")

    assert result["search_query"] == "software royalty PE India USA DTAA"


@pytest.mark.asyncio
async def test_extract_intent_drops_unknown_null_and_empty_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "trade training takeover Kolkata section 37(1)",
        "intent": ["acts"],
        "filters": {"city": "Kolkata", "act": None, "court": "", "section": "37(1)"},
    })

    result = await extract_intent(gateway, "trade training takeover Kolkata section 37(1)")

    # "section" is dropped unconditionally now (see Task 2 Step 4), independent
    # of what "unknown null and empty filters" covers - only "city" (unrecognized
    # key), "act": None, and "court": "" are this test's actual subject.
    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_extracts_bench_and_judge_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "ruling of the Principal Bench of the Income Tax Appellate Tribunal on Modvat credit authored by Judge D.Y. Chandrachud",
        "intent": ["caselaws"],
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
        "search_query": "Modvat credit ruling",
        "intent": ["caselaws"],
        "filters": {"bench": "Principal Bench", "judge": "D.Y. Chandrachud"},
    })

    result = await extract_intent(gateway, "Modvat credit ruling")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_drops_unrecognized_category_values():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "q", "intent": ["acts", "not_a_real_category"], "filters": {},
    })

    result = await extract_intent(gateway, "q")

    assert result["intent"] == ["acts"]


@pytest.mark.asyncio
async def test_extract_intent_dedupes_category_values():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "q", "intent": ["acts", "acts", "caselaws"], "filters": {},
    })

    result = await extract_intent(gateway, "q")

    assert sorted(result["intent"]) == ["acts", "caselaws"]


@pytest.mark.asyncio
async def test_extract_intent_accepts_multi_label_category():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "case law on section 54F exemption eligibility",
        "intent": ["acts", "caselaws"],
        "filters": {},
    })

    result = await extract_intent(gateway, "case law on section 54F exemption eligibility")

    assert sorted(result["intent"]) == ["acts", "caselaws"]


@pytest.mark.asyncio
async def test_extract_intent_accepts_each_allowed_category_label():
    for label in ["acts", "rules", "caselaws", "articles", "commentary", "tariff"]:
        gateway = AsyncMock()
        gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [label], "filters": {}})

        result = await extract_intent(gateway, "q")

        assert result["intent"] == [label]


@pytest.mark.asyncio
async def test_extract_intent_accepts_empty_category_list():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    result = await extract_intent(gateway, "q")

    assert result["intent"] == []


@pytest.mark.asyncio
async def test_extract_intent_falls_back_when_shape_is_invalid():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": None,
        "intent": "acts",  # not a list - malformed shape
        "filters": "none",
    })

    result = await extract_intent(gateway, "original")

    assert result == {
        "original_query": "original", "search_query": "original", "intent": [], "filters": {},
    }


@pytest.mark.asyncio
async def test_extract_intent_rejects_invented_year_and_court():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "income tax deduction in 2024 decided by Delhi High Court",
        "intent": ["caselaws"],
        "filters": {},
    })

    result = await extract_intent(gateway, "income tax deduction")

    assert result["search_query"] == "income tax deduction"


@pytest.mark.asyncio
async def test_extract_intent_preserves_user_supplied_year_and_section_numbers():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "income tax section 80HH deduction in 1985",
        "intent": ["acts"],
        "filters": {"section": "80HH"},
    })

    result = await extract_intent(gateway, "1985 income tax section 80HH deduction")

    assert result["search_query"] == "income tax section 80HH deduction in 1985"
    # "section" is dropped unconditionally now, regardless of category (Task 2 Step 4).
    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_always_drops_section_filter_regardless_of_category():
    """The old intent!=provision_lookup gate compared against a value that no
    longer exists post category-rewrite - made explicit instead: "section" is
    now unconditionally stripped from filters, for every category including
    "acts" (where the old gate would have kept it)."""
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "section 92C text",
        "intent": ["acts"],
        "filters": {"section": "92C"},
    })

    result = await extract_intent(gateway, "section 92C text")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_rejects_lossy_rewrite():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "sale of scrap under 80HH",
        "intent": ["acts"],
        "filters": {},
    })

    query = "80HH scrap sale yes useless drum sale no metallic wire factory"
    result = await extract_intent(gateway, query)

    assert result["search_query"] == query


@pytest.mark.asyncio
async def test_extract_intent_requests_model_for_slm_role():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    await extract_intent(gateway, "q")

    gateway.get_model.assert_awaited_once_with(role="slm")


@pytest.mark.asyncio
async def test_extract_intent_uses_llama_tuned_prompt_for_llama_model():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

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
    gateway.chat.return_value = json.dumps({"search_query": "q", "intent": [], "filters": {}})

    await extract_intent(gateway, "q")

    assert captured.get("level") == "WARNING"


@pytest.mark.asyncio
async def test_extract_intent_drops_non_iso_or_invented_date_filters():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "income tax cases",
        "intent": ["caselaws"],
        "filters": {"date_range": {"gte": "not specified", "lte": "2024-12-31"}},
    })

    result = await extract_intent(gateway, "income tax cases")

    assert result["filters"] == {}


@pytest.mark.asyncio
async def test_extract_intent_forwards_model_override_and_skips_get_model():
    gateway = AsyncMock()
    gateway.chat.return_value = json.dumps({
        "search_query": "candidate model test",
        "intent": [],
        "filters": {},
    })

    await extract_intent(gateway, "candidate model test", model="google/gemma-4-E4B-it")

    gateway.get_model.assert_not_awaited()
    call_kwargs = gateway.chat.await_args.kwargs
    assert call_kwargs["model"] == "google/gemma-4-E4B-it"


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


def test_build_chunk_context_drops_stopword_led_court_city_spans():
    """merge_court_city merges any token immediately before court/high/tribunal,
    including stopwords ("the court", "the tribunal") - injecting these as
    authoritative structural spans nudges the SLM toward a bogus court filter
    that _sanitize_filters can't catch (its substring check trivially passes
    a stopword phrase that IS literally in the query)."""
    result = _build_chunk_context("assessee approached the court after the tribunal order")

    spans = json.loads(result) if result is not None else []
    texts = [span["text"] for span in spans]
    assert "the court" not in texts
    assert "the tribunal" not in texts


def test_build_chunk_context_does_not_escape_non_ascii_text():
    """json.dumps defaults to ensure_ascii=True, which would \\uXXXX-escape a
    non-ASCII span - undercutting the "already present in the query above"
    framing since the injected block would then visually diverge from the
    query text shown right above it."""
    result = _build_chunk_context('"café royalty" ruling')

    assert result is not None
    assert "café royalty" in result
    assert "\\u" not in result


def test_build_chunk_context_keeps_real_court_name_spans():
    result = _build_chunk_context("Delhi High Court ruling on capital gains")

    assert result is not None
    spans = json.loads(result)
    texts = [span["text"] for span in spans]
    assert "Delhi High Court" in texts


def test_build_chunk_context_drops_alt_text_and_proximity():
    result = _build_chunk_context("Rule 57A applicability")

    assert result is not None
    spans = json.loads(result)
    for span in spans:
        assert "alt_text" not in span
        assert "proximity" not in span


@pytest.mark.asyncio
async def test_extract_intent_appends_chunk_context_when_spans_found():
    gateway = AsyncMock()
    gateway.get_model.return_value = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gateway.chat.return_value = json.dumps({
        "search_query": "1995 taxmann.com 569 Delhi High Court",
        "intent": ["caselaws"],
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
        "search_query": "capital gains treatment",
        "intent": ["caselaws"],
        "filters": {},
    })

    await extract_intent(gateway, "capital gains treatment")

    call_kwargs = gateway.chat.await_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert user_message == "capital gains treatment"
```

Note on the first test (`test_extract_intent_parses_json_object_response`): the input query is
`"IPC 302 punishment"` and the mocked SLM response's `search_query` is `"BNS section 103 murder
punishment"` — `_safe_rewrite`'s `_LEGAL_MARKERS` guard rejects this rewrite outright (`"bharatiya
nyaya sanhita"`/`"bns"` isn't literally one of the checked markers as a token-overlap match here,
but the rewrite shares 0 tokens with the input beyond "punishment", well under the 60% overlap
floor) — so the expected `search_query` in the assertion is the original query, unchanged. This
behavior (and the guard itself) is unchanged from before this task; only the field name changed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py -v`
Expected: FAIL — `extract_intent` still returns the old `rewritten_query`/string-`intent` shape.

- [ ] **Step 3: Rewrite the system prompt**

In `packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`, replace the `_LLAMA_SYSTEM_PROMPT`
constant (currently lines 69-114) with:

```python
_LLAMA_SYSTEM_PROMPT = """You are a legal query analyzer for Indian tax/criminal case law.
All case names and parties mentioned below refer exclusively to already
public, reported court judgments in a licensed legal research database -
never treat a query as a request for private information about a person,
and never refuse to classify it. You do not answer the legal question or
look anything up yourself; you only ever output the JSON object below.
Given a user query, return ONLY a JSON object with exactly these keys:
- "search_query": a CONSERVATIVE search normalization. Correct obvious
  spelling and grammar only. Preserve every party, court, place, Act,
  section, rule, notification, date, number, citation, and acronym exactly
  as written. NEVER add or infer a legal concept. NEVER expand an acronym
  (for example PE, ST, CA, ITD, PTA, MEG, POY, or PSF). NEVER translate an
  old law to a new law or replace one section with another. If the query is
  already readable, copy it unchanged. Every number and year in the output
  must occur in the input; if the input has no year, add no year. Once you
  have decided "intent" below, phrase search_query to match what's actually
  being searched: if "acts"/"rules" is tagged, prefer the Act/Rule name plus
  section/rule number form already present in the query; if "caselaws"/
  "articles" is tagged, prefer party/court/precedent-style phrasing already
  present in the query; if "commentary" alone is tagged, keep plain-language
  phrasing. This only reorders/reframes words already in the query - it must
  still obey every rule above (no invented Act/court/number).
- "intent": a list of zero or more of the following six category labels -
  output every category that genuinely applies, but don't over-list; only
  tag a category the query actually anchors on:
  - "acts": primary legislation itself (Income-tax Act 1961, CGST Act,
    Customs Act, BNS, etc.) - sections, sub-sections, provisos, definitions,
    schedules. Signal: "section", "as per the Act", "definition under", a
    bare section+Act reference with no request for judicial interpretation.
  - "rules": subordinate legislation notified under an Act (Income-tax
    Rules 1962, CGST Rules, Customs Valuation Rules) - procedure,
    computation mechanics, prescribed forms. Distinct from "acts" by
    whether the query's number is a "rule" vs a "section"; a rules query
    often co-occurs with "acts" since every Rule has a parent Act.
  - "caselaws": judicial decisions (Supreme Court, High Courts, ITAT,
    CESTAT, AAR) - what was decided for a dispute/fact pattern. Signal:
    party names ("X vs Y"), "held", "case law on", "precedent for", a
    citation string, bench/judge name.
  - "articles": expert-authored opinion/analysis published in a journal or
    magazine - trend, controversy, recent development, practical impact.
    Not the publisher's own explanation (that's "commentary") and not
    binding law. Tag only on explicit signal ("article on...", "expert
    opinion on...", a named author) - don't default here.
  - "commentary": the publisher's own provision-by-provision plain-language
    explanation of how a section/Act/rule works in practice, no author
    byline. Distinct from "acts" (raw statutory text) and "articles" (named
    author's opinion piece). Default landing spot for "explain X" / "how
    does X work" queries that aren't clearly "articles".
  - "tariff": customs/GST tariff classification and rates - HSN code
    lookups, duty rates, rate schedules, exemption notifications tied to a
    specific tariff heading/good. Distinct from "acts"/"rules" even though
    tariff notifications are issued under that law - if the ask is "what
    HSN/duty rate for [a specific good]", it's "tariff".
  Output an empty list when no category confidently applies. Never output
  any other value.
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
exactly {"party": "Ramesh Gupta"} and intent is ["caselaws"].

Example: query "case law on section 54F exemption eligibility" anchors on
both a case-law request and a specific Act section, so intent is
["acts", "caselaws"].

Forbidden rewrites:
- "80HH scrap sale" must not mention BNS or any other Act.
- "software royalty PE" must retain "PE" without guessing its expansion.
- "69C diamond cash sale" must not add CGST Act or replace section 69C.
- "59/98-ST certification" must not add Customs Act.

""" + build_schema_context()
```

- [ ] **Step 4: Update `_ALLOWED_INTENTS`, `_sanitize_filters`, `_validate_result`, `_fallback_intent`**

In the same file:

1. Replace the `_ALLOWED_FILTERS`/`_ALLOWED_INTENTS`/`_LEGAL_MARKERS` block (currently lines
   136-143) — rename `_ALLOWED_INTENTS` to `_ALLOWED_CATEGORIES` with the new value set:

```python
_ALLOWED_FILTERS = {"court", "act", "section", "date_range", "party", "bench", "judge"}
_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}
_LEGAL_MARKERS = {
    "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
    "bharatiya sakshya adhiniyam", "indian penal code", "income-tax act",
    "income tax act", "cgst act", "igst act", "customs act",
    "code of criminal procedure", "indian evidence act",
}
```

2. In `_sanitize_filters`, remove the `intent: str` parameter and the `if key == "section" and
   intent != "provision_lookup": continue` block (currently lines 172, 191-201 partially — keep
   the `date_range` handling and the literal-substring check, just drop the intent parameter and
   replace the conditional section-skip with an unconditional one). Full replacement:

```python
def _sanitize_filters(query: str, filters) -> dict:
    if not isinstance(filters, dict):
        return {}
    clean = {}
    for key, value in filters.items():
        if key not in _ALLOWED_FILTERS:
            continue
        # "section" is unconditionally dropped: it only resolves correctly against
        # ACT/RULE-group documents whose heading IS the section number verbatim (see
        # es_client.py::_section_heading_queries) - not case law that merely cites the
        # section. The old gate compared against intent=="provision_lookup", a value
        # that doesn't exist post category-rewrite (intent is now a category list, not
        # that 4-value enum) - rather than leave that comparison silently always-false,
        # it's made explicit here. Confirmed live (pre-rewrite): a conceptual query with
        # a bare "section 92C" filter went from 70 unfiltered Milvus hits (including the
        # gold doc) to 0 filtered hits. Revisit once section-filter gating is rebuilt
        # around category (not part of this change - see
        # docs/superpowers/specs/2026-08-14-category-collection-routing-design.md).
        if key == "section":
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
```

3. Replace `_validate_result` and `_fallback_intent`:

```python
def _fallback_intent(query: str) -> dict:
    """Used when the SLM refuses or returns unparseable output (e.g. Llama's
    safety training treating "case law for X vs. Y" as a request for private
    info about a named person) - degrade to a plain semantic search instead
    of failing the whole AI Mode request."""
    return {"original_query": query, "search_query": query, "intent": [], "filters": {}}


def _validate_categories(intent) -> list[str]:
    if not isinstance(intent, list):
        return []
    seen: list[str] = []
    for value in intent:
        if isinstance(value, str) and value in _ALLOWED_CATEGORIES and value not in seen:
            seen.append(value)
    return seen


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

4. In `extract_intent`, the `on_step` call already spreads `**result` — no change needed there
   beyond what the new `result` dict shape already provides. Confirm the function body still
   reads (only the `on_step` payload's `query` key stays for backward-compat with existing trace
   consumers, alongside the new `original_query`):

```python
    if on_step is not None:
        await on_step("intent", {"query": query, **result})
```

(This line is unchanged from before — `query` and `result["original_query"]` are always equal by
construction, kept both for now since existing trace consumers may key on `query`.)

- [ ] **Step 5: Update `schema_context.py` wording**

In `packages/common/src/common/schema_context.py`, line 38, change:

```python
        "Searchable collections (all are searched together, phrase rewritten_query "
        "to read naturally against each of them):\n"
```

to:

```python
        "Searchable collections (searched together unless routed by category, "
        "phrase search_query to read naturally against each of them):\n"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_intent.py packages/common/tests/test_schema_context.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/intent.py \
        packages/retrieval-api/tests/test_ai_mode_intent.py \
        packages/common/src/common/schema_context.py
git commit -m "feat: replace 4-value intent enum with multi-label category taxonomy"
```

---

### Task 3: `retrieve.py` — drop RRF weighting, add collection routing

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_retrieve.py` (rewrite)

**Interfaces:**
- Consumes: `common.schemas.collections_for_intent(intent: list[str]) -> list[str]` (Task 1).
- Produces: `retrieve(gateway, milvus_client, search_query: str, doc_id_allowlist, intent:
  list[str] = [], on_step=None) -> list[dict]` — note the parameter rename `rewritten_query` →
  `search_query` and `intent` default changes from `"unknown"` (str) to `[]` (list). Consumed by
  Task 4 (`pipeline.py`) and Task 5 (`retrieval_eval.py`).

- [ ] **Step 1: Replace the test file**

Replace `packages/retrieval-api/tests/test_ai_mode_retrieve.py` in full with:

```python
from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.retrieve import rrf_merge, retrieve


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


def test_rrf_merge_default_weights_are_neutral():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_upweights_dense_list_over_sparse_when_explicitly_passed():
    # rrf_merge itself still accepts explicit weights (used by callers other than
    # retrieve(), and by these direct unit tests) - only retrieve() no longer
    # resolves non-neutral weights from intent.
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=1.5, sparse_weight=0.5)

    assert merged[0]["chunk_id"] == "a"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


def test_rrf_merge_upweights_sparse_list_over_dense_when_explicitly_passed():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=0.5, sparse_weight=1.5)

    assert merged[0]["chunk_id"] == "c"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


@pytest.mark.asyncio
async def test_retrieve_embeds_search_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await retrieve(gateway, milvus_client=object(), search_query="q", doc_id_allowlist=["d1"])

    gateway.embed.assert_awaited_once_with(role="query_embed", text="q")
    assert result[0]["chunk_id"] == "a"


def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = collection_trace({"ruling": rows})

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

    result = await module.retrieve(gateway, milvus_client=object(), search_query="q", doc_id_allowlist=None, on_step=on_step)

    assert steps == ["milvus_dense", "milvus_sparse", "rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}


@pytest.mark.asyncio
async def test_retrieve_always_uses_neutral_rrf_weighting(monkeypatch):
    """Category no longer drives RRF weighting at all (rejected during
    brainstorming - see docs/superpowers/specs/2026-08-14-category-collection-
    routing-design.md). Every intent value, including ones that would have
    skewed weighting under the old 4-value enum, must resolve to (1.0, 1.0)."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await module.retrieve(
        gateway, milvus_client=object(), search_query="q", doc_id_allowlist=None,
        intent=["caselaws"], on_step=on_step,
    )

    rrf_step = next(data for step, data in steps if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 1.0
    assert rrf_step["sparse_weight"] == 1.0


@pytest.mark.asyncio
async def test_retrieve_routes_collections_by_intent(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    await module.retrieve(
        gateway, milvus_client=object(), search_query="q", doc_id_allowlist=None, intent=["acts"],
    )

    assert seen_collections == [["act_section"], ["act_section"]]  # dense pass, sparse pass


@pytest.mark.asyncio
async def test_retrieve_defaults_to_all_collections_when_intent_omitted(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module
    from common.schemas import MILVUS_COLLECTIONS

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    await module.retrieve(gateway, milvus_client=object(), search_query="q", doc_id_allowlist=None)

    assert seen_collections[0] == MILVUS_COLLECTIONS


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_unfiltered_when_allowlist_zeroes_everything(monkeypatch):
    """A resolved doc_id_allowlist that's non-empty but wrong-typed/disjoint from the
    target Milvus collections must not silently return zero candidates when an
    unfiltered search would find real matches - retry once unfiltered instead,
    within the same routed collection set."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        if doc_id_allowlist is not None:
            return {"ruling": []}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await module.retrieve(
        gateway, milvus_client=object(), search_query="q",
        doc_id_allowlist=["wrong-doc-id"], intent=["caselaws"], on_step=on_step,
    )

    assert result[0]["chunk_id"] == "a"
    assert [step for step, _ in steps] == ["filter_fallback", "milvus_dense", "milvus_sparse", "rrf_merge"]
    fallback_data = next(data for step, data in steps if step == "filter_fallback")
    assert fallback_data["doc_id_allowlist_count"] == 1
    gateway.embed.assert_awaited_once()  # retry reuses the already-computed embedding
    # every hybrid_search call (both the initial pair and the retry pair) used the
    # same routed collection set - the retry drops the allowlist, not the routing.
    assert all(collections == ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"] for collections in seen_collections)


@pytest.mark.asyncio
async def test_retrieve_does_not_fall_back_when_no_allowlist_was_applied(monkeypatch):
    """Zero hits with no allowlist at all is just a genuinely empty result, not the
    disjoint-allowlist failure mode - must not trigger a pointless retry."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    calls = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        calls.append(doc_id_allowlist)
        return {"ruling": []}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await module.retrieve(gateway, milvus_client=object(), search_query="q", doc_id_allowlist=None)

    assert result == []
    assert len(calls) == 2  # dense + sparse, no retry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: FAIL — old `rewritten_query` parameter name, no routing, `_INTENT_RRF_WEIGHTS` still present.

- [ ] **Step 3: Rewrite `retrieve.py`**

Replace `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py` in full:

```python
from common.milvus_client import hybrid_search
from common.schemas import collections_for_intent
from retrieval_api.ai_mode.intent import OnStep
from retrieval_api.gateway_client import GatewayClient
from retrieval_api.trace_utils import collection_trace


def rrf_merge(
    dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60,
    dense_weight: float = 1.0, sparse_weight: float = 1.0,
) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list, weight in ((dense_ranked, dense_weight), (sparse_ranked, sparse_weight)):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]


def _flatten(by_collection: dict[str, list[dict]]) -> list[dict]:
    flattened = [row for rows in by_collection.values() for row in rows]
    return sorted(flattened, key=lambda row: row["score"], reverse=True)


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    search_query: str,
    doc_id_allowlist: list[str] | None,
    intent: list[str] | None = None,
    on_step: OnStep | None = None,
) -> list[dict]:
    collections = collections_for_intent(intent or [])

    dense_vector = await gateway.embed(role="query_embed", text=search_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=collections, dense_vector=dense_vector,
        sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    sparse_by_collection = await hybrid_search(
        milvus_client, collections=collections, dense_vector=None,
        sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )

    # Circuit breaker: a resolved doc_id_allowlist that's non-empty but the wrong kind of
    # document for these collections silently zeroes every collection even though an
    # unfiltered search would find real matches. If the allowlist was non-empty but
    # produced zero hits everywhere, retry once unfiltered rather than returning nothing -
    # the embedding is already computed, so this only costs the two Milvus round-trips.
    # Retries against the SAME routed collection set - a routed-but-genuinely-wrong-
    # category query should surface as zero results, not silently widen to every
    # collection (that would defeat the point of routing).
    if doc_id_allowlist and not any(dense_by_collection.values()) and not any(sparse_by_collection.values()):
        if on_step is not None:
            await on_step("filter_fallback", {
                "reason": "doc_id_allowlist matched zero Milvus results across every routed collection; retrying unfiltered",
                "doc_id_allowlist_count": len(doc_id_allowlist),
            })
        dense_by_collection = await hybrid_search(
            milvus_client, collections=collections, dense_vector=dense_vector,
            sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
        )
        sparse_by_collection = await hybrid_search(
            milvus_client, collections=collections, dense_vector=None,
            sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
        )

    if on_step is not None:
        await on_step("milvus_dense", collection_trace(dense_by_collection))
        await on_step("milvus_sparse", collection_trace(sparse_by_collection))

    # RRF fusion weight is always neutral - category does not drive dense/sparse
    # weighting (considered during brainstorming, explicitly rejected; see
    # docs/superpowers/specs/2026-08-14-category-collection-routing-design.md).
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
        await on_step("rrf_merge", {
            "candidate_count": len(merged), "top_candidates": top_candidates,
            "dense_weight": 1.0, "sparse_weight": 1.0,
        })

    return merged
```

Note this drops `common.config.get_settings` and the `intent_rrf_weighting_enabled` setting
entirely from this file (no longer referenced — weighting is unconditionally neutral now, no
kill switch needed since there's nothing to switch off).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py \
        packages/retrieval-api/tests/test_ai_mode_retrieve.py
git commit -m "feat: route Milvus collections by intent category, drop RRF weighting"
```

---

### Task 4: `pipeline.py` — field renames, pass intent list through

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Consumes: `extract_intent()`'s new shape (Task 2), `retrieve()`'s new signature (Task 3).
- Produces: no change to `run_ai_mode()`'s own signature or return shape.

- [ ] **Step 1: Update the test file**

In `packages/retrieval-api/tests/test_ai_mode_pipeline.py`, apply these exact replacements
throughout the file:

- Every fake `extract_intent`'s returned dict: change `{"rewritten_query": "rewritten", "intent":
  "conceptual", "filters": {}}` (and similar with different placeholder values) to use
  `"search_query"` instead of `"rewritten_query"`, and a list for `"intent"` instead of a bare
  string (e.g. `"intent": "conceptual"` → `"intent": ["caselaws"]`, `"intent": "x"` → `"intent":
  ["caselaws"]`).
- Every fake `retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, intent,
  on_step=None)` signature: rename the `rewritten_query` parameter to `search_query`.
- `test_run_ai_mode_success_path`: change `assert received_intent["value"] == "conceptual"` to
  `assert received_intent["value"] == ["caselaws"]`.
- `test_run_ai_mode_emits_all_seven_trace_steps_in_order_end_to_end`'s `fake_chat` return value:
  change `'{"rewritten_query": "rewritten query", "intent": "conceptual", "filters": {}}'` to
  `'{"search_query": "rewritten query", "intent": ["caselaws"], "filters": {}}'`. Its final
  assertion block (`rrf_step["dense_weight"] == 1.5` / `rrf_step["sparse_weight"] == 0.5`) must
  change to `rrf_step["dense_weight"] == 1.0` and `rrf_step["sparse_weight"] == 1.0` — weighting
  is neutral now regardless of category (Task 3).

Concretely, `test_run_ai_mode_success_path` becomes:

```python
@pytest.mark.asyncio
async def test_run_ai_mode_success_path(monkeypatch):
    import retrieval_api.ai_mode.pipeline as module

    async def fake_extract_intent(gateway, query, on_step=None):
        return {"original_query": query, "search_query": "rewritten", "intent": ["caselaws"], "filters": {}}

    async def fake_resolve_allowlist(es_client, filters, on_step=None):
        return None

    received_intent = {}

    async def fake_retrieve(gateway, milvus_client, search_query, doc_id_allowlist, intent, on_step=None):
        received_intent["value"] = intent
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
    assert received_intent["value"] == ["caselaws"]
```

Apply the same `search_query`/list-`intent` pattern to `test_run_ai_mode_succeeds_with_party_only_filter`
and `test_run_ai_mode_forwards_on_step_to_every_stage`'s fake functions (their assertions don't
reference `intent`'s value directly, only the field names in the fakes need updating).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: FAIL — `pipeline.py` still passes `intent_result["rewritten_query"]`.

- [ ] **Step 3: Update `pipeline.py`**

In `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`, change lines 26-32 from:

```python
            with langfuse.start_as_current_observation(
                as_type="chain", name="retrieve", input={"rewritten_query": intent_result["rewritten_query"]},
            ) as span:
                candidates = await retrieve(
                    gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist,
                    intent_result["intent"], on_step=on_step,
                )
                span.update(output={"num_candidates": len(candidates)})
```

to:

```python
            with langfuse.start_as_current_observation(
                as_type="chain", name="retrieve", input={"search_query": intent_result["search_query"]},
            ) as span:
                candidates = await retrieve(
                    gateway, milvus_client, intent_result["search_query"], doc_id_allowlist,
                    intent_result["intent"], on_step=on_step,
                )
                span.update(output={"num_candidates": len(candidates)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py \
        packages/retrieval-api/tests/test_ai_mode_pipeline.py
git commit -m "refactor: thread renamed search_query field through pipeline.py"
```

---

### Task 5: `retrieval_eval.py` — field renames, drop weighting, route the rewritten-query branch

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`
- Test: `packages/retrieval-api/tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `extract_intent()`'s new shape (Task 2), `collections_for_intent()` (Task 1).
- Produces: `evaluate_case()`'s return dict still has a `"rewritten_query"` key for backward
  compatibility with existing result-file readers — **keep this key name in the output dict**,
  only rename the internal variable and the field read from `extract_intent`'s result. This is a
  deliberate exception to the field rename: `evaluate_case`'s JSON output is a persisted eval
  artifact format, not an internal API — renaming it would break every existing
  `.eval-results/*.json` file's schema for no benefit to this task.

- [ ] **Step 1: Update the test file**

In `packages/retrieval-api/tests/test_retrieval_eval.py`, every fake `extract_intent` currently
returns `{"rewritten_query": ..., "filters": {}, "intent": "test"}` (or similar) — change the
key it reads from to `"search_query"` and make `"intent"` a list, e.g. `{"search_query": query,
"filters": {}, "intent": ["caselaws"]}`. The test still asserts on `result["rewritten_query"]` in
`evaluate_case`'s *return value* (Interfaces note above) — that assertion does not change.

Rewrite `test_evaluate_case_threads_intent_rrf_weights_into_rrf_merge`
(`packages/retrieval-api/tests/test_retrieval_eval.py:122`) — this test's entire premise (category
drives RRF weighting) is gone. Replace it with:

```python
@pytest.mark.asyncio
async def test_evaluate_case_always_uses_neutral_rrf_weighting(monkeypatch):
    """Category no longer drives RRF weighting (dropped from retrieve.py in
    Task 3) - evaluate_case must mirror that: always neutral (1.0, 1.0)
    regardless of what extract_intent classifies."""
    import retrieval_api.retrieval_eval as module

    case = {
        "id": "T1", "class": "direct", "query": "q",
        "gold_doc_ids": ["d1"], "expected_collections": ["ruling"], "pass_at": 10,
    }

    async def fake_raw_search(client, query, limit=50):
        return []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "t2", "score": 5.0}]}

    async def fake_embed(role, text):
        return [0.1, 0.2]

    gateway = AsyncMock()
    gateway.embed.side_effect = fake_embed

    async def fake_intent(gateway, query, model=None):
        return {"search_query": query, "filters": {}, "intent": ["caselaws"]}

    monkeypatch.setattr(module, "raw_search", fake_raw_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "extract_intent", fake_intent)
    monkeypatch.setattr(module, "resolve_allowlist", AsyncMock(return_value=None))
    monkeypatch.setattr(module, "rerank_top_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(module, "run_agentic_search", AsyncMock(return_value=None))

    result = await module.evaluate_case(
        case, gateway, es_client=object(), milvus_client=object(), langfuse_enabled=False, skip_agentic=True,
    )

    # dense-only chunk "a" and sparse-only chunk "c" are both rank-1 in their
    # list - neutral weighting must tie them, so RRF's own dedupe/ordering
    # (not a weight skew) decides. Assert the merged ranks list contains both
    # rather than asserting a specific winner (a tie's iteration order isn't
    # this test's subject).
    assert result["ranks"]["rrf"] in (1, 2)
```

Check the existing full test file at `packages/retrieval-api/tests/test_retrieval_eval.py` for
the exact imports/fixtures already in scope (e.g. `AsyncMock` is already imported at the top —
confirm before adding a duplicate import) and match this new test's style/helpers to what's
already there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: FAIL — old field names, `_INTENT_RRF_WEIGHTS` import still present.

- [ ] **Step 3: Update `retrieval_eval.py`**

In `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`:

1. Line 25, change the import from:

```python
from retrieval_api.ai_mode.retrieve import _flatten, _INTENT_RRF_WEIGHTS, rrf_merge
```

to:

```python
from common.schemas import collections_for_intent
from retrieval_api.ai_mode.retrieve import _flatten, rrf_merge
```

2. Lines 186-207 (the intent/rewrite/RRF block inside `evaluate_case`), change from:

```python
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

            dense_weight, sparse_weight = _INTENT_RRF_WEIGHTS.get(intent.get("intent"), (1.0, 1.0)) if intent else (1.0, 1.0)
            merged = rrf_merge(
                _flatten(rewritten_dense), _flatten(rewritten_sparse),
                dense_weight=dense_weight, sparse_weight=sparse_weight,
            )
```

to:

```python
            intent = await measured("intent", extract_intent(gateway, query, model=slm_model))
            rewritten_query = intent.get("search_query", query) if intent else query
            routed_collections = collections_for_intent(intent.get("intent") or []) if intent else MILVUS_COLLECTIONS
            allowlist = await measured("filters", resolve_allowlist(es_client, intent.get("filters", {}))) if intent else None
            rewritten_vector = raw_vector if rewritten_query == query else await measured(
                "rewritten_embedding", gateway.embed(role="query_embed", text=rewritten_query),
            )
            rewritten_dense = (
                await measured("rewritten_dense", hybrid_search(
                    milvus_client, routed_collections, rewritten_vector, rewritten_query,
                    doc_id_allowlist=allowlist, limit=limit,
                )) if rewritten_vector is not None else None
            ) or {name: [] for name in routed_collections}
            rewritten_sparse = await measured("rewritten_sparse", hybrid_search(
                milvus_client, routed_collections, None, rewritten_query,
                doc_id_allowlist=allowlist, limit=limit,
            )) or {name: [] for name in routed_collections}

            # RRF fusion weight is always neutral - category does not drive
            # dense/sparse weighting (see Task 3 / the routing design spec).
            merged = rrf_merge(_flatten(rewritten_dense), _flatten(rewritten_sparse))
```

Note: `raw_dense`/`raw_sparse` (the unrewritten-query baseline, computed earlier in the same
function against the raw `query` before intent classification even runs) are **not** routed —
they stay against `MILVUS_COLLECTIONS` unconditionally, since they exist purely as a
before/after-rewrite diagnostic baseline, not the production retrieval path. Only the
`rewritten_*` branch (which mirrors what `retrieve.py`/`pipeline.py` actually do in production)
gets routed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_retrieval_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/retrieval_eval.py \
        packages/retrieval-api/tests/test_retrieval_eval.py
git commit -m "refactor: route retrieval_eval.py's rewritten-query branch by category, drop RRF weighting"
```

---

### Task 6: `intent_eval.py` + `evals/intent_filter_cases.json` — category accuracy scoring

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/intent_eval.py`
- Modify: `evals/intent_filter_cases.json`
- Test: `packages/retrieval-api/tests/test_intent_eval.py`

**Interfaces:**
- Consumes: `extract_intent()`'s new shape (Task 2).
- Produces: `load_intent_cases` now requires `{"id", "query", "expected_filters",
  "expected_categories"}`; `check_intent_case` gains a second parameter.

- [ ] **Step 1: Update the test file**

Replace `packages/retrieval-api/tests/test_intent_eval.py` in full:

```python
import json
from pathlib import Path

import pytest

from retrieval_api.intent_eval import check_intent_case, load_intent_cases


def test_repository_intent_filter_dataset_has_cases_and_unique_ids():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    assert len(cases) >= 12
    assert len({case["id"] for case in cases}) == len(cases)


def test_repository_intent_filter_dataset_covers_every_category():
    root = Path(__file__).parents[3]
    cases = load_intent_cases(root / "evals" / "intent_filter_cases.json")

    covered = {category for case in cases for category in case["expected_categories"]}
    assert covered == {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}


def test_load_intent_cases_validates_required_keys(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "F1", "query": "q"}]))

    with pytest.raises(ValueError, match="missing"):
        load_intent_cases(path)


def test_load_intent_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"id": "F1", "query": "q1", "expected_filters": {}, "expected_categories": []},
        {"id": "F1", "query": "q2", "expected_filters": {}, "expected_categories": []},
    ]))

    with pytest.raises(ValueError, match="duplicate"):
        load_intent_cases(path)


def test_check_intent_case_matches_exact_filters_and_categories():
    assert check_intent_case(
        {"court": "Bombay High Court"}, {"court": "Bombay High Court"}, ["caselaws"], ["caselaws"],
    ) == (True, True)


def test_check_intent_case_flags_filter_mismatch_independently_of_category():
    filters_ok, categories_ok = check_intent_case(
        {"court": "Bombay High Court"}, {}, ["caselaws"], ["caselaws"],
    )
    assert filters_ok is False
    assert categories_ok is True


def test_check_intent_case_flags_category_mismatch_independently_of_filters():
    filters_ok, categories_ok = check_intent_case(
        {}, {}, ["acts"], ["caselaws"],
    )
    assert filters_ok is True
    assert categories_ok is False


def test_check_intent_case_category_match_is_order_independent():
    filters_ok, categories_ok = check_intent_case(
        {}, {}, ["acts", "caselaws"], ["caselaws", "acts"],
    )
    assert categories_ok is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_intent_eval.py -v`
Expected: FAIL — `check_intent_case` doesn't accept 4 arguments yet, dataset lacks
`expected_categories`.

- [ ] **Step 3: Update `intent_eval.py`**

Replace `packages/retrieval-api/src/retrieval_api/intent_eval.py` in full:

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
        required = {"id", "query", "expected_filters", "expected_categories"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate query id: {case['id']}")
        seen.add(case["id"])
    return cases


def check_intent_case(
    expected_filters: dict, actual_filters: dict,
    expected_categories: list[str], actual_categories: list[str],
) -> tuple[bool, bool]:
    filters_ok = expected_filters == actual_filters
    categories_ok = set(expected_categories) == set(actual_categories)
    return filters_ok, categories_ok


async def run(gateway_url: str, model: str | None, dataset_path: str | Path) -> None:
    cases = load_intent_cases(dataset_path)
    gateway = GatewayClient(base_url=gateway_url, trace_enabled=False)
    passed = 0
    for case in cases:
        try:
            result = await extract_intent(gateway, case["query"], model=model)
        except Exception as exception:
            print(f"ERROR {case['id']}: {exception}")
            continue
        filters_ok, categories_ok = check_intent_case(
            case["expected_filters"], result["filters"],
            case["expected_categories"], result["intent"],
        )
        ok = filters_ok and categories_ok
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {case['id']}: "
            f"filters(expected={case['expected_filters']} actual={result['filters']} ok={filters_ok}) "
            f"categories(expected={case['expected_categories']} actual={result['intent']} ok={categories_ok})"
        )
    print(f"\n{passed}/{len(cases)} passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only intent/filter/category extraction accuracy check")
    parser.add_argument("--gateway-url", default="http://localhost:8011")
    parser.add_argument("--model", default=None, help="Override the slm role's model")
    parser.add_argument("--dataset", default="evals/intent_filter_cases.json")
    args = parser.parse_args()
    asyncio.run(run(args.gateway_url, args.model, args.dataset))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Extend `evals/intent_filter_cases.json`**

Read the current file first (`evals/intent_filter_cases.json`, 12 cases, `F01`-`F12`) and add
`"expected_categories"` to every existing case, then append 6 new cases so every category and a
multi-label case are covered. The existing 12 are all caselaw-flavored (built for filter
extraction), so most get `["caselaws"]`; check each query's actual wording against the taxonomy
in Task 2 before assigning — don't default all 12 to `["caselaws"]` blindly if any of them reads
as non-case-law. Append these 6 new cases at minimum:

```json
  {"id": "C01", "query": "definition of capital asset under section 2(14)", "expected_filters": {}, "expected_categories": ["acts"]},
  {"id": "C02", "query": "Rule 3 perquisite valuation method", "expected_filters": {}, "expected_categories": ["rules"]},
  {"id": "C03", "query": "article on GST implications of the new e-invoicing mandate", "expected_filters": {}, "expected_categories": ["articles"]},
  {"id": "C04", "query": "explain how section 54F exemption works", "expected_filters": {}, "expected_categories": ["commentary"]},
  {"id": "C05", "query": "HSN code and GST rate for solar panels", "expected_filters": {}, "expected_categories": ["tariff"]},
  {"id": "C06", "query": "case law on section 54F exemption eligibility", "expected_filters": {}, "expected_categories": ["acts", "caselaws"]}
```

Use the existing file's exact JSON formatting/indentation style when merging these in — read the
file first to match it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_intent_eval.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/intent_eval.py \
        packages/retrieval-api/tests/test_intent_eval.py \
        evals/intent_filter_cases.json
git commit -m "feat: extend intent_eval.py to score category accuracy alongside filters"
```

---

### Task 7: CLAUDE.md rule 4 rewrite

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update rule 4**

In `CLAUDE.md`, replace the current rule 4:

```
4. **AI Mode searches all 11 Milvus collections every query.** No intent-based collection
   routing. (7 original + `act_section`/`rule_section`/`article_section`/`commentary_section`
   added 2026-08-11 upstream; `tariff_section` exists in the pipeline schema but is parked in
   `_disabled_collections` — not live, don't add it here yet.)
```

with:

```
4. **AI Mode routes which Milvus collections get searched by the `intent` category tag**
   (`extract_intent()`'s multi-label `acts`/`rules`/`caselaws`/`articles`/`commentary`/`tariff`
   classification), via `collections_for_intent()` (`common/schemas.py`). Empty or
   unrecognized-only `intent` falls back to searching all 11 collections. `tariff_section` has
   no routing entry yet — not live (`_disabled_collections` upstream). RRF fusion weight stays
   neutral (1.0/1.0) regardless of category — this was considered and explicitly rejected during
   design; don't reintroduce category-based dense/sparse weighting. See
   `docs/superpowers/specs/2026-08-13-intent-category-classification-design.md` and
   `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update hard rule 4 for category-based collection routing"
```

---

### Task 8: Full suite verification

**Files:** none — verification only.

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest`
Expected: all packages pass (no regressions in `agents`, `model-gateway`, or any test not
explicitly touched above).

- [ ] **Step 2: Grep for any remaining old-shape references**

Run: `grep -rn "rewritten_query" packages/retrieval-api/src packages/agents/src 2>/dev/null` (or
PowerShell `Select-String` equivalent) — expect only the one deliberate exception in
`retrieval_eval.py`'s output dict key (Task 5 Interfaces note). Also run: `grep -rn
"_INTENT_RRF_WEIGHTS\|provision_lookup\|citation_lookup" packages/ 2>/dev/null` — expect zero
matches anywhere (old enum fully retired).

- [ ] **Step 3: Report**

If both checks are clean, this plan is complete. If either grep finds an unexpected leftover
reference, fix it and re-run the affected package's tests before committing the fix separately.
