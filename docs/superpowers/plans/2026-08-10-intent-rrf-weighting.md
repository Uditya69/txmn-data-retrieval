# Intent-Driven RRF Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `intent` label classified in Phase 1 (`docs/superpowers/specs/2026-08-10-intent-extraction-redesign-design.md`) its first consumer — bias AI Mode's dense/sparse RRF fusion weighting based on classified query intent, without ever touching which Milvus collections are searched.

**Architecture:** `rrf_merge` (in `retrieve.py`) gains optional `dense_weight`/`sparse_weight` multipliers on each ranked list's reciprocal-rank contribution, defaulting to `1.0`/`1.0` (today's exact behavior). A module-level `_INTENT_RRF_WEIGHTS` dict maps each of the 4 intent labels to a `(dense_weight, sparse_weight)` pair; `retrieve()` looks up the caller-supplied `intent` string in it (defaulting to neutral `(1.0, 1.0)` for anything unrecognized) and passes the resolved weights through. `pipeline.py`'s one call site threads `intent_result["intent"]` into `retrieve()`.

**Tech Stack:** Python 3.11, pytest-asyncio.

## Global Constraints

- Python 3.11, not 3.14.
- No intent-based Milvus collection routing, ever (CLAUDE.md hard rule) — this plan only reweights fusion of the existing dense-50/sparse-50 candidate set; it must never change which collections are searched or how many results are fetched (`limit=50` per side stays unchanged).
- No change to `filter_resolve.py`, `_safe_rewrite`, `_sanitize_filters`, or the ES allowlist mechanism.
- Default weights (`1.0`/`1.0`) must exactly preserve today's `rrf_merge` behavior for any caller that doesn't pass weights.
- Run `uv run pytest` from repo root after each task; it aggregates all packages.

---

### Task 1: Weighted `rrf_merge`, intent→weight lookup, and pipeline wiring

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Modify: `packages/retrieval-api/tests/test_ai_mode_retrieve.py`
- Modify: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Produces: `rrf_merge(dense_ranked, sparse_ranked, k=60, dense_weight=1.0, sparse_weight=1.0) -> list[dict]` — same return shape as before (each row gets `rrf_score` added).
- Produces: `retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, intent="unknown", on_step=None) -> list[dict]` — `intent` is a new 5th parameter inserted before `on_step`, with a default so no existing positional-argument caller outside this codebase breaks; `on_step`'s existing trace payload for the `"rrf_merge"` step gains `"dense_weight"`/`"sparse_weight"` keys.
- Consumes (unchanged): `hybrid_search` from `common.milvus_client`.

- [ ] **Step 1: Write the failing `rrf_merge` weighting tests**

Add to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
def test_rrf_merge_default_weights_match_prior_unweighted_behavior():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_upweights_dense_list_over_sparse():
    # "a" is dense-rank-1-only; "c" is sparse-rank-1-only. Equal weight would
    # tie them (both contribute 1/(60+1)); upweighting dense must break the
    # tie in "a"'s favor.
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=1.5, sparse_weight=0.5)

    assert merged[0]["chunk_id"] == "a"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


def test_rrf_merge_upweights_sparse_list_over_dense():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=0.5, sparse_weight=1.5)

    assert merged[0]["chunk_id"] == "c"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: the two new weighting tests FAIL with `TypeError: rrf_merge() got an unexpected keyword argument 'dense_weight'`. The first new test (default-weights) should already PASS since it doesn't use the new params — it's there to pin today's behavior before the signature changes.

- [ ] **Step 3: Implement weighted `rrf_merge`**

In `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`, replace `rrf_merge`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: all `rrf_merge` tests PASS, including the two pre-existing ones
(`test_rrf_merge_combines_and_ranks_by_reciprocal_rank`,
`test_rrf_merge_dedupes_by_chunk_id`) — they don't pass weight kwargs, so the
new defaults must keep them green with no edits.

- [ ] **Step 5: Write the failing `retrieve()` intent-weighting tests**

Add to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_resolves_conceptual_intent_to_dense_weighted_rrf(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await module.retrieve(
        gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, intent="conceptual",
    )

    # conceptual -> dense_weight=1.5, sparse_weight=0.5: the dense-only chunk
    # must outrank the sparse-only chunk despite both being rank-1 in their list.
    assert result[0]["chunk_id"] == "a"


@pytest.mark.asyncio
async def test_retrieve_resolves_citation_lookup_intent_to_sparse_weighted_rrf(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "c", "doc_id": "d2", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await module.retrieve(
        gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, intent="citation_lookup",
    )

    assert result[0]["chunk_id"] == "c"


@pytest.mark.asyncio
async def test_retrieve_defaults_to_neutral_weighting_for_unrecognized_intent(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "sparse", "score": 5.0}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    # Neither "unknown" (a real intent label) nor a totally unrecognized
    # string should raise or behave differently from each other - both must
    # resolve to neutral (1.0, 1.0) weighting.
    result_unknown = await module.retrieve(
        gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, intent="unknown",
    )
    result_unrecognized = await module.retrieve(
        gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None, intent="not_a_real_label",
    )

    assert result_unknown[0]["rrf_score"] == result_unrecognized[0]["rrf_score"]


@pytest.mark.asyncio
async def test_retrieve_defaults_intent_param_to_unknown_when_omitted(monkeypatch):
    """Backward-compatibility: existing callers that don't pass `intent` at
    all (e.g. the eval harness, if any direct caller exists) must keep
    getting today's neutral weighting."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    result = await module.retrieve(gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None)

    assert result[0]["chunk_id"] == "a"


@pytest.mark.asyncio
async def test_retrieve_includes_resolved_weights_in_rrf_merge_trace_step(monkeypatch):
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
        gateway, milvus_client=object(), rewritten_query="q", doc_id_allowlist=None,
        intent="provision_lookup", on_step=on_step,
    )

    rrf_step = next(data for step, data in steps if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 0.5
    assert rrf_step["sparse_weight"] == 1.5
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: FAIL — `retrieve()` doesn't accept an `intent` keyword yet
(`TypeError: retrieve() got an unexpected keyword argument 'intent'`), and
`test_retrieve_includes_resolved_weights_in_rrf_merge_trace_step` also fails
on the missing kwarg before ever reaching its assertion.

- [ ] **Step 7: Implement `retrieve()`'s intent-weight lookup**

In `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`, add the
lookup table near the top of the file (after the imports) and update
`retrieve()`:

```python
_INTENT_RRF_WEIGHTS: dict[str, tuple[float, float]] = {
    "citation_lookup": (0.5, 1.5),
    "provision_lookup": (0.5, 1.5),
    "conceptual": (1.5, 0.5),
    "unknown": (1.0, 1.0),
}


async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    rewritten_query: str,
    doc_id_allowlist: list[str] | None,
    intent: str = "unknown",
    on_step: OnStep | None = None,
) -> list[dict]:
    dense_weight, sparse_weight = _INTENT_RRF_WEIGHTS.get(intent, (1.0, 1.0))

    dense_vector = await gateway.embed(role="query_embed", text=rewritten_query)

    dense_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=dense_vector,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_dense", collection_trace(dense_by_collection))

    sparse_by_collection = await hybrid_search(
        milvus_client, collections=MILVUS_COLLECTIONS, dense_vector=None,
        sparse_query_text=rewritten_query, doc_id_allowlist=doc_id_allowlist, limit=50,
    )
    if on_step is not None:
        await on_step("milvus_sparse", collection_trace(sparse_by_collection))

    merged = rrf_merge(
        _flatten(dense_by_collection), _flatten(sparse_by_collection),
        dense_weight=dense_weight, sparse_weight=sparse_weight,
    )

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
            "dense_weight": dense_weight, "sparse_weight": sparse_weight,
        })

    return merged
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: all tests PASS, including the two pre-existing tests that were in
the file before this task
(`test_retrieve_embeds_rewritten_query_and_merges_dense_sparse`,
`test_retrieve_emits_dense_sparse_and_rrf_merge_steps`) — neither passes
`intent`, so the default must keep them green with no edits to those two.

- [ ] **Step 9: Wire `intent` through `pipeline.py`, and update its test stubs**

In `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`, change the
`retrieve` call inside `run_ai_mode`:

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

In `packages/retrieval-api/tests/test_ai_mode_pipeline.py`, every `fake_retrieve`
stub currently has the signature
`async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, on_step=None):`
— `pipeline.py` now calls `retrieve` with 5 positional arguments (the new
`intent_result["intent"]` inserted before the `on_step` keyword), so every
stub with that old signature will raise `TypeError: fake_retrieve() takes
from 4 to 5 positional arguments but 6 were given` the moment its test runs
against the modified pipeline. Update each of the 3 occurrences (in
`test_run_ai_mode_success_path`, `test_run_ai_mode_succeeds_with_party_only_filter`,
and `test_run_ai_mode_forwards_on_step_to_every_stage`) to accept the new
parameter:

```python
    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, intent, on_step=None):
```

For `test_run_ai_mode_success_path` specifically, also assert the value is
forwarded correctly — change its `fake_extract_intent` to return a real
enum value and capture what `fake_retrieve` receives:

```python
    async def fake_extract_intent(gateway, query, on_step=None):
        return {"rewritten_query": "rewritten", "intent": "conceptual", "filters": {}}

    received_intent = {}

    async def fake_retrieve(gateway, milvus_client, rewritten_query, doc_id_allowlist, intent, on_step=None):
        received_intent["value"] = intent
        return [{"chunk_id": "a", "doc_id": "d1", "text": "t", "rrf_score": 0.9}]
```

and after the existing `result == {...}` assertion, add:

```python
    assert received_intent["value"] == "conceptual"
```

Finally, in `test_run_ai_mode_emits_all_seven_trace_steps_in_order_end_to_end`
(the real, unmocked-`retrieve` integration test at the bottom of the file),
add an assertion on the `rrf_merge` trace step's new weight fields — this
test's `fake_chat` already returns `"intent": "conceptual"`, so the resolved
weights must be `(1.5, 0.5)`:

```python
    rrf_step = next(data for step, data in collected if step == "rrf_merge")
    assert rrf_step["dense_weight"] == 1.5
    assert rrf_step["sparse_weight"] == 0.5
```

- [ ] **Step 10: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests pass (existing count + the new `rrf_merge`/`retrieve`
weighting tests from Steps 1 and 5, plus the updated pipeline assertions).

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py \
        packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py \
        packages/retrieval-api/tests/test_ai_mode_retrieve.py \
        packages/retrieval-api/tests/test_ai_mode_pipeline.py
git commit -m "feat: add intent-driven RRF dense/sparse weighting"
```

---

### Task 2: Retrieval-rank regression check

**Files:** none modified — this is a validation task, matching Phase 1's
Task 5 pattern (evaluate, decide, report; no code changes unless a
regression is found and needs fixing).

**Interfaces:**
- Consumes: `retrieval_eval.py --sample12` (existing harness, unmodified by
  this plan).

This task changes retrieval ranking directly (unlike Phase 1's
extraction-only work), so — per the design doc's Validation section — it
needs a real regression check against the existing eval harness, not just
unit tests.

- [ ] **Step 1: Snapshot the pre-change baseline, if not already on record**

Check whether a `dev`-branch baseline run already exists in
`.eval-results/latest.json` or `docs/small-model-eval-results.md` using
today's default `slm`/`reranker`/`synthesis` models (post the
`gemma-4-26B-A4B-it` swap). If not, run one before merging Task 1's commit,
so the comparison isn't confounded by an unrelated model change:

```bash
git stash  # if Task 1's commit isn't yet on this branch tip, skip this
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8011 --sample12 --skip-agentic --no-langfuse \
  --run-name pre-rrf-weighting-baseline
```

(Start a local gateway first the same way Phase 1's Task 5 did: `uv run
uvicorn model_gateway.main:app --port 8011 --app-dir packages/model-gateway/src`
from the repo root — note the `--app-dir` flag, required because
`GatewaySettings` loads `.env` relative to CWD.)

- [ ] **Step 2: Run the eval with Task 1's change applied**

```bash
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8011 --sample12 --skip-agentic --no-langfuse \
  --run-name intent-rrf-weighting
```

- [ ] **Step 3: Compare per-stage pass counts**

Compare the `rrf`/`reranker`/`citation_valid`/`gold_cited` columns between
the baseline and this run, per query class (direct/indirect/adversarial —
see `docs/retrieval-eval-queries.md`'s pass criteria). The
`citation_lookup`/`provision_lookup`-favored queries are the *direct*-class
queries in the eval set (they're the ones anchored on citations/section
numbers) — a flat-or-negative result specifically on direct-class queries is
the strongest signal something in the weighting is backwards, and should be
investigated (e.g. by checking whether `intent.py` is actually classifying
those queries as `citation_lookup`/`provision_lookup` rather than
`conceptual`/`unknown` — a classification miss would silently make this
whole feature inert for that query rather than actively harmful, which is a
different failure mode worth distinguishing from an actively wrong weight
direction).

- [ ] **Step 4: Decide**

If no stage shows a regression versus baseline: the change is validated,
nothing further to do (Task 1's commit stands as merged).

If a regression is found: do not fix it inside this validation task. Report
the specific regressed stage/query IDs and stop — decide with your human
partner whether to adjust the weight magnitude (design doc's `1.5`/`0.5`
choice was not eval-derived, just a reasonable starting point) or revert
Task 1 pending further investigation.

- [ ] **Step 5: Record the result**

Add a short section to `docs/small-model-eval-results.md` (or a new doc if
this eval set outgrows that file's scope) recording the before/after
comparison and the decision, following the same evidentiary style as the
existing "Later findings" section (exact commands run, exact pass counts,
explicit verdict).

---

## Self-review notes

- **Spec coverage:** Design's item 1 (weighted `rrf_merge`) → Task 1 Steps 1-4.
  Item 2 (intent→weight lookup) → Task 1 Steps 5-8. Item 3 (pipeline wiring)
  → Task 1 Step 9. Item 4 (trace visibility) → Task 1 Steps 5/7/9 (the new
  `dense_weight`/`sparse_weight` trace fields and their tests). Validation
  section → Task 2 in full.
- **Type consistency:** `retrieve()`'s new `intent: str = "unknown"`
  parameter name and position (5th, before `on_step`) is identical between
  the design doc, Task 1's implementation, and Task 1's pipeline-wiring call
  site — no renaming drift.
- **No placeholders:** every step includes literal code; Task 2 is
  intentionally a validation-only task (no code), matching Phase 1's Task 5
  precedent, and its "if a regression is found" branch names concrete next
  actions (adjust magnitude or revert) rather than a vague TODO.
