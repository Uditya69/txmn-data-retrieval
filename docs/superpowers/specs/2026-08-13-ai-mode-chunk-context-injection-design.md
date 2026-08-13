# AI Mode intent extraction — inject chunk_query structural context — design

## Goal

Improve `extract_intent`'s (AI Mode's SLM classification step) intent/filter accuracy by
giving it pre-parsed structural signal about the query, instead of making it re-derive
entity boundaries (citation spans, section numbers, court names) from raw text on its own.
This matters more as we move toward smaller `slm`-role candidates (see
`docs/superpowers/specs/2026-08-10-small-model-eval-harness-design.md`): a smaller model is
less reliable at inferring implicit structure, so making structure explicit in the prompt
should close some of that gap without adding model capacity.

## Current state

`extract_intent` (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py:193-219`) sends
the SLM a system prompt (`_LLAMA_SYSTEM_PROMPT`, intent.py:35-80, ~650-750 tokens) already
concatenated with `build_schema_context()` (`common/schema_context.py`, ~250-300 tokens) —
this tells the model what filter *fields and known values* exist (courts, acts, etc.). The
user message is just the raw query string (intent.py:201). No per-query structural parsing
is passed in.

Separately, `common/query_tokenizer.py::chunk_query(query) -> list[dict]` (query_tokenizer.py:252)
already runs deterministically, in-process, with no network call, inside Instant mode's ES
query building (`common/es_client.py`) — it segments a query into typed spans:

```python
[{"text": "1995 taxmann.com 569", "type": "citation", "proximity": 2, "alt_text": None},
 {"text": "Delhi High Court",     "type": "court_city", "proximity": 0, "alt_text": None},
 {"text": "Rule 57A",             "type": "section",    "proximity": 0, "alt_text": "057A"},
 ...]
```

`type` is one of `text | section | citation | court_city | quoted`. `proximity` and
`alt_text` are ES `match_phrase`-slop concerns with no bearing on intent classification.
This function is not currently called anywhere in AI Mode.

Note: `analyze_query` (query_tokenizer.py:155) is a different, more verbose function used
only for Instant mode's diagnostic trace step (`instant/search.py:113`, explicitly
"diagnostic-only, not on the search path"). It is not part of this change — `chunk_query`'s
output is the leaner, prompt-appropriate shape (~500 chars vs. ~1400 chars of JSON for a
14-word query).

## Change

In `extract_intent`, call `chunk_query(query)` and fold a trimmed projection of its output
into the SLM call as additional per-query context, alongside the existing raw query and
schema context.

**Projection:** drop `proximity` and `alt_text` (ES-only, not intent signal). Keep only
`text` and `type` per chunk, and drop any chunk with `type == "text"` (a bare word run
carries no structural signal beyond what the model already sees in the raw query — passing
it back adds tokens without adding information). Net effect: only citation/section/
court_city/quoted spans are surfaced, e.g.:

```json
[{"text": "1995 taxmann.com 569", "type": "citation"},
 {"text": "Delhi High Court", "type": "court_city"}]
```

If `chunk_query` returns no non-`text` chunks (a purely conceptual query with no
recognizable entities), omit the block entirely rather than sending an empty list — an
empty structural-context block is a wasted ~15-20 tokens per call at scale.

**Placement:** append as a labeled block after the raw query in the user message (not the
system prompt) — this is per-query data, and `_LLAMA_SYSTEM_PROMPT` is a module-level
constant shared across all calls. Exact wording/labeling is an implementation-time detail;
the labeling must make clear this is a structural hint about the query, not something the
model is supposed to add to `rewritten_query` verbatim (the existing rewrite rules in
intent.py:41-49 already forbid inventing content — the new block must not read as an
instruction to include this text if it doesn't already appear in the query, which it always
will since it's extracted from that same query).

**Cost:** for a typical multi-clause query with 1-3 recognized entity spans, this adds
roughly 30-80 tokens per call — small relative to the existing ~900-1050 token system prompt,
proportionate to the stated "meaningful signal, not context bloat" goal.

**No change to:**
- `chunk_query`, `analyze_query`, or any other `query_tokenizer.py` function.
- Instant mode (`instant/search.py`) — it has no SLM call today and this change does not add
  one; it stays a zero-LLM preview path.
- `_sanitize_filters`, `_safe_rewrite`, `_validate_result`, or any other post-SLM validation
  logic in intent.py — the SLM's output is still independently sanitized against the raw
  query exactly as today. The new context only changes what the model sees going in.

## Non-goals

- **No cross-validation / auto-correction.** This change does not add code that checks or
  forces SLM output based on `chunk_query`'s tags (e.g. auto-setting `intent="citation_lookup"`
  because a `citation`-type chunk exists, or auto-populating a `court` filter from a
  `court_city` chunk). That is a materially different, larger feature — more surface, more
  failure modes, and it couldn't be isolated in the same eval pass as this change. If the
  eval below shows the SLM still misses cases where `chunk_query` had already correctly
  identified the entity, that's the trigger for scoping cross-validation as its own follow-up
  spec, not something to fold in here.
- **No change to `_LLAMA_SYSTEM_PROMPT` content itself** beyond adding the new per-query
  block at call time — the existing instructions (rewrite rules, filter rules, forbidden
  rewrites) are untouched.
- **No new model-shape branching.** `_system_prompt_for_model` (intent.py:83-99) is untouched;
  the new context block is appended the same way regardless of which model handles `slm`.

## Validation plan (not executed as part of this spec)

Use the eval infrastructure already built for the small-model comparison round
(`docs/superpowers/specs/2026-08-10-small-model-eval-harness-design.md`):

1. Baseline run: current prompt, current `slm` model (Meta-Llama-3.1-8B-Instruct), no
   chunk-context injection.
2. Candidate run A: chunk-context injection added, same model — isolates whether the change
   itself is a net improvement or regression on intent/filter accuracy, independent of any
   model swap.
3. Candidate run B: chunk-context injection added, candidate smaller model
   (Qwen3-4B-Instruct-2507) — tests the actual motivating hypothesis, that structural context
   narrows the accuracy gap for a smaller model.
4. Compare all three via `compare_eval_runs.py` against the existing query sample, on
   intent-classification accuracy and filter-precision specifically (not just downstream
   retrieval recall, since this change targets the classification step directly).

A regression in run A (structural context makes the current model *worse*, e.g. because it
starts echoing chunk text into `rewritten_query` unnaturally) is grounds to drop this change
without proceeding to run B.

## Out of scope

- Any change to Instant mode.
- ES `_analyze` API usage (the deterministic `chunk_query` path was chosen specifically to
  avoid the added network round-trip and live-index dependency this would introduce before
  the SLM call).
- Cross-validation/auto-correction logic (see Non-goals).
- Running the validation eval itself — this spec documents the plan; execution is a
  follow-up implementation task.
