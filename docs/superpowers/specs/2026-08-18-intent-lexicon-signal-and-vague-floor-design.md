# Intent classification: lexicon signal + too-vague-to-tag floor

Date: 2026-08-18
Status: proposed

## Problem

`extract_intent` (`ai_mode/intent.py`) over-guesses category tags on genuinely vague
queries. Confirmed with a new eval (`collection_routing_eval.py`,
`evals/collection_routing_cases.json`) run against the real model: of 8 deliberately
vague cases, 3 came back confidently (and wrongly) tagged instead of empty, e.g.
`"capital gains"` → `["commentary"]` and `"help with income tax"` → `["commentary"]` —
bare topic words and generic requests with no actual legal content to anchor on.

Root cause: the classification prompt's category definitions include phrasing-shape
rules (a "default landing spot" for explain-style queries, a fact-pattern/scenario
signal for caselaws-shaped questions) that can fire on how a query is *shaped* even when
it names nothing concrete. The prompt's only abstain instruction —
`"Output an empty list when no category confidently applies"` — gives the model no
defined threshold for "confidently," and an 8B model is a weaker, noisier calibrator on
short/ambiguous input than a larger one, with no exposed chain-of-thought to
second-guess itself before committing to JSON in one pass.

This matters because `collections_for_intent()` (`common/schemas.py`) routes narrowly on
a non-empty `intent` — a wrong tag silently searches the wrong 1-8 collections instead of
all 11. Empty `intent` is always safe (search-all fallback); a wrong non-empty tag is the
only real failure mode this pipeline has for category routing.

## Goals

- Give the SLM an explicit, code-computed signal for "this query has no recognizable
  legal anchor at all" — currently invisible to it.
- Add a deterministic floor: when the lexical pipeline (structural chunking + legal
  lexicon + shape classification) finds nothing, force `intent = []` regardless of what
  the SLM tagged. Searching all 11 collections is accepted as strictly preferable to any
  risk of a wrong-collection search, full stop — see "Explicit ruling" below for what
  this costs.
- Reuse the existing legal-lexicon mechanism (`common/query_tokenizer.py`'s
  `classify_query_shape`/`expand_query_synonyms`, already powering `/v1/query-analysis`)
  rather than building a second detector.

## Non-goals

- Not fixing the deeper root cause (the category definitions' own phrasing-shape
  triggers) — tracked as a separate, later prompt-tuning task.
- Not touching `_sanitize_filters`/`_validate_categories`'s existing behavior — this adds
  a new check alongside them, doesn't change what they do.
- Not adding a hard override anywhere else in the classification path — this is a
  narrowly scoped exception, not a precedent for general SLM-output overriding.
- No change to `collections_for_intent()` itself — its existing empty-intent-searches-all
  behavior is exactly what this leans on.
- No word-count or query-shape (e.g. "ends in `?`") exemption — see "Explicit ruling."

## Explicit ruling — this deliberately reverts part of an already-shipped fix

`extract_intent`'s `caselaws` category already recognizes fact-pattern/scenario
questions (e.g. `"is X taxable when Y"`) as a valid signal **even with zero literal
anchor** — no Act, section, party, citation, or court name required. That fix exists
specifically because keyword/lexicon matching cannot recognize a scenario; only reading
the sentence's meaning can.

The floor this spec adds triggers on exactly that same condition — no lexical anchor
found — and forces `intent = []` unconditionally. There is no way to keep both: "no
anchor → tag it anyway (fact-pattern)" and "no anchor → force empty" are the same
trigger with opposite actions, and lexical/keyword matching has no way to distinguish
"a real fact-pattern scenario" from "a bare vague topic" — both look identical to a
word-bank lookup (neither contains a recognized term). Only semantic reading (i.e. the
SLM itself) can tell them apart, and this floor runs after the SLM's output and
overrides it regardless.

**Decision, made explicitly and knowingly:** every fact-pattern question with no literal
anchor will be forced back to `intent = []` by this floor, discarding the fact-pattern
signal's benefit for that entire class of query. Traded away because a guaranteed-safe
(if less precise) search-all outcome was judged strictly preferable to any residual risk
of a wrong-collection search — including on the subset of anchor-less queries that would
otherwise have been tagged correctly. This is a real, accepted cost of this design, not
an oversight — recorded here so it isn't rediscovered as a "regression" later.

## Design

### 1. Shared anchor detector

New helper in `ai_mode/intent.py`, built from three signals already computed or
available (`_build_chunk_context`'s existing `chunk_query` output, plus
`common.query_tokenizer`'s `classify_query_shape`/`expand_query_synonyms` — not
currently imported into this file, newly wired in):

```python
def _has_legal_anchor(query: str, chunk_context: str | None) -> bool:
    if chunk_context is not None:
        return True  # a structural span (citation/section/court/date/party) was found
    if expand_query_synonyms(query) != query:
        return True  # a legal-lexicon term/abbreviation was recognized
    if classify_query_shape(query) != "plain":
        return True  # provision/citation shape implies an anchor
    return False
```

Single detector, two consumers below — avoids duplicating the anchor logic.

### 2. Lexicon signal — soft prompt hint

When `_has_legal_anchor` is `False`, append a note to `extract_intent`'s user message,
inline after `chunk_context` and `has_anchor` are both computed (no new function needed):

```python
if not has_anchor:
    user_message += (
        "\n\nLexicon check: no known legal term, Act/section reference, citation, or "
        "party pattern was recognized anywhere in this query."
    )
```

`_LLAMA_SYSTEM_PROMPT` gains one sentence near its existing "Output an empty list when no
category confidently applies" instruction: a "Lexicon check" note is strong evidence to
abstain. §3's hard floor overrides the SLM's `intent` output regardless of what this hint
achieves — but the hint is not inert: the prompt itself instructs `search_query`'s
phrasing to follow whichever intent the model believes applies ("if 'commentary' alone is
tagged, keep plain-language phrasing", etc.), so nudging the model toward abstaining can
still shape `search_query`'s rewrite even though the final `intent` field is decided by
§3 either way. Also kept as defense in depth for the SLM call itself, and because §3
could in principle be relaxed later without touching this.

### 3. Too-vague-to-tag — hard floor

New check in `_validate_result` (alongside `_validate_categories`/`_sanitize_filters`,
same guardrail tier — reject content the SLM shouldn't be trusted on, never add):

```python
def _too_vague_to_tag(query: str, chunk_context: str | None) -> bool:
    return not _has_legal_anchor(query, chunk_context)
```

No word count, no phrasing-shape exemption — see "Explicit ruling" above for why.

**Signature change required.** `_validate_result` is currently `_validate_result(query:
str, result) -> dict` — it has no access to `chunk_context` today (that's a local
variable inside `extract_intent`, computed before `_validate_result` is called). This
becomes `_validate_result(query: str, result, chunk_context: str | None) -> dict`, and
its `intent` line changes from unconditionally calling `_validate_categories(...)` to:

```python
"intent": [] if _too_vague_to_tag(query, chunk_context) else _validate_categories(result.get("intent")),
```

The one call site inside `extract_intent` (`result = _validate_result(query, result)`)
updates to pass `chunk_context` through. `_fallback_intent`'s path is unaffected — it
already returns `intent: []` unconditionally when the SLM's response is unparseable, so
`_too_vague_to_tag` has nothing to add there.

### 4. Filters

No change needed. `_sanitize_filters` already requires every filter value to be a literal
substring of the query — a vague, anchor-free query already can't produce a filter that
survives that check, so `_too_vague_to_tag` firing has nothing additional to clean up on
the filters side.

## Data flow

```
extract_intent(query)
  chunk_context = _build_chunk_context(query)              # existing
  has_anchor = _has_legal_anchor(query, chunk_context)      # NEW, shared
       |
       +-- False --> user_message += "Lexicon check: ..."  # NEW soft hint (§2)
       |
  [SLM call happens regardless]
       |
  _validate_result(query, raw_result, chunk_context)
       +-- not has_anchor?                                  # NEW hard floor (§3)
             True  --> intent forced to []
             False --> intent = _validate_categories(...) as today
```

## Testing

- `_has_legal_anchor`: unit tests for all 4 branches (chunk_context present; synonym
  match; non-plain shape; none of the above → `False`).
- Lexicon-signal wiring: `extract_intent` test confirming the "Lexicon check" note
  appears in the user message for a vague query (`chunk_context is None`, no synonym
  match, plain shape) and is absent for a query with a recognized anchor.
- `_too_vague_to_tag`: unit tests — no anchor → `True` regardless of length; anchor
  present → `False` regardless of length; explicit test that a fact-pattern-shaped query
  with no anchor (e.g. `"is X taxable when Y"`) still returns `True` — documenting the
  accepted tradeoff as a test, not just prose, so a future change to this function has to
  consciously break the test to reintroduce fact-pattern tagging.
- `_validate_result`: test that a raw SLM result with a non-empty `intent` gets forced to
  `[]` when `_too_vague_to_tag` is `True`, and passes through `_validate_categories`
  unchanged when `False`.
- Eval re-run: `collection_routing_eval.py` against the real model after implementation —
  confirm `"capital gains"` and `"help with income tax"` flip from `wrong` to
  `safe-empty`. The dataset's third vague case, `"gift from father taxable?"` (R13),
  already has `expected_categories: []` (no dataset edit needed) — it was previously
  `wrong` only because the model tagged it `["caselaws"]` and nothing forced that back to
  empty; post-fix it should also read `safe-empty`, confirming the ruling above is
  actually in effect, not just documented. Confirm no regression in pass/fail status on the 10 confident cases — not the same as
  "no change at all": anchor detection was not individually verified for every case (e.g.
  R06, `"expert opinion article on the recent controversy around faceless assessment"`,
  has no digit after "article" so `SECTION_PATTERN` misses it, and its lexicon-synonym
  coverage is unconfirmed). Word count no longer provides a safety net here (§3 dropped
  it), so if a confident case genuinely lacks an anchor, this floor will force it to `[]`
  — that still counts as `PASS` under `collection_routing_eval.py`'s own rule (empty is
  always safe), just in the `safe-empty` bucket instead of `exact`. A case moving from
  `exact` to `safe-empty` is an expected, accepted outcome of this design, not a defect;
  a case moving to `wrong` would be a real regression and the only thing this check
  should treat as a failure.

## Open questions / risks

- This does not fix vague queries that *do* happen to contain some recognized lexicon
  term but are still fundamentally unfocused (e.g. a rambling multi-topic question that
  happens to mention "tax") — `_has_legal_anchor` would return `True` for those, the
  floor never fires, and they remain fully dependent on the SLM's own (unfixed, per
  Non-goals) judgment. Not addressed here.
- The accepted tradeoff (fact-pattern questions with no anchor lose their signal) has no
  measured cost yet — how often real users phrase a genuine case-law question with zero
  legal terms is unknown. Worth watching via the eval dataset growing with real traffic
  patterns over time, and revisiting this ruling if that cost turns out to be larger than
  expected.
