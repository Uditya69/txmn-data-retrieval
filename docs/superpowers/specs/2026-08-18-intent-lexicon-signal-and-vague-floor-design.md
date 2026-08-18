# Intent classification: lexicon signal + too-vague-to-tag floor

Date: 2026-08-18
Status: proposed

## Problem

`extract_intent` (`ai_mode/intent.py`) over-guesses category tags on genuinely vague
queries. Confirmed with a new eval (`collection_routing_eval.py`,
`evals/collection_routing_cases.json`) run against the real model: of 8 deliberately
vague cases, 3 came back confidently (and wrongly) tagged instead of empty:

| Query | Expected | Actual |
|---|---|---|
| `"gift from father taxable?"` | `[]` | `["caselaws"]` |
| `"capital gains"` | `[]` | `["commentary"]` |
| `"help with income tax"` | `[]` | `["commentary"]` |

Root cause: `_LLAMA_SYSTEM_PROMPT`'s category definitions are phrasing-shape rules, not
anchor-requiring rules. `commentary`'s "default landing spot for 'explain X' / 'how does
X work' queries" and `caselaws`'s "what was decided for a dispute/fact pattern" signal
both fire on how a query is *shaped*, not on whether it names anything concrete (an Act,
section, party, court, or citation). A bare topic word or a personal fact-pattern
question can match that shape with zero real legal anchor present. The prompt's only
abstain instruction — `"Output an empty list when no category confidently applies"` —
gives the model no defined threshold for "confidently," and an 8B model is a weaker,
noisier calibrator on short/ambiguous input than a larger one, with no exposed
chain-of-thought to second-guess itself before committing to JSON in one pass.

This matters because `collections_for_intent()` (`common/schemas.py`) routes narrowly on
a non-empty `intent` — a wrong tag silently searches the wrong 1-8 collections instead of
all 11. Empty `intent` is always safe (search-all fallback); a wrong non-empty tag is the
only real failure mode this pipeline has for category routing.

## Goals

- Give the SLM an explicit, code-computed signal for "this query has no recognizable
  legal anchor at all" — currently invisible to it.
- Add a deterministic floor that forces `intent = []` for queries too short and too
  anchor-free to trust any classification, regardless of what the SLM tagged — the first
  code-level override in this classification path, justified by the asymmetry above
  (a false-positive here costs precision via a broader search, never a wrong one).
- Reuse the existing legal-lexicon mechanism (`common/query_tokenizer.py`'s
  `classify_query_shape`/`expand_query_synonyms`, already powering `/v1/query-analysis`)
  rather than building a second detector.

## Non-goals

- Not fixing the deeper root cause (the loose `commentary`/`caselaws` category
  definitions themselves) — tracked as a separate, later prompt-tuning task. This spec's
  two fixes catch the two shapes of the problem this design addresses (compound-empty
  lexicon signal, short+anchor-free queries); a longer vague query with no anchor that
  isn't caught by either still relies on the definitions' own (unfixed) looseness.
- Not touching `_sanitize_filters`/`_validate_categories`'s existing behavior — this adds
  a new check alongside them, doesn't change what they do.
- Not adding a hard override anywhere else in the classification path — this is a
  narrowly scoped exception (see "Why a hard rule is safe here" below), not a precedent
  for general SLM-output overriding.
- No change to `collections_for_intent()` itself — its existing empty-intent-searches-all
  behavior is exactly what these two fixes lean on.

## Design

### 1. Shared anchor detector

New helper in `ai_mode/intent.py`, built from three signals already computed or
available (`_build_chunk_context`'s existing `chunk_query` output, plus
`common.query_tokenizer`'s `classify_query_shape`/`expand_query_synonyms` — not
currently imported into this file, newly wired in):

```python
def _has_legal_anchor(query: str, chunk_context: str | None) -> bool:
    if chunk_context is not None:
        return True  # a structural span (date/section/court/etc.) was already found
    if expand_query_synonyms(query) != query:
        return True  # a legal-lexicon term/abbreviation was recognized
    if classify_query_shape(query) != "plain":
        return True  # provision/citation shape implies an anchor
    return False
```

Single detector, two consumers below — avoids duplicating the anchor logic.

### 2. Lexicon signal — soft prompt hint

When `_has_legal_anchor` is `False`, append a note to `extract_intent`'s user message
(same pattern as the existing `chunk_context` injection):

```
Lexicon check: no known legal term, Act/section reference, citation, or party pattern
was recognized anywhere in this query.
```

`_LLAMA_SYSTEM_PROMPT` gains one sentence near its existing "Output an empty list when no
category confidently applies" instruction: a "Lexicon check" note is strong evidence to
abstain unless the query's actual wording — not just its general subject — names
something concrete.

This is a hint, not a floor — the model can still tag despite it. Applies to queries of
any length; catches the compound-empty case (no structural span, no lexicon match, plain
shape) regardless of word count.

### 3. Too-vague-to-tag — hard floor

New check in `_validate_result` (alongside `_validate_categories`/`_sanitize_filters`,
same guardrail tier — reject content the SLM shouldn't be trusted on, never add):

```python
def _too_vague_to_tag(query: str, chunk_context: str | None) -> bool:
    return len(query.split()) <= 5 and not _has_legal_anchor(query, chunk_context)
```

When `True`, `_validate_result` forces `result["intent"] = []` regardless of what the SLM
returned — overriding, not merely filtering, the model's own tag. This is deliberately
narrower than the soft hint: only fires for short queries (`<= 5` words) with zero anchor
signal, not any vague-shaped query of any length.

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

**Why a hard rule is safe here specifically:** `collections_for_intent()` treats empty
intent as "search all 11 collections" — never an exclusion. A false-positive (a genuinely
answerable 5-word-or-fewer query that happens to use no recognized legal term) degrades
to a broader search, not a wrong one. A false-negative under the *old* soft-hint-only
behavior (a vague query keeps its wrongly-guessed tag) silently searches the wrong
collections instead. That asymmetry — force-empty's worst case is strictly milder than
the status quo's worst case — is what makes a hard override defensible here, unlike a
general override on SLM category judgment elsewhere in this pipeline.

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
  _validate_result(query, raw_result)
       +-- _too_vague_to_tag(query, chunk_context)?          # NEW hard floor (§3)
             True  --> intent forced to []
             False --> intent = _validate_categories(...) as today
```

## Testing

- `_has_legal_anchor`: unit tests for all 4 branches (chunk_context present; synonym
  match; non-plain shape; none of the above → `False`).
- `_build_lexicon_signal`-equivalent wiring: `extract_intent` test confirming the
  "Lexicon check" note appears in the user message for a vague query (`chunk_context is
  None`, no synonym match, plain shape) and is absent for a query with a recognized
  anchor.
- `_too_vague_to_tag`: unit tests — `<=5` words + no anchor → `True`; `<=5` words + anchor
  present → `False`; `>5` words + no anchor → `False` (word-count floor only, not a
  general vague-query catch); boundary at exactly 5 and 6 words.
- `_validate_result`: test that a raw SLM result with a non-empty `intent` gets forced to
  `[]` when `_too_vague_to_tag` is `True`, and passes through `_validate_categories`
  unchanged when `False`.
- Eval re-run: `collection_routing_eval.py` against the real model after implementation —
  confirm R13 (`"gift from father taxable?"`, 4 words), R14 (`"capital gains"`, 2 words),
  R18 (`"help with income tax"`, 4 words) all flip from `wrong` to `safe-empty` (all are
  `<=5` words with no anchor, so the hard floor should catch all three directly — the
  soft hint alone wasn't guaranteed to). Confirm no regression on the 10 confident cases —
  all are well over 5 words (7-11 each in the current dataset), so `_too_vague_to_tag`'s
  word-count gate alone guarantees it never fires on them, independent of whether each one
  also has a detectable anchor (not individually verified for every case — e.g. R06,
  `"expert opinion article on the recent controversy around faceless assessment"`, has no
  digit after "article" so `SECTION_PATTERN` doesn't match it, and it's not confirmed
  whether any token hits a lexicon synonym either — its safety here rests on word count,
  not on a verified anchor).
- Dataset gap worth closing during implementation: `evals/collection_routing_cases.json`
  has no *short* (`<=5` words) confident case — e.g. `"section 54F exemption"` (3 words,
  has an anchor) — so the live eval never exercises the one case where the hard floor's
  own condition (`<=5` words) and a real anchor overlap. The unit tests cover this
  combination directly; the dataset doesn't. Add one such case alongside the fix.

## Open questions / risks

- The `5`-word threshold is a starting value matched to the three known failure cases
  (2, 4, and 4 words), not derived from a larger sample — same caveat as the persona
  system's `query_count >= 20` threshold. Worth revisiting once more real query data is
  observed, ideally via the eval script's dataset growing over time.
- This does not fix longer vague queries with no anchor (e.g. a rambling 8-word question
  naming no Act/section/party) — those still rely entirely on the soft hint and the
  unfixed category-definition looseness. Tracked as a separate follow-up (tightening
  `commentary`/`caselaws`'s definitions to require an anchor), not part of this spec.
