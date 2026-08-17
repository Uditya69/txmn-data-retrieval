# Persona context: trust gating + relevance judgment

Date: 2026-08-17
Status: proposed

## Problem

Persona (`packages/persona`) currently feeds AI Mode's `synthesize` step
(`ai_mode/synthesize.py`) with a rendered context string
(`persona.prompt.render_persona_context`), but not `extract_intent`
(`ai_mode/intent.py`) — the classification step that decides `intent`
category tags and `filters`. Two related gaps, both stemming from the same
root cause (no sample-size signal reaches consumers of persona data):

1. **New-user noise.** `merge_category_affinity` is a running average from
   query 1 — a single query can swing a category's affinity to 1.0 (100%).
   `expertise_level`/`query_style` are worse: `merge_expertise_patch` is a
   last-write-wins overwrite, so one SLM guess on query 1 permanently
   replaces any prior label until contradicted. `render_persona_context`
   has no confidence/sample-size check before rendering these as flat,
   unhedged statements ("This user frequently asks about X; expertise
   level: Y").
2. **No relevance check.** Even with enough history to be statistically
   sound, a persona built from past queries can be irrelevant to the
   *current* query — a caselaws-heavy user asking a brand-new-topic
   question (e.g. a tariff/HSN lookup) still gets nudged toward caselaws
   framing today. Better no persona context than wrong persona context.

## Goals

- Gate persona *surfacing* (not recording) on sample size, so a thin
  persona doesn't get spoken about with false confidence.
- Make `expertise_level`/`query_style` build up like `category_affinity`
  already does, instead of flipping on one guess.
- Let the SLM itself judge, per-query, whether the persona hint fits the
  query at hand — for both direct and indirect mismatches — rather than
  relying only on a global sample-size gate.
- Wire persona context into `extract_intent`, matching the pattern
  `synthesize` already uses.

## Non-goals

- No change to `category_affinity`'s averaging formula — already a fair
  running mean, not the noisy piece.
- No new Mongo update-pipeline atomicity fix for `record_signal`'s
  read-modify-write race — out of scope, tracked separately (see existing
  `repository.py` note).
- No backfill script for existing docs — migration happens lazily on next
  write (see Migration below).
- No change to `_sanitize_filters`/`_validate_categories` guardrails —
  persona remains a soft prior for `intent` tagging, never a source of
  filter values.

## Design

### 1. Vote-tally schema for expertise_level / query_style

Replace last-write-wins with a running tally, mirroring
`category_affinity`'s incremental spirit:

```json
{
  "expertise_votes": {"student": 0, "practitioner": 3, "expert": 1},
  "query_style_votes": {"broad": 1, "precise-citation": 3},
  "expertise_level": "practitioner",
  "query_style": "precise-citation",
  "category_affinity": {...},
  "query_count": 4,
  ...
}
```

`expertise_level`/`query_style` top-level fields are kept, always set to
the current tally mode (highest vote count; ties keep the previous mode
rather than churn) — so any other code reading the doc shape is
unaffected.

`merge_expertise_patch` changes from overwrite to: for each key present
in the validated patch, increment that value's vote count by 1, then
recompute the mode. A patch with no valid keys leaves tallies unchanged
(same as today's `if not patch: return existing`).

**Migration**: on the first `record_signal` call after this ships, if a
doc has the old string field but no tally field yet, seed
`expertise_votes = {old_value: 1}` (and same for `query_style_votes`)
before adding the new vote. No separate backfill step — docs migrate lazily
as their users query again. A doc that never queries again stays in the
old shape indefinitely, which is fine: `render_persona_context` (below)
falls back to reading the plain string field when no tally is present.

### 2. Trust gate in `render_persona_context`

```python
def render_persona_context(persona: dict | None) -> str:
    if not persona or persona.get("query_count", 0) < 20:
        return ""
    ...  # existing rendering logic, reading tally-derived mode
```

Threshold is a module-level constant (`_TRUST_THRESHOLD = 20`), not a
magic number, so it's discoverable and adjustable.

The elided rendering logic must handle both doc shapes: a doc can pass
the `query_count >= 20` gate while still being pre-migration (old plain
`expertise_level`/`query_style` string fields, no tally yet — migration
only happens on the next *write*, per §1). Render from the tally mode if
`expertise_votes`/`query_style_votes` are present, else fall back to the
plain string fields directly.

This only gates *surfacing*. `record_signal` keeps writing on every
query regardless of count — persona keeps accumulating signal from query
1, it's just not spoken about to the LLM until there's enough of it.

### 3. Shared relevance-judgment instruction

New constant in `persona/prompt.py` (next to `render_persona_context`,
since both consumers already import from there):

```python
RELEVANCE_INSTRUCTION = (
    "The note above is a prior about this user's typical usage, not a "
    "fact about this query. Use it only if this query is genuinely "
    "ambiguous on its own. If the query's own content conflicts with or "
    "is unrelated to the note, ignore the note and rely on the query "
    "alone."
)
```

Appended by both call sites wherever `persona_context` is non-empty —
`ai_mode/intent.py`'s user-message assembly and `ai_mode/synthesize.py`'s
`system_prompt` assembly — so the instruction can't drift between the two
prompts. This is a prompt-level, not code-level, mechanism: the same SLM
call already reasoning over query meaning for classification/synthesis
naturally extends that reasoning to judging relevance of the hint,
covering indirect mismatches (topically unrelated but no shared
vocabulary) the same way it covers direct ones — there is no separate
literal-match guardrail possible here (unlike `_sanitize_filters`), this
is pure prompt trust, at the same trust level the rest of `extract_intent`
already runs on.

### 4. `extract_intent` gains persona wiring

```python
async def extract_intent(
    gateway, query: str, on_step: OnStep | None = None, model: str | None = None,
    persona_context: str = "",
) -> dict:
    ...
    user_message = query
    if chunk_context is not None:
        user_message += f"\n\nStructural spans...:\n{chunk_context}"
    if persona_context:
        user_message += f"\n\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
    ...
```

Persona goes in the user message (per-query content), alongside the
existing `chunk_context` block — not the system prompt — matching the
existing pattern for query-specific injected context.

### 5. `synthesize.py`

```python
system_prompt = _SYSTEM_PROMPT
if persona_context:
    system_prompt += f"\n{persona_context}\n{RELEVANCE_INSTRUCTION}"
```

### 6. `pipeline.py` wiring

`run_ai_mode` already receives `persona_context` and forwards it to
`synthesize`. Add the same forwarding to the `extract_intent` call:

```python
intent_result = await extract_intent(gateway, query, on_step=on_step, persona_context=persona_context)
```

No change needed in `ws.py` — it already computes `persona_context` once
per request and passes it into `run_ai_mode`.

## Data flow (updated)

```
ws.py: get_persona() -> render_persona_context() [gated on query_count>=20]
                              |
                              v
                      persona_context (str, "" if thin or no persona)
                              |
              +---------------+---------------+
              v                               v
      extract_intent (NEW)              synthesize (existing)
      user_message += persona_context   system_prompt += persona_context
      + RELEVANCE_INSTRUCTION           + RELEVANCE_INSTRUCTION
```

Write side (unchanged shape, changed merge logic):

```
ai_mode_result.intent -> record_persona_signal -> extract_expertise_signal (SLM)
                                                 -> record_signal
                                                      -> merge_expertise_patch (NEW: tally, not overwrite)
                                                      -> merge_category_affinity (unchanged)
                                                      -> query_count += 1
```

## Testing

- `persona/tests/test_merge.py`: tally accumulation across multiple
  patches; mode computation incl. tie-keeps-previous; old-string-doc
  migration seeds a 1-vote tally correctly.
- `persona/tests/test_prompt.py`: gate returns `""` at `query_count` 0,
  1, 19; renders at 20+; renders from tally mode not raw overwritten
  field; `RELEVANCE_INSTRUCTION` constant exported and stable text.
- `retrieval-api/tests/test_ai_mode_intent.py`: `persona_context` param
  reaches `user_message`; empty `persona_context` (default) unchanged
  from current behavior; instruction text present when non-empty.
- `retrieval-api/tests/test_ai_mode_synthesize.py`: same, for
  `system_prompt`.
- `retrieval-api/tests/test_ai_mode_pipeline.py`: `persona_context`
  forwarded to both `extract_intent` and `synthesize` calls.
- `retrieval-api/tests/test_ws_persona_wiring.py`: existing gate gets
  exercised at the ws.py level implicitly via `render_persona_context`;
  no new ws.py logic to test directly (unchanged file).

## Open questions / risks

- Tie-break on tally mode (`{practitioner: 2, student: 2}`) keeps
  previous mode rather than picking arbitrarily — needs the previous
  mode passed into the mode-computation function, a small signature
  change from today's stateless `merge_expertise_patch(existing, patch)`
  (it already receives `existing`, so the previous mode is available on
  that dict — no new parameter needed, just reading it before overwrite).
- Threshold of 20 is a starting value agreed on in this design session,
  not derived from data — worth revisiting once real query-count
  distributions are observed.
