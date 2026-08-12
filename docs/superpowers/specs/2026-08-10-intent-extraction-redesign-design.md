# Intent extraction redesign — design

Redesign of AI Mode's `slm` stage (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`):
enforce schema-valid model output instead of prompt-only free text, and extract the full
set of filterable ES fields. Motivated by user report: extraction "feels random" —
rewrites/filters aren't reliably grounded in the corpus schema.

**Phased**: get intent/filter extraction schema-correct and confidence-checked first
(Phase 1, this plan). Retrieval-side changes — including using the `intent` label for
anything at all — are explicitly deferred to a later Phase 2, once Phase 1 is trusted and
the retrieval pipeline itself is more settled ("its not mature atp"). This plan does not
wire `intent` into RRF weighting or anything else; `intent` remains classified and emitted
on the trace event, same as today, just via the new schema-enforced mechanism.

## Problems found (current state)

1. **No schema enforcement.** `extract_intent` prompts for JSON and parses it by finding
   the outermost `{...}` via regex (`_extract_json_object`). The model is free to emit
   anything JSON-shaped or not; correctness rests entirely on the prompt plus post-hoc
   guards (`_safe_rewrite`, `_sanitize_filters`). DeepInfra supports native
   `response_format` JSON-schema mode on nearly all models, free, no perf cost — unused.
2. **Filter fields incomplete.** `_ALLOWED_FILTERS` = `{court, act, section, date_range,
   party}`. The ES schema also has queryable `masterinfo.info.bench` and `otherinfo.judge`
   (used today only for citation display, `es_client.py`'s `fetch_document_metadata`) —
   never surfaced to the SLM, so a query naming a bench or judge can never turn into a
   filter no matter how well the model classifies.
3. **`intent` is dead output.** Computed, validated, emitted on the `on_step` trace event —
   and never read by `pipeline.py` or `retrieve.py`. Classifying it costs a share of the
   model's attention/tokens for no behavioral effect. **Out of scope for this plan** — see
   Phasing above; this plan keeps `intent` classified-but-unused, same as today, and only
   fixes *how* it (and the filters) get extracted.
4. **No filter-accuracy eval dataset.** The existing 53-query set
   (`docs/retrieval-eval-queries.md`) has gold `doc_id`s for retrieval-rank scoring only —
   no query has a documented gold filter extraction. Today there is no automated signal
   for "did the SLM extract the right court/act/section/party," only "did the pipeline
   still find the right document despite whatever it extracted."
5. **Model choice was already fixed correctly** (`Qwen/Qwen3-30B-A3B`, adopted per
   `docs/small-model-eval-results.md` — clean pass on this exact task). This design
   evaluates swapping to `google/gemma-4-E4B-it` (strong structured-output track record on
   the synthesis role) as a candidate, not a blind replacement.

## Non-goals / explicit constraints

- **No retrieval-side changes of any kind in this plan** — that includes RRF weighting,
  any other `intent` consumer, and any change to `retrieve.py`. Deferred to Phase 2 per
  the Phasing note above, once Phase 1's extraction is trusted and the retrieval pipeline
  is more settled. `intent`'s eventual Phase-2 use is still constrained by CLAUDE.md's
  hard rule against intent-based Milvus collection routing — the 7 Milvus collections are
  different textual facets of the *same* judgments (summary, digest, headnotes, facts,
  held, ruling, metadata), not topical categories, so any future use of `intent` should
  stay ranking-only (e.g. RRF weighting), never collection selection.
- Not attempting citation-lookup query fast-paths or synthesis prompt-shape changes.
- Not changing `_safe_rewrite`'s or `_sanitize_filters`'s anti-hallucination logic — those
  guard against the model inventing content even inside a valid schema, and stay as-is.
- No full `retrieval_eval.py` retrieval-rank run in this plan (that exercises the
  retrieval pipeline, which is out of scope). Model/mechanism confidence comes only from
  the prompt-only gold-filter check (item 4 below).

## Design

### 1. Schema-enforced extraction

`GatewayClient.chat()` (and the DeepInfra adapter beneath it) gains an optional
`response_format` passthrough parameter. `extract_intent` calls it with a JSON-schema
`response_format` matching the shape below; on a schema-mode response, `intent.py` calls
`json.loads(response)` directly — `_extract_json_object`'s brace-finding regex is deleted.
The `_validate_result`/`_safe_rewrite`/`_sanitize_filters` content guards are unchanged;
schema mode guarantees shape, not truthfulness.

Output shape:

```json
{
  "rewritten_query": "string",
  "intent": "citation_lookup | provision_lookup | conceptual | unknown",
  "filters": {
    "court": "string?", "act": "string?", "section": "string?", "party": "string?",
    "bench": "string?", "judge": "string?",
    "date_range": {"gte": "date?", "lte": "date?"}
  }
}
```

### 2. Filter field expansion

Add `bench` and `judge`:
- `intent.py`: `_ALLOWED_FILTERS` gains `"bench"`, `"judge"`.
- `schema_context.py`: `KNOWN_FILTER_FIELDS` gains `"bench"`, `"judge"`; prompt context
  documents them alongside the existing fields.
- `es_client.py`: `_TERM_FILTER_FIELDS` gains `bench` → `masterinfo.info.bench`,
  `judge` → `otherinfo.judge` (term-filter, same pattern as `court`/`act`/`section`).
- Prompt keeps the existing "only if literally in the query" instruction for these two
  fields as well — same anti-hallucination posture as the other four.
- `court`'s known unreliability (frequently empty across the corpus, per `es_client.py`'s
  existing comment) is left as-is; not fixed by this design, just not made worse.

### 3. Intent taxonomy (classification only — no consumer yet)

Four labels: `citation_lookup` (query anchored on a party/case name or citation),
`provision_lookup` (anchored on a section/act/rule number), `conceptual` (open legal
question, no strong lexical anchor), `unknown` (fallback — anything the model can't
confidently classify, including today's `_fallback_intent` degrade path).

This enum is defined now (it's part of the schema in item 1) so the model has a concrete,
constrained target to classify into rather than an open-ended "one short label" string —
that constraint itself is expected to improve classification consistency. But nothing
consumes the label yet; it's emitted on the `on_step` trace event exactly as today. Phase 2
(separate plan, later) decides how the label affects retrieval, e.g. RRF dense/sparse
weighting — and per the Non-goals section above, that consumer must stay ranking-only.

### 4. Filter-accuracy eval dataset

A new small gold-filter set (10-15 queries), reusing existing direct-class queries from
`docs/retrieval-eval-queries.md` that already contain literal court/act/section/party/date
mentions (e.g. Q01, Q11, Q30 name a court/section pair explicitly), each annotated with the
expected extracted filter dict. Lives alongside the existing eval query doc/JSON (exact
file TBD at plan time — likely a new `evals/intent_filters.json` mirroring
`evals/retrieval_cases.json`'s pattern). A test/check runs `extract_intent` against each
query and asserts the extracted filters match the gold dict (allowing the model's exact
string casing/formatting where the corpus itself doesn't canonicalize).

### 5. Model candidate: gemma-4-E4B-it — prompt-only validation

Per the phasing decision, this plan does **not** run the full `retrieval_eval.py`
retrieval-rank sample (that exercises retrieval, which is explicitly deferred). Validation
is the prompt-only gold-filter check from item 4: run `extract_intent` directly (no
ES/Milvus) with `model="google/gemma-4-E4B-it"` against the gold-filter set and a handful
of rewrite-quality queries, and compare pass rate against the same check run with the
current `Qwen/Qwen3-30B-A3B`.

Adopt `gemma-4-E4B-it` for `DEEPINFRA_CHAT_MODEL_SLM` if it matches or beats
`Qwen/Qwen3-30B-A3B` on this check. If it's worse, keep `Qwen/Qwen3-30B-A3B` and land items
1-4 above independently of the model swap — the schema/filter-extraction redesign is
valuable regardless of which model sits behind it. Either way, a full retrieval-rank
confirmation (the two-stage validation originally proposed) is Phase 2/future work, once
retrieval-side changes are actually happening and there's a reason to re-run the full
pipeline eval anyway.

## Error handling

- Schema-mode response that still fails `json.loads` (provider-side schema-mode bug, or a
  model that doesn't honor `response_format`): falls back to today's `_fallback_intent`
  path exactly as a JSON-decode failure does now — no new failure mode introduced.
- Missing/malformed `bench`/`judge` filter values: same sanitization path as existing
  filters (`_sanitize_filters`) — dropped silently if not literally present in the query.
- Intent label outside the 4-value enum (schema mode should prevent this, but a
  non-compliant model could still emit stray text in a non-schema-mode fallback path):
  treated as `unknown`, i.e. neutral RRF weighting — never crashes the pipeline.

## Testing

- `intent.py`: existing tests updated for the new schema call path (mock `gateway.chat`
  returning schema-mode JSON directly, no more brace-finding fixture inputs needed) plus
  new tests for `bench`/`judge` sanitization and the 4-value intent enum.
- `es_client.py`: new test for `bench`/`judge` term-filter construction.
- `GatewayClient`/DeepInfra adapter: new test asserting `response_format` is passed
  through on the chat call when provided.
- New filter-accuracy check against the gold-filter eval set (item 4 above) — run
  manually/on-demand like the existing `retrieval_eval.py` harness, not part of the
  `uv run pytest` suite (external ES/Milvus/gateway dependency, same as the existing
  retrieval-rank eval).
