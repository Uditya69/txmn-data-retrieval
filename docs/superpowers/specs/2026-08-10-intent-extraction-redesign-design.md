# Intent extraction redesign — design

Redesign of AI Mode's `slm` stage (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`):
enforce schema-valid model output instead of prompt-only free text, extract the full set
of filterable ES fields, and give the currently-dead `intent` label an actual downstream
effect. Motivated by user report: extraction "feels random" — rewrites/filters aren't
reliably grounded in the corpus schema.

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
   model's attention/tokens for no behavioral effect.
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

- **No intent-based Milvus collection routing** (CLAUDE.md hard rule). The 7 Milvus
  collections are different textual facets of the *same* judgments (summary, digest,
  headnotes, facts, held, ruling, metadata), not topical categories — skipping one based
  on a fallible classifier risks silently losing the exact passage that answers the
  query. Intent's influence is scoped to **ranking only** (RRF weighting), which cannot
  reduce recall the way collection-skipping would.
- Not attempting citation-lookup query fast-paths, synthesis prompt-shape changes, or any
  other intent consumer beyond RRF weighting in this pass — those are separate,
  independently-scoped future work if desired.
- Not changing `_safe_rewrite`'s or `_sanitize_filters`'s anti-hallucination logic — those
  guard against the model inventing content even inside a valid schema, and stay as-is.

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

### 3. Intent taxonomy → RRF weighting

Four labels: `citation_lookup` (query anchored on a party/case name or citation),
`provision_lookup` (anchored on a section/act/rule number), `conceptual` (open legal
question, no strong lexical anchor), `unknown` (fallback — anything the model can't
confidently classify, including today's `_fallback_intent` degrade path).

`retrieve.py`'s RRF stage takes an intent-derived weight pair (dense_weight,
sparse_weight): `citation_lookup`/`provision_lookup` bias sparse higher (lexical/BM25
signal is more reliable for exact-anchor queries), `conceptual` biases dense higher,
`unknown` keeps today's neutral 50/50 — meaning any query the model can't classify gets
exactly today's behavior, not a new failure mode.

### 4. Filter-accuracy eval dataset

A new small gold-filter set (10-15 queries), reusing existing direct-class queries from
`docs/retrieval-eval-queries.md` that already contain literal court/act/section/party/date
mentions (e.g. Q01, Q11, Q30 name a court/section pair explicitly), each annotated with the
expected extracted filter dict. Lives alongside the existing eval query doc/JSON (exact
file TBD at plan time — likely a new `evals/intent_filters.json` mirroring
`evals/retrieval_cases.json`'s pattern). A test/check runs `extract_intent` against each
query and asserts the extracted filters match the gold dict (allowing the model's exact
string casing/formatting where the corpus itself doesn't canonicalize).

### 5. Model candidate: gemma-4-E4B-it

Two-stage validation before adoption, cheapest-first:
1. **Prompt-only pass**: run `extract_intent` directly (no ES/Milvus) against the new
   gold-filter set and a handful of rewrite-quality queries; eyeball/assert against gold.
2. **Full retrieval-rank pass**: one run of the existing 12-query stratified sample via
   `retrieval_eval.py --slm-model google/gemma-4-E4B-it`. This **cannot** use the
   `--cache-dir` stage cache the way synthesis-only comparisons did — `slm_model` is part
   of the cache key precisely because it changes `rewritten_query`, which changes the
   embedding and everything downstream. One full (~7-9 min) run is unavoidable.

Adopt `gemma-4-E4B-it` for `DEEPINFRA_CHAT_MODEL_SLM` only if step 2 shows no retrieval-rank
regression vs. the current `Qwen/Qwen3-30B-A3B` baseline (same bar `docs/small-model-eval-
results.md` used to adopt Qwen3-30B-A3B originally). If it regresses, keep
`Qwen/Qwen3-30B-A3B` and land steps 1-4 above independently of the model swap — the schema/
filter/intent-weighting redesign is valuable regardless of which model sits behind it.

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
- `retrieve.py`: new tests for RRF weighting per intent label, including the `unknown` →
  today's-50/50 no-regression case.
- `es_client.py`: new test for `bench`/`judge` term-filter construction.
- `GatewayClient`/DeepInfra adapter: new test asserting `response_format` is passed
  through on the chat call when provided.
- New filter-accuracy check against the gold-filter eval set (item 4 above) — run
  manually/on-demand like the existing `retrieval_eval.py` harness, not part of the
  `uv run pytest` suite (external ES/Milvus/gateway dependency, same as the existing
  retrieval-rank eval).
