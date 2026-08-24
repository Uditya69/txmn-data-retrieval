# Intent → category classification redesign — design

Replaces the AI Mode `slm` stage's output shape (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`).

Supersedes `docs/superpowers/specs/2026-08-11-intent-category-classification-design.md`, which
speced the same category taxonomy but was never implemented (work went a different direction —
`bb6b0da`/`49007b4` added chunk-context injection to the old 4-value-intent code instead). Reused
here: the six-category taxonomy and its definitions, the `original_query`/`intent`/`search_query`
output shape. Changed here: **filters extraction is kept**, not dropped — the 08-11 spec retired
`filters`/`doc_id_allowlist` resolution along with the old intent enum; this version keeps that
machinery untouched and adds category as a purely additive field.

## Motivation

The 4-value `intent` enum (`citation_lookup`/`provision_lookup`/`conceptual`/`unknown`) classifies
how a query is anchored (by citation, by provision number, or not at all) but says nothing about
*what kind of content* the user wants. Requested instead: classify each query against the actual
content categories Taxmann's platform covers — acts, rules, caselaws, articles, commentary,
tariff — so the tag is meaningful on its own (visible in traces/responses) even before any
retrieval-side consumer (collection routing) exists for it. Collection routing itself is an
explicit non-goal of this spec (see Scope decision).

## Scope decision (confirmed with user)

- Category is **classification only, no retrieval-side routing yet**. `act_section`,
  `rule_section`, `article_section`, `commentary_section` aren't even wired into
  `common.schemas.MILVUS_COLLECTIONS` yet (only the 7 caselaw-flavored collections are); wiring
  them in and routing by category is future work, out of scope here.
- **`CLAUDE.md` hard rule 4** ("AI Mode searches all 7 Milvus collections every query. No
  intent-based collection routing.") is not touched by this spec — nothing here routes anything.
  Revisit that rule's wording only when routing is actually built.
- **`filters` extraction is kept as-is**, running alongside the new `intent` category field.
  `_sanitize_filters`, `resolve_allowlist`, and `pipeline.py`'s allowlist span are all unchanged.
- The old 4-value `intent` enum is **replaced**, not kept alongside — `citation_lookup` /
  `provision_lookup` / `conceptual` are dropped. Two downstream consumers of that enum go inert
  as a direct, accepted consequence:
  - `retrieve.py`'s `_INTENT_RRF_WEIGHTS` dense/sparse weighting — deleted; RRF weighting is
    always neutral (1.0/1.0) until a category-based replacement is designed later.
  - `intent.py`'s section-filter gate (`if key == "section" and intent != "provision_lookup"`) —
    the value it compared against no longer exists, so left as dead code it would silently
    always take the "drop `section`" branch. Made explicit instead: `section` filter is
    unconditionally dropped in `_sanitize_filters`, with a comment explaining this is disabled
    pending a category-based gate, not a bug.
- `intent_eval.py`'s harness is **extended**, not replaced — it keeps scoring `filters` accuracy
  and gains category-list accuracy scoring alongside.

## Output shape

```json
{
  "original_query": "<verbatim input query, unchanged>",
  "intent": ["acts", "caselaws"],
  "search_query": "<conservative rewrite, same normalization rules as before>",
  "filters": {"court": "Bombay High Court"}
}
```

- `original_query`: the raw input query, passed through unchanged. **Not produced by the model**
  — the prompt schema does not ask for it; `extract_intent` already has `query` as a plain
  argument and sets `"original_query": query` directly in code. Asking the SLM to echo it back
  would waste tokens and add a needless hallucination surface (a model could "helpfully" normalize
  whitespace/casing on echo) for a value the code already holds verbatim.
- `intent`: a list of zero or more of the six category labels below. Multi-label — a query may
  genuinely belong to more than one category (e.g. "case law on section 54F exemption
  eligibility" is both `acts` and `caselaws`). Empty list is valid and replaces today's
  `"unknown"` — used when no category confidently applies.
- `search_query`: renames today's `rewritten_query` field. Same conservative, filler-stripped
  normalization logic (`_safe_rewrite`'s anti-hallucination guardrails: preserve every
  number/identifier, ≥60% token overlap, no invented Act/court names) — unchanged, just renamed.
- `filters`: unchanged shape and validation (`_sanitize_filters`), except the `section` key is
  now unconditionally dropped (see Scope decision).

## Category taxonomy (prompt content)

Six labels, each given a full definition + signal words + worked example + disambiguation from
its nearest neighbor in the system prompt, so the model has a broad enough view to classify
confidently rather than guessing off a bare label name. Grounded in real `tm-dp/data/*` documents,
not assumed:

- **acts** — primary legislation itself (Income-tax Act 1961, CGST Act, Customs Act, BNS, etc.):
  sections, sub-sections, provisos, definitions, schedules. Real doc shape: heading `"Section - N"`.
  Signal: "section", "as per the Act", "definition under", bare section+Act reference with no
  request for judicial interpretation.
- **rules** — subordinate legislation notified *under* an Act (Income-tax Rules 1962, CGST Rules,
  Customs Valuation Rules): procedure, computation mechanics, prescribed forms. Real doc shape:
  heading `"Rule - N"`. Distinct from `acts` by whether the query's number is a "rule" vs a
  "section"; a rules query often co-occurs with `acts` since every Rule has a parent Act.
- **caselaws** — judicial decisions (Supreme Court, High Courts, ITAT, CESTAT, AAR): what was
  decided for a dispute/fact pattern. Signal: party names ("X vs Y"), "held", "case law on",
  "precedent for", a citation string, bench/judge name.
- **articles** — expert-authored opinion/analysis published in a journal/magazine, cited like
  `[1994] 75 Taxman 167 (ART)` (real doc field: `sortbyauthor`): trend, controversy, recent
  development, practical impact. Not the publisher's own explanation (that's `commentary`) and
  not binding law. Tag only on explicit signal ("article on...", "expert opinion on...", a named
  author) — don't default here.
- **commentary** — Taxmann's own provision-by-provision plain-language explanation of how a
  section/Act/rule works in practice, no author byline (real doc heading shape:
  `"CHAPTER N: CLAUSE M..."`). Distinct from `acts` (raw statutory text) and `articles` (named
  author's opinion piece). Default landing spot for "explain X" / "how does X work" queries that
  aren't clearly `articles`.
- **tariff** — customs/GST tariff classification and rates: HSN code lookups, duty rates, rate
  schedules, exemption notifications tied to a specific tariff heading/good. Distinct from
  `acts`/`rules` even though tariff notifications are issued under that law — if the ask is "what
  HSN/duty rate for [a specific good]", it's `tariff`.

Prompt also states explicitly: output every category that genuinely applies (don't force a single
pick), but don't over-list — only categories the query actually anchors on.

## Code changes

### `intent.py`
- System prompt: add the category taxonomy block above (multi-label array output, `search_query`
  framing replacing `rewritten_query`). Keep the existing `filters` section, `_LEGAL_MARKERS`
  forbidden-rewrite examples, and structural-span injection (`_build_chunk_context`) unchanged.
- `_ALLOWED_INTENTS` → `_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}`.
- `_validate_result(query, result)`: validate `intent` as a list — drop values outside
  `_ALLOWED_CATEGORIES`, dedupe, empty list is valid. Rename `rewritten_query` field to
  `search_query`. Set `"original_query": query` from the function's own `query` argument, not
  from `result`. `filters` validation unchanged except the `section` key is now always dropped
  (comment explains why, per Scope decision).
- `_fallback_intent(query)`: `{"original_query": query, "intent": [], "search_query": query, "filters": {}}`.
- `_safe_rewrite`: unchanged logic, applied to `search_query`.

### `pipeline.py`
- Field renames only (`rewritten_query`→`search_query`). `resolve-allowlist` span and
  `resolve_allowlist()` call stay as they are today — still fed from `intent_result["filters"]`.
- `retrieve()` call: stop passing `intent_result["intent"]` (now a list; wrong type for
  `retrieve()`'s dict-keyed lookup). No replacement passed — `retrieve()`'s `intent` param keeps
  its `"unknown"` default.

### `retrieve.py`
- Delete `_INTENT_RRF_WEIGHTS`. `rrf_merge` call always uses `dense_weight=1.0, sparse_weight=1.0`.
  Drop the `intent` parameter from `retrieve()`'s signature entirely (nothing computes it
  meaningfully anymore) rather than leave an always-unused parameter — update `pipeline.py`'s
  call site to match.
- `get_settings().intent_rrf_weighting_enabled` (`common/config.py`): no longer referenced here.
  Leave the settings field in place (removing a settings field is a separate, unrelated cleanup)
  but note it's now dead in a comment at the field.

### `retrieval_eval.py`
- Same field renames (`rewritten_query`→`search_query`). Drop the intent-based RRF weight lookup
  (`_INTENT_RRF_WEIGHTS.get(intent.get("intent"), ...)`) — weights hardcoded to `(1.0, 1.0)`,
  matching `retrieve.py`.

### `intent_eval.py` (extended, not replaced)
- `load_intent_cases`: required keys become `{"id", "query", "expected_filters", "expected_categories"}`.
- `check_intent_case`: two checks — `expected_filters == actual_filters` (unchanged) and
  `set(expected_categories) == set(actual["intent"])` (new, order-independent). A case fails if
  either check fails; report which one(s) failed in the printed line.
- `run()`/`main()`: same shape, prints expected vs actual for both filters and categories.

### `evals/intent_filter_cases.json`
- Extended in place: add `"expected_categories"` to each of the existing 12 cases (they're all
  caselaw-flavored — filter extraction cases — so most will be `["caselaws"]` or `[]`, matching
  the query). Add new cases covering `acts`, `rules`, `articles`, `commentary`, `tariff`, and a
  couple of multi-label cases, so all six categories and multi-label get real coverage. Exact
  additions finalized at plan time.

**Cache invalidation note:** `retrieval_eval.py`'s on-disk stage cache
(`stage_cache_path(cache_dir, case_id, slm_model, reranker_model)`) does not key on prompt
version — cached `rewritten_query` entries from before this change will silently look valid but
reflect the old prompt/schema. Clear or bump the eval cache directory when this ships.

## Error handling
- Unchanged fallback path: schema-mode JSON that still fails `json.loads` (or the SLM refusing) →
  `_fallback_intent`, same as today.
- Category values outside the 6-label set: dropped, not error — matches `_sanitize_filters`'s
  existing silent-drop posture for unrecognized filter keys.
- Empty `intent` list is a valid, non-error outcome (replaces `"unknown"`).

## Testing
- `intent.py`: existing tests updated for new field names (`search_query`, `original_query`); new
  tests for multi-label output, empty-list case, category values outside the allowed set being
  dropped, and `filters` still working unchanged (including `section` now always dropped
  regardless of category).
- `pipeline.py`: tests updated for the `search_query` rename and the `retrieve()` call no longer
  passing `intent`. `resolve_allowlist` call assertions unchanged.
- `retrieve.py`, `retrieval_eval.py`: tests updated to drop assertions on intent-based RRF weight
  selection — always neutral now.
- `intent_eval.py`: manual/on-demand run against the extended dataset (same posture as today —
  not part of `uv run pytest`, external gateway dependency).
