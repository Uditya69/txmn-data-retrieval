# Intent → category classification redesign — design

Replaces the AI Mode `slm` stage's output shape (`packages/retrieval-api/src/retrieval_api/ai_mode/intent.py`).
Supersedes `docs/superpowers/specs/2026-08-10-intent-extraction-redesign-design.md`, which
expanded the old filter-extraction taxonomy (added `bench`/`judge`) and kept the 4-value
`intent` enum (`citation_lookup`/`provision_lookup`/`conceptual`/`unknown`) classified-but-unused.
That direction is abandoned here: filters and the 4-value enum are dropped, replaced by a
multi-label content-category classification.

## Motivation

User report: the query expander "not able to extract anything useful" — the 4-value intent
enum and filter extraction weren't giving a legible signal of *what the user is actually
asking about*. Requested instead: classify each query against the actual content categories
Taxmann's platform covers (acts, rules, caselaws, articles, commentary, tariff), so the
signal is meaningful on its own even before any retrieval-side consumer exists for it.

## Scope decision (confirmed with user)

- Category is **classification only for now** — no retrieval-side filtering/routing yet.
  Filtering by category is an explicit future step, once labeling quality is verified
  (mirrors the prior spec's own phasing instinct, applied to the new taxonomy instead).
- Old `intent` enum and `filters` extraction are **dropped, not kept alongside** the new
  field. This retires: doc_id allowlist filtering (`filter_resolve.py`'s consumer in
  `pipeline.py`) and RRF intent-based dense/sparse weighting (`retrieve.py`'s
  `_INTENT_RRF_WEIGHTS`) — both go inert (see Pipeline impact below). Accepted as an
  intentional side effect; can be rebuilt later on top of the new category signal.
- `intent_eval.py`'s harness is **rewritten**, not deleted, to score category-list accuracy
  instead of filter-dict accuracy.

## Output shape

```json
{
  "original_query": "<verbatim input query, unchanged>",
  "intent": ["acts", "caselaws"],
  "search_query": "<conservative rewrite, same normalization rules as before>"
}
```

- `original_query`: the raw input query, passed through unchanged. **Not produced by the
  model** — the prompt schema does not ask for it; `extract_intent` already has `query` as
  a plain argument and sets `"original_query": query` directly in code. Asking the SLM to
  echo it back would waste tokens and add a needless hallucination surface (a model could
  "helpfully" normalize whitespace/casing on echo) for a value the code already holds
  verbatim.
- `intent`: a list of zero or more of the six category labels below. Multi-label —
  a query may genuinely belong to more than one category (e.g. "case law on section 54F
  exemption eligibility" is both `acts` and `caselaws`). Empty list allowed when no
  category confidently applies (replaces today's `"unknown"`).
- `search_query`: replaces today's `rewritten_query` field name only — the conservative,
  filler-stripped normalization logic (`_safe_rewrite`'s anti-hallucination guardrails:
  preserve every number/identifier, ≥60% token overlap, no invented Act/court names) is
  unchanged, just renamed.

## Category taxonomy (prompt content)

Six labels, each given a full definition + signal words + worked example + disambiguation
from its nearest neighbor in the system prompt, so the model has a broad enough view to
classify confidently rather than guessing off a bare label name:

- **acts** — primary legislation itself (Income-tax Act 1961, CGST Act, Customs Act, BNS,
  etc.): sections, sub-sections, provisos, definitions, schedules. Signal: "section",
  "as per the Act", "definition under", bare section+Act reference with no request for
  judicial interpretation.
- **rules** — subordinate legislation notified *under* an Act (Income-tax Rules 1962, CGST
  Rules, Customs Valuation Rules): procedure, computation mechanics, prescribed forms.
  Distinct from `acts` by whether the query's number is a "rule" vs a "section"; a rules
  query often co-occurs with `acts` since every Rule has a parent Act.
- **caselaws** — judicial decisions (Supreme Court, High Courts, ITAT, CESTAT, AAR): what
  was decided for a dispute/fact pattern. Signal: party names ("X vs Y"), "held", "case law
  on", "precedent for", a citation string, bench/judge name.
- **articles** — expert-authored opinion/analysis published in a journal/magazine: trend,
  controversy, recent development, practical impact. Not the publisher's own explanation
  (that's `commentary`) and not binding law.
- **commentary** — publisher's own provision-by-provision plain-language explanation of how
  a section/Act/rule works in practice. Distinct from `acts` (raw statutory text) and
  `articles` (named author's opinion piece).
- **tariff** — customs/GST tariff classification and rates: HSN code lookups, duty rates,
  rate schedules, exemption notifications tied to a specific tariff heading/good. Distinct
  from `acts`/`rules` even though tariff notifications are issued under that law — if the
  ask is "what HSN/duty rate for [a specific good]", it's `tariff`.

Prompt also states explicitly: output every category that genuinely applies (don't force a
single pick), but don't over-list — only categories the query actually anchors on.

## Code changes

### `intent.py`
- New system prompt: category definitions above, multi-label array output, `search_query`
  framing. Prompt schema requests only `intent` and `search_query` from the model —
  `original_query` is never asked for (see Output shape note above). Drop the `filters`
  section of the prompt entirely. Keep the existing "conservative rewrite" rules and
  forbidden-rewrite examples, reframed under `search_query`.
- `_ALLOWED_INTENTS` → `_ALLOWED_CATEGORIES = {"acts", "rules", "caselaws", "articles", "commentary", "tariff"}`.
- `_validate_result(query, result)`: validates `intent` as a list (drop values outside
  `_ALLOWED_CATEGORIES`, dedupe, empty list is valid), renames `rewritten_query` field to
  `search_query`. Sets `"original_query": query` from the function's own `query` argument,
  not from `result`. Drop `_sanitize_filters` call and the function itself (no longer used).
- `_fallback_intent(query)`: `{"original_query": query, "intent": [], "search_query": query}`.
- `_safe_rewrite`: unchanged logic, applied to `search_query`.
- `_ALLOWED_FILTERS`, `KNOWN_FILTER_FIELDS` prompt wiring in `schema_context.py`: no longer
  referenced by this prompt — leave `schema_context.py` itself alone (still used for the
  collection-description block), just stop pulling filter-field docs into this prompt.

### `pipeline.py`
- Drop the `resolve-allowlist` span and `resolve_allowlist()` call entirely (no `filters`
  to feed it) — `doc_id_allowlist` passed to `retrieve()` becomes always `None`.
- `retrieve()` call: stop passing `intent_result["intent"]` (now a list; wrong type for
  `retrieve()`'s dict-keyed lookup). No replacement passed — `retrieve()`'s `intent` param
  keeps its `"unknown"` default.

### `retrieve.py`
- No signature change. `intent` param stays defaulted `"unknown"`; since nothing passes it
  anymore, `_INTENT_RRF_WEIGHTS.get(intent, ...)` always resolves to `(1.0, 1.0)` —
  weighting goes inert, as scoped. Smallest possible diff; can be revisited when a category
  based RRF signal is designed later.

### `retrieval_eval.py`
- Same field renames (`rewritten_query`→`search_query`). Drop the `resolve_allowlist` call
  and the intent-based RRF weight lookup (`_INTENT_RRF_WEIGHTS.get(intent.get("intent"), ...)`)
  — weights hardcoded to `(1.0, 1.0)`.

### `intent_eval.py` (rewritten, not deleted)
- `load_intent_cases`: required keys become `{"id", "query", "expected_categories"}`.
- `check_intent_case(expected: list, actual: list) -> bool`: set comparison,
  `set(expected) == set(actual)` (order-independent).
- `run()`/`main()`: same shape, field names updated; prints expected vs actual category
  lists.

### `evals/intent_filter_cases.json` → rewritten dataset
Existing 12 cases are all caselaw-flavored (built for filter extraction) — renaming the
field alone would only exercise one of six categories. Replace with a new set spanning all
six categories including multi-label cases. Starting set (to grow over time):

```json
[
  {"id": "C01", "query": "definition of capital asset under section 2(14)", "expected_categories": ["acts"]},
  {"id": "C02", "query": "Rule 3 perquisite valuation method", "expected_categories": ["rules"]},
  {"id": "C03", "query": "case law for Ramesh Gupta vs. Income-tax Officer", "expected_categories": ["caselaws"]},
  {"id": "C04", "query": "article on GST implications of the new e-invoicing mandate", "expected_categories": ["articles"]},
  {"id": "C05", "query": "explain how section 54F exemption works", "expected_categories": ["commentary"]},
  {"id": "C06", "query": "HSN code and GST rate for solar panels", "expected_categories": ["tariff"]},
  {"id": "C07", "query": "case law on section 54F exemption eligibility", "expected_categories": ["acts", "caselaws"]},
  {"id": "C08", "query": "Delhi High Court ruling under the Income-tax Act, 1961 on capital gains exemption", "expected_categories": ["acts", "caselaws"]},
  {"id": "C09", "query": "Rule 6(3)(c) input tax credit reversal explained", "expected_categories": ["rules", "commentary"]},
  {"id": "C10", "query": "Can a court order investigators to trace and seize a company's property when investors were defrauded?", "expected_categories": []}
]
```
C10 mirrors the old dataset's `F10` (which asserted `expected_filters: {}`) — a general
conceptual question with no literal category anchor. Kept as `[]` rather than guessed into
`caselaws` just because it mentions "court", to preserve its original job: exercise the
empty-list path.

Exact filename TBD at plan time (likely `evals/intent_category_cases.json`, replacing
`evals/intent_filter_cases.json`).

**Cache invalidation note:** `retrieval_eval.py`'s on-disk stage cache
(`stage_cache_path(cache_dir, case_id, slm_model, reranker_model)`) does not key on prompt
version — cached `rewritten_query` entries from before this change will silently look valid
but reflect the old prompt/schema. Clear or bump the eval cache directory when this ships.

## Error handling
- Unchanged fallback path: schema-mode JSON that still fails `json.loads` (or the SLM
  refusing) → `_fallback_intent`, same as today.
- Category values outside the 6-label set: dropped, not error — matches
  `_sanitize_filters`'s existing silent-drop posture for unrecognized keys.
- Empty `intent` list is a valid, non-error outcome (replaces `"unknown"`).

## Testing
- `intent.py`: existing tests updated for new schema/field names; new tests for multi-label
  output, empty-list case, and category values outside the allowed set being dropped.
- `pipeline.py`, `retrieval_eval.py`: tests updated to stop asserting on `filters`/old
  `intent` enum/allowlist-resolution calls.
- `intent_eval.py`: manual/on-demand run against the new dataset (same posture as today —
  not part of `uv run pytest`, external gateway dependency).
