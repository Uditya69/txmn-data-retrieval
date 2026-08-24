# Instant mode ES retrieval redesign — design

Scope: `packages/common/src/common/es_client.py` (`raw_search`, used only by Instant mode's
`_run_es` in `packages/retrieval-api/src/retrieval_api/instant/search.py`). AI Mode's intent/
category taxonomy and filter-field fixes are a separate, later spec — not in scope here.

> **Update, post-implementation: section 2's `function_score` ranking boost is disabled.**
> First, two missing/zero-value bugs in the formula below were found and patched (landmarkruling
> populated on only 2.1% of the corpus, `missing: 0.0001` compounding through
> log2p+factor+`boost_mode: multiply` into a ~30,000x penalty for the other 98%; court_boost
> a real, present `0` on 45.8% of the corpus, same multiply-to-zero effect). Patched and
> verified fixed on the live index. But a full 53-query Instant-mode eval
> (`evals/retrieval_cases.json`) run with the *patched* formula still active passed only 21/53,
> versus 42/53 with `function_score` skipped entirely (plain BM25 text relevance, no boost).
> The multiplicative `documenttypeboost x court_boost x landmarkruling` stack still routinely
> outweighs real query-text relevance by 10-50x for docs with strong boost values but a weaker
> text match, even fully patched — an architecture problem (`boost_mode: "multiply"` itself),
> not another instance of the missing-data class of bug. `raw_search` now sends the field query
> directly (still bool/should, still query-shape-aware per section 3), skipping the
> `function_score` wrapper; `_wrap_function_score` is kept in `es_client.py`, unused, as a
> record of the formula and this finding. The `landmarkruling: -10` blacklist exclusion (a
> content filter, not a ranking signal) was kept independently via a separate `must_not` wrapper
> so disabling the boost didn't silently let blacklisted docs back into results.
> Re-enabling ranking boost requires redesigning the combination (bounded/additive instead of
> multiplicative), not just flipping section 2 back on — left as a follow-up, not done here.

## Motivation

Instant mode is meant to be a fast raw ES+Milvus preview (`< 1s`, no LLM calls). An audit of
the live ES index (`researchindex_aic_test`, 410,427 docs — confirmed by the user to be the
real collection, not a stale test snapshot) found:

1. **Field coverage bug.** `raw_search`'s `_RAW_SEARCH_FIELDS = ["facts_text", "held_text",
   "headnotes_text"]` (`es_client.py:6-8`) searches fields populated on only 25.7% / 26.6% /
   58.1% of documents respectively, while `heading`, `subheading`, and `fullcontent` — all
   100% populated — are never searched. For roughly 3 in 4 documents, Instant mode's query
   runs against empty fields.
2. **No ranking signal beyond ES `_score`.** No `function_score` exists anywhere in
   `es_client.py`. Two real, populated boost fields sit unused: `court_boost` (99.9%
   populated, range 0–294, avg 112) and `documenttypeboost` (100% populated, range 0–10,000,
   avg 4,869). `landmarkruling` (range -10 to 20, only 2.1% populated, `-10` marks an
   excluded/blacklisted doc — same convention as the sibling `centax-node` codebase's
   `services/searchText.js` aging function) is sparse but cheap to include.
3. **No query-shape awareness.** Every query — a citation, a section reference, or a plain
   legal question — gets the same flat `multi_match` with equal field weight and
   `fuzziness: AUTO`. `centax-node` (the sibling Node service covering the same corpus)
   handles this via a token-classification dictionary (`constants/token.js`, ~3,300 curated
   rows) driving per-field boost/slop.

Confirmed unusable on this index (do not build on these): `searchboosttext` (0% populated —
centax's star composite-metadata field doesn't exist here), `boostpopularity` (0%),
`masterinfo.info.{court,act,section,bench}.*` (0% across all 410k docs, all content groups —
the AI Mode filter fields target dead data; out of scope for this spec but flagged as a known
bug for the follow-up AI Mode spec), `incometaxactinfo`/`companyactinfo`/`tariffinfo`
structured fields (0% populated on this index).

## Non-goals

- No LLM/model calls anywhere in this path — must stay well under 1s.
- No new filter/facet behavior in Instant mode (ranking-only for now; filters are a later,
  separate step, per user decision).
- No changes to AI Mode's `intent.py`, `filter_resolve.py`, or Milvus retrieval — separate
  spec.
- No per-request Milvus/ES fusion — Instant mode keeps running ES and Milvus in parallel,
  independently surfaced (hard rule, unchanged).

## Design

### 1. Field fix

Replace the flat 3-field `multi_match` with a `bool.should` multi-field query:

- `heading`, `subheading` — highest boost (100% populated, most reliable signal).
- `fullcontent` — baseline boost (100% populated, full-text recall).
- `facts_text`, `held_text`, `headnotes_text` — moderate boost, kept as-is: still a strong
  signal when present (58% of docs for `headnotes_text`), just no longer the only fields
  queried.

Exact boost numbers are a tuning detail settled during implementation/eval, not fixed here —
the structural fix (search the always-populated fields at all) is what this spec locks in.

### 2. Ranking fix — `function_score`

Wrap the field query in a `function_score`, reusing centax's already-tuned formula for the
one field where our index's value range matches centax's exactly:

- `field_value_factor(documenttypeboost, factor=0.2, modifier="sqrt", missing=0.0001)` —
  centax's exact constants (`services/searchText.js:718-896`); same field, same use, ports
  directly since the value range (0–10,000) matches what centax tunes against.
- `field_value_factor(court_boost, factor=0.01, modifier="none")` — new; sized to this
  field's smaller range (max contribution ≈3) since centax has no direct analog for this
  field name.
- `field_value_factor(landmarkruling, factor=1.2, modifier="log2p")` (centax's exact
  constants) plus a `must_not: {term: {landmarkruling: -10}}` filter to exclude blacklisted
  docs, matching centax's own blacklist convention on the same field.
- `boost_mode: "multiply"` (ES default, matches centax — base text relevance × the above).

### 3. Query-shape-aware boosting (heuristic, no LLM)

A three-layer system, ported conceptually from `centax-node`'s `queryAnalyzer.js` +
`token.js`, rebuilt with a cleaner architecture — same domain knowledge, not the same code:

**Layer A — data (known terms).** A typed lexicon (not centax's stringly-encoded
`Params: "tag;slop;boost;group"` format) holding: known Acts, known courts, known journal
abbreviations, common legal stopwords/synonyms. Populated from `centax-node`'s
`constants/token.js` (~3,300 rows — port the *values*, not the file format or its two
undefined `ElementType`/`TokenType` enums, which are referenced but never defined anywhere in
that repo). Where the real ES index has ground-truth values (`groups.group.name`:
CASELAWS/ACT/RULE/COMMENTARY/Experts Opinion/Tariff; `categories.name`: 16 subject areas),
those are used directly instead of guessed. Where the index has no data (act/section names,
confirmed 0% populated), the `token.js`-derived values are the only source and are used as-is.

**Layer B — structural regex (grammar, not data).** Fixed patterns that don't come from any
dataset: section/rule/article number format (`Section 54F`, `Rule 6(3)`, `u/s 80C`), citation
format (`2024 ITR 123`, `(2023) 5 SCC 100`, `AIR 2022 SC 456`), party-name pattern (`X vs Y`,
`X v. Y`).

**Layer C — tokenizer procedure.** Ported behavior (not code) from `queryAnalyzer.js`,
rewritten as a documented Python function per rule — each function's docstring states which
centax behavior it replaces:
- Keyword+number merge (`"Section" + "6"` → `"Section 6"`), backtrack if the following token
  isn't a number.
- Court+city merge with backtrack (`"Delhi" + "High Court"` → `"Delhi High Court"`; back off
  to treating them as separate tokens if no match).
- Journal-never-discard: a recognized journal token is always kept even without a full
  citation around it.
- Stopword/synonym skip (multi-word stopwords like "in case of" removed; single stopwords
  left alone, matching centax's asymmetry).
- Quoted-phrase extraction (`"Income India"` kept as one token).
- Citation-spacing normalization (`2024taxman.com` → `2024 taxman.com`).

**Output → boost profile.** The tokenizer's classification feeds one of three boost profiles
into the field query from Section 1: **citation-style** (favor `heading`/phrase-match, tight
slop), **provision-style** (favor `subheading`/`headnotes_text`, section-number normalized
before matching), **plain-text** (current flat behavior, no special-casing). No filtering,
no routing — ranking only, per the Non-goals section.

### Module layout

- `packages/common/src/common/legal_lexicon.py` — Layer A data (loaded once at import time,
  in-memory dict/set lookups) + Layer B regex constants. Single source of truth; the later
  AI Mode spec will import from here too instead of duplicating (`schema_context.py`'s
  `KNOWN_ACTS`/`KNOWN_COURTS` and `intent.py`'s `_LEGAL_MARKERS` currently overlap without
  being the same object — this module is where that duplication eventually resolves, though
  fixing the existing duplication is out of scope for this spec).
- `packages/common/src/common/query_tokenizer.py` — Layer C procedural rules, one function
  per rule, each documented with which `queryAnalyzer.js` behavior it replaces.
- `packages/common/src/common/es_client.py` — `raw_search` updated to: run the tokenizer,
  pick a boost profile, build the multi-field `bool.should` query, wrap in `function_score`.

### Data extraction

A one-time (rerunnable) script reads `centax-node`'s `constants/token.js`, extracts
`{KeyWordExact, TypeCode, ZoneType, SearchText}` per row (dropping the `Params` field
entirely — its slop/boost/group values don't transfer meaningfully to this system's own
`function_score`-based boosting), and writes a typed JSON/YAML lexicon file. Script and
source file live in this repo; not a runtime dependency on `centax-node`.

## Testing

- Unit tests (fake ES client, existing pattern in this repo) on: field-list construction,
  `function_score` structure/constants, each tokenizer rule in isolation (merge, backtrack,
  journal-never-discard, stopword skip, quoted-phrase, citation-spacing), boost-profile
  selection for citation/provision/plain-text sample queries.
- Before/after comparison: run a fixed set of real queries against the live index, diff
  result ordering, sanity-check no regression (informal, not a blocking CI gate — same
  posture as this repo's existing `retrieval_eval.py`/`intent_eval.py` harnesses).

## Open questions for implementation time

- Exact boost weights for Section 1's field list (starting point vs tuned via eval).
- Whether the `token.js` port keeps all ~3,300 rows or filters to a smaller high-confidence
  subset (e.g. drop rows with empty `SearchText` that add no value once `Params` is dropped).
