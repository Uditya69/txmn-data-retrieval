# repotaxmannapi Exact-Replica Query/Boost Port — Design

## Context

Instant mode's current ES query builder (`common/query_tokenizer.py` + `common/es_client.py`)
was ported from `centax-node` (the older Node.js research backend), using additive
(`boost_mode: sum`) scoring — deliberately, per CLAUDE.md's hard rule 5, because a prior
multiply-mode formula (`_wrap_function_score`) was measured on this repo's own 53-query eval
set at 21/53 pass vs 42/53 with it disabled (a single zero-valued field, e.g. `court_boost=0`
on 45.8% of the corpus, zeroed the entire relevance score under `multiply`).

Investigation on 2026-09-01 found a second, more current production system —
`repotaxmannapi` (a .NET/NEST codebase, read-only access, different remote) — whose query
logic is verified to be the actual source of query shapes seen in real trace comparisons
(e.g. `heading` phrase-boost `155000`, `groupBoost` up to `10000000`, an 8-tier recency
ladder, `viewcount`/`total_score`/`documenttypeboost`/`court_boost`/`landmarkruling` field
boosts, all combined under `boost_mode: Multiply`).

The user has asked for an **exact, byte-identical replica** of `repotaxmannapi`'s query and
scoring logic, including its `Multiply` combination mode — an explicit, informed override of
CLAUDE.md's hard rule (confirmed with the user directly: literal multiply-mode behavior,
not the boost values kept in this repo's existing safe additive design).

## Scope

Two layers, both replicated exactly:

1. **Query-building / tokenization** — `TaxmannQueryAnalizer.cs` (~2000 lines): a state-machine
   tokenizer driven by `TokenParserElastic.resx`, a generated resource dictionary (354KB/917KB)
   of thousands of keyword entries, each keyed by uppercased word, each value encoding
   `tag;proximity;boost;groupID` (`SetPrimaryTag`/`GetResource` in the C# source). Token
   categories: Month, Synonym, KeyWordOnly, Country, KeyWordType2, KeyWordOrStopWord, KeyWord,
   KeyWordHighCourt, KeyWordDated, Zone, NumAlphaZone, Journal, Court, SBTM, MultiWordStopWord,
   StopWord (see `ElementType`/`TokenType`/`ProximityDefault` enums in the source).

2. **Query assembly + scoring** — `SearchTextElastic.cs`'s `GetQuery` (~1300 lines: phrase
   boosts 105000/155000/80000/70000/65000, `SUB`-prefix exclusion logic for sub-section
   collisions, `PH`/`Excus` field-suffix branching) and `GlobalSearchResearch.cs`'s
   `FunctionScore` stack (groupBoost resolution from `iGroupID`, per-edition subgroup boosts,
   8-tier recency ladder `1d/7d/1M/3M/1y/2y/5y/150y` weights `18/15/13/10/8/5/3.5/1.5`,
   `documenttypeboost`/`court_boost`/`total_score`/`viewcount`/`landmarkruling` field_value_factor
   stack, two penalty functions (StateGST-non-caselaws `0.03`, Finance-Act-old-year `0.02`),
   all combined under `BoostMode(FunctionBoostMode.Multiply)`.

## Approach: parallel path behind a flag, isolated branch

- All work happens on a new git branch off `dev` (suggested name:
  `feature/repotaxmannapi-exact-replica`), never committed directly to `dev`, merged only
  after explicit sign-off on the Phase 5 eval comparison.
- New Python modules live alongside the existing ones, not replacing them:
  `common/repotaxmannapi_tokenizer.py`, `common/repotaxmannapi_query_builder.py`.
- Instant mode gains an opt-in flag (`boost_source: "sum" | "repotaxmannapi"`, default `"sum"` —
  unchanged behavior) rather than a silent default-behavior flip, so the existing eval-verified
  path stays live throughout development and comparison.
- Known, accepted risk: this reintroduces the exact multiply-mode failure mode CLAUDE.md
  documents (a single zero-valued field can zero a doc's entire score) — by explicit user
  request, not an oversight. The eval run in Phase 5 will re-measure this repo's own regression
  number under the new path before anyone considers flipping the default.

## Phases

### Phase 1 — Dictionary extraction
Pull `TokenParserElastic.resx`'s keyword table into a Python-loadable JSON
(`packages/common/src/common/data/repotaxmannapi_token_dictionary.json`), one entry per
keyword: `{tag, proximity, boost, group_id}`. Verify against a sample of known entries
(e.g. `RULE`, `SECTION`, a court name) read directly from the `.resx`/`.Designer.cs` pair.
Deliverable: extraction script + generated JSON + a handful of spot-check tests.

### Phase 2 — Tokenizer port
Port `TaxmannQueryAnalizer.cs`'s token-classification state machine to Python
(`repotaxmannapi_tokenizer.py`), backed by the Phase 1 dictionary. TDD: one test per token
category (Month, Synonym, Citation, Notification, Court, Journal, Zone, StopWord,
MultiWordStopWord, etc.), each derived from reading the corresponding C# branch, not guessed.

### Phase 3 — Query-builder port
Port `SearchTextElastic.cs`'s `GetQuery` to Python (`repotaxmannapi_query_builder.py`):
phrase-boost tiers, `SUB`-prefix exclusion, `PH`/`Excus` branching. Output: the same
`bool`/`should`/`match_phrase` structure shape as the real ES query bodies already captured
in this session's trace comparisons.

### Phase 4 — Scoring port
Port `GlobalSearchResearch.cs`'s `FunctionScore` stack: `groupBoost` resolution, edition
subgroup boosts, the real 8-tier recency ladder, the five field_value_factor functions
(`documenttypeboost`/`court_boost`/`total_score`/`viewcount`/`landmarkruling`), the two
penalty functions, combined with `boost_mode: "multiply"` / `score_mode: "multiply"` (ES's
literal equivalent of NEST's `FunctionBoostMode.Multiply` with no explicit `ScoreMode`, which
also defaults to `multiply`).

### Phase 5 — Wire-up + eval
Add the `boost_source` flag to Instant mode's `raw_search`/`_build_field_query` call sites.
Run `evals/retrieval_cases.json` against both paths (existing sum-mode default vs new
repotaxmannapi-multiply path) and report the comparison — pass/fail counts, which specific
queries flip which way — before any decision on making it the default.

## Open questions / risks

- `total_score` field semantics are unknown (populated but empty on docs checked so far) —
  Phase 4 will port the field_value_factor call as-is (matching production literally) even
  without understanding what feeds it, per the "exact replica" mandate.
- The `Comparative` group and `CentralGST`/Circular-Notification special-case boosts
  (`GlobalSearchResearch.cs`) reference taxonomy nodes not yet confirmed to exist in this
  repo's own index in the same shape — Phase 3/4 will verify each id against the live index
  before porting it, same diligence as the edition-boost work already done.
- This is read-only access to `repotaxmannapi` — no ability to compile-check the C# source
  being read from, so ported logic is verified by careful reading + live ES verification of
  ids/fields, not by running the original.
