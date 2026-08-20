# Instant Mode query classifier: ML-based ES/Milvus routing

Date: 2026-08-20
Status: proposed

## Problem

Instant Mode always queries ES (sparse/BM25), Milvus dense, and Milvus sparse
unconditionally (`instant/search.py::run_instant`) and only optionally fuses them via a
manual `rrf` toggle. There is no per-query decision about which backend(s) are actually
relevant to the query's shape — a pure citation lookup like `"Section 52"` pays the cost
of a semantic Milvus search it doesn't need, and a conversational query like `"how do I
evade tax"` gets no benefit from that dense search unless a human flips the toggle.

A separate, existing rule-based classifier, `query_tokenizer.classify_query_shape()`
(`citation | provision | plain`), already exists but is used only to pick ES field-boost
weights (and optionally RRF weights when `rrf=True`) — never to decide whether to skip a
backend entirely.

This spec adds a small supervised ML classifier — CPU-only, sub-5ms, no LLM/SLM call —
that decides, per query, which backend(s) Instant Mode should actually search, and
retires the rule-based shape classifier by having the new model's output drive both jobs
it previously split across two mechanisms.

## Goals

- A 3-class classifier — `KEYWORD | HYBRID | INTENT` — that decides backend routing:
  `KEYWORD` → ES only, `INTENT` → Milvus only, `HYBRID` → both + RRF fusion. This replaces
  the manual `rrf` toggle as Instant Mode's default routing behavior.
- Reuse the existing shared lexicon/tokenizer substrate
  (`common/legal_lexicon.py`, `common/query_tokenizer.py` — courts, journals, synonyms,
  normalizations, stopwords, citation/section/party regexes) as raw feature inputs. No new
  gazetteer or synonym data gets built; it already exists and is already shared with AI
  Mode's `extract_intent()`.
- Retire `classify_query_shape()` entirely. The new model's label also replaces its role
  in `es_client.py`'s `_BOOST_PROFILES` selection and `rerank.py`'s `_SHAPE_RRF_WEIGHTS` —
  one classification step drives both ES boost-profile selection and backend routing,
  instead of two separate mechanisms that happened to share input primitives.
- Confidence-gated fallback: below a tuned `predict_proba` threshold, fall back to today's
  always-both, no-fusion behavior rather than trust a low-confidence routing call.
- Feature extractors are written as generic, taxonomy-agnostic transformers (text in,
  named vector out) rather than hardcoded to this 3-class label set — an extensibility
  seam for a possible future non-LLM classifier serving AI Mode's category taxonomy, which
  is explicitly out of scope for this spec (see Non-goals).
- Fold two small, already-flagged-as-duplicate lexicon lists —
  `schema_context.KNOWN_COURTS` (9 entries) and `intent._LEGAL_MARKERS` (10 entries) — into
  the shared `legal_lexicon.json`, retiring the hand-maintained copies.

## Non-goals

- Not building a category/intent classifier for AI Mode (`acts/rules/caselaws/articles/
  commentary/tariff`) or removing any SLM call from `extract_intent()`. That taxonomy is
  SLM-driven for reasons (scenario/fact-pattern recognition with zero literal anchor,
  per `docs/superpowers/specs/2026-08-18-intent-lexicon-signal-and-vague-floor-design.md`)
  that a lexical-only classifier cannot replicate. If this model proves accurate and fast
  enough in production, a follow-on spec can evaluate reusing its feature-extraction layer
  for an AI-Mode-facing model — not attempted here.
- Not building new gazetteer/synonym/normalization data — `legal_lexicon.json` already has
  courts (148), journals (22), synonyms (1101), normalizations (1299), stopwords (121).
- Not changing `common/query_tokenizer.py`'s normalization pipeline itself
  (`normalize_citation_spacing`, `expand_query_synonyms`, `chunk_query`) — reused as-is.
- Not adding automated CI-triggered retraining or a model-serving microservice. Retraining
  is a manual, human-reviewed step; inference is in-process, not a network call.
- Not deciding whether a trailing `"?"` is noise or signal — resolved by making it a model
  feature (`has_trailing_question_mark`, computed post-normalization) rather than a hard
  rule, letting training data settle it.

## Architecture

```
query
  │
  ▼
query_tokenizer normalization (unchanged: normalize_citation_spacing, lexicon
normalize/expand_query_synonyms, structural cleanup)
  │
  ▼
instant_classifier.classify(query) -> (label, confidence)
  │  [FeatureUnion: citation-regex, gazetteer, structural, intent-language, TF-IDF word
  │   + char n-grams]  ->  LogisticRegression(class_weight="balanced")
  │
  ├─ confidence >= threshold ──────────────┐
  │                                        ▼
  │                          label ∈ {KEYWORD, HYBRID, INTENT}
  │                                        │
  │                     ┌──────────────────┼──────────────────┐
  │                     ▼                  ▼                  ▼
  │                 ES only          ES + Milvus dense    Milvus only
  │              (_BOOST_PROFILES     + Milvus sparse      (_BOOST_PROFILES
  │               ["KEYWORD"])        + RRF fusion          ["INTENT"] — boost
  │                                  (_BOOST_PROFILES        profile still
  │                                   ["HYBRID"], RRF        applies if ES is
  │                                   weights by label)      ever also queried)
  │
  └─ confidence < threshold ─────────────────────────────────────────────────┐
                                                                              ▼
                                                     fallback: query all backends,
                                                     no fusion (today's default
                                                     behavior, unchanged)
```

`es_client.py` and `run_instant()` each ask the classifier's `labels` module for their
piece of the mapping (boost-profile key, routing decision) rather than hardcoding it
independently — a query's label is computed once per request and consumed twice.

## Module layout

New submodule inside `packages/common` (no new package — this repo is trending toward
fewer, lighter-weight packages, and this belongs next to the lexicon/tokenizer/ES-client
code it extends):

```
packages/common/src/common/instant_classifier/
    __init__.py     # public API: classify(query: str) -> ClassifierResult(label, confidence)
    labels.py        # KEYWORD/HYBRID/INTENT + label -> (boost_profile_key, routing) mapping
    features.py       # 5 feature-extractor groups, each a standalone, taxonomy-agnostic
                        # transformer: citation-regex, gazetteer, structural,
                        # intent-language, TF-IDF wiring
    pipeline.py         # FeatureUnion + LogisticRegression assembly; load/save artifact
packages/common/src/common/data/
    instant_classifier_model.joblib        # trained artifact, committed to git
    instant_classifier_model_meta.json      # version, training-data hash, confidence
                                              # threshold, per-class eval metrics
packages/common/data/instant_classifier/    # training data, not shipped in the package build
    train.jsonl        # grows over time: query_text, label, source, date_added
    eval_frozen.jsonl    # 150-300 examples, locked once created, never trained on
packages/common/scripts/
    train_instant_classifier.py   # offline: reads train.jsonl, fits pipeline, sweeps
                                    # predict_proba threshold against eval_frozen.jsonl,
                                    # writes both artifact files
```

`features.py` transformers take normalized text and return a named vector regardless of
what label set consumes them — they don't know about `KEYWORD/HYBRID/INTENT`. `labels.py`
is the only file that knows the taxonomy and what each label maps to downstream.

## Feature groups

| Group | What it computes | Source |
|---|---|---|
| Citation regex | binary/count hits | `legal_lexicon.SECTION_PATTERN`, `CITATION_PATTERN`, `PARTY_PATTERN` |
| Gazetteer | `has_court_mention`, `has_journal_mention`, `has_legal_term_mention` (binary per category) | `legal_lexicon.is_known_court`, `is_known_journal`, `synonyms` keys |
| Structural | token count, `has_trailing_question_mark`, has quotes | computed post-normalization |
| Intent language | question words, first-person, modals, conditionals | new small keyword list — general English function words, not legal-domain, so not part of `legal_lexicon.json` |
| TF-IDF word + char n-grams | phrasing style, citation typos/variants (`"Section 138"` / `"s.138"`) | `sklearn.TfidfVectorizer`, fit at training time |

No feature hard-overrides the label (unchanged PRD principle) — every signal is a
weighted input the `LogisticRegression` learns to balance. The old
`classify_query_shape()`'s regex/lexicon primitives are reused as raw feature inputs here,
not as a pre-computed shortcut feature — the new model learns its own weighting rather
than inheriting the old function's hardcoded rule thresholds.

## Training data & model lifecycle

- **Source:** a sample of real historical Instant Mode queries, manually labeled against
  the taxonomy's calibration anchors (`"Section 52"` → KEYWORD, `"Where is Section 52
  applicable"` → HYBRID, `"How do I evade tax"` → INTENT). Deliberate oversampling toward
  rarer patterns for HYBRID, since it's the smallest class in raw traffic too.
  Minimum starting size ~500-1000 examples; HYBRID kept above ~100-150.
- **Frozen eval set:** `eval_frozen.jsonl`, 150-300 examples, never trained on — gates
  every retrain for regressions before promotion.
- **Retrain cadence:** manual — edit `train.jsonl`, rerun `train_instant_classifier.py`,
  review the eval metrics it prints, commit the new artifact + meta file. No CI-triggered
  auto-retrain.
- **Confidence threshold:** chosen by sweeping `predict_proba` cutoffs against
  `eval_frozen.jsonl` during training, stored in `instant_classifier_model_meta.json`, read
  at startup — not hardcoded in application code.
- **Artifact size:** expected a few hundred KB (TF-IDF vocab + logistic weights) — small
  enough to commit directly to git, no external artifact storage needed.
- **Loading:** `retrieval-api` loads the joblib artifact once at process startup, same
  pattern as any other startup-time resource. No per-request disk I/O, no network call —
  required to hit the <5ms classification budget.

## Error handling

- **Confidence below threshold:** fall back to today's behavior — query all backends, no
  fusion, return unmerged per-source results.
- **Model artifact missing or fails to load at startup:** fail loud — `retrieval-api`
  refuses to start. A silent fallback here would masquerade a deployment bug as "the model
  is just always low-confidence" and never surface as an incident.
- **Empty/malformed query text:** short-circuits before the model runs, same fallback path
  — never a model prediction on garbage input.
- **Feature-extraction or prediction error at request time** (e.g. a TF-IDF edge case):
  caught at the call site in `run_instant()`, falls back to always-both. A classifier bug
  degrades Instant Mode to its current behavior; it never takes the request down.

## Lexicon cleanup (bundled into this pass)

`schema_context.KNOWN_COURTS` (9 hardcoded entries, AI-mode LLM prompt context) and
`intent._LEGAL_MARKERS` (10 hardcoded Act names, AI-mode rewrite-safety checks) were
already flagged in `docs/superpowers/specs/2026-08-11-instant-mode-es-retrieval-redesign-design.md`
as divergent duplicates of `legal_lexicon.json` that were never unified. Since this work
touches the same lexicon file, fold both into `legal_lexicon.json` (extending `courts` and
adding the Act names to `synonyms` or a new small `acts` list as appropriate) and update
their two call sites to read from `legal_lexicon` instead of a local hardcoded list.

## Testing

- `packages/common/tests/test_instant_classifier.py`:
  - Unit tests per feature extractor group (regex hits on known citation shapes,
    gazetteer hits on known/unknown terms, structural features computed post-normalization).
  - Pipeline-level tests on the three PRD calibration anchors.
  - Confidence-threshold fallback test: mock a low-confidence prediction, assert routing
    falls back to always-both.
  - `eval_frozen.jsonl` accuracy check as an explicit test, not just a training-time
    script — fails CI if a committed model artifact regresses below a floor accuracy,
    catching a bad retrain-and-commit before it ships.
- Update `es_client.py` boost-profile tests and `instant/search.py`/`rerank.py` routing
  tests to the new `KEYWORD/HYBRID/INTENT` taxonomy — `classify_query_shape()`'s
  `citation/provision/plain` tests are retired along with the function.
  `_BOOST_PROFILES` and `_SHAPE_RRF_WEIGHTS` keys get renamed accordingly.
- New tests for the `schema_context`/`intent.py` cleanup: assert both call sites now read
  from `legal_lexicon` and produce the same (or intentionally corrected) court/Act
  coverage as before.
