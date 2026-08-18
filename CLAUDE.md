# CLAUDE.md

Guidance for Claude Code sessions in this repo.

## What this is

Retrieval service for Taxmann caselaw. Three paths, one query: Instant (raw ES+Milvus preview), AI Mode (SLM rewrite → RRF → rerank → LLM synthesis), and Agentic search (an LLM tool-calling agent over the same ES/Milvus tools, with citation validation, served at `/ws/agent` and the `packages/agents` package). Full design: `docs/superpowers/specs/2026-08-03-retrieval-system-design.md`. Full build plan (17 tasks, TDD, each with brief/report): `docs/superpowers/plans/2026-08-03-retrieval-system.md`.

Standalone repo. No code dependency on `data-extraction-pipeline` (the sibling repo that populates Milvus/ES) — own client code here, kept in sync by hand with its `schemas/Milvus.json`/`schemas/ES.json`.

## Hard rules — do not violate

1. **`query_embed` role goes through Voyage, never DeepInfra or any other provider.** The Milvus corpus's `dense_vector` was embedded with Voyage by the ingestion pipeline. A different embed model produces vectors in a different space — cosine similarity against the corpus silently becomes meaningless, no error thrown. This is wired in `model-gateway`'s `ROLE_PROVIDER_MAP` (`config.py`) — don't "simplify" it to one provider.
2. **Milvus's `sparse_vector` is never set by client code.** It's computed server-side by a native BM25 `Function` at insert time. To *query* it, search with `anns_field="sparse_vector"` and `data=[<raw query text>]` — pass the text string directly, Milvus converts it. See `common/milvus_client.py::_search_one`. Passing a computed vector or `None` here is wrong.
3. **No raw-score ranking fusion between ES and Milvus.** `doc_id` is join-only (citation lookup, filter allowlist) by default. Don't blend ES's lexical score and Milvus's cosine/BM25-distance score directly — they're on incomparable scales. Two sanctioned exceptions, both rank-based, never raw-score:
   - Instant mode's opt-in `rerank` toggle (`instant/rerank.py::rrf_merge_by_doc_id`) fuses ES + Milvus dense + Milvus sparse by *rank position* via RRF.
   - AI Mode's ES sparse-fallback (`ai_mode/retrieve.py::_flatten()`, for `ruling`/`act_section`/`rule_section`/`article_section`/`commentary_section` — the collections whose Milvus `sparse_vector` was dropped) ranks Milvus-native and ES-origin sparse hits *locally within their own source*, then interleaves by rank position to build the list `rrf_merge()` fuses — raw scores from the two sources are never compared. See `docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md`.

   Don't extend raw-score blending elsewhere off the back of either exception.
4. **AI Mode routes which Milvus collections get searched by the `intent` category tag**
   (`extract_intent()`'s multi-label `acts`/`rules`/`caselaws`/`articles`/`commentary`/`tariff`
   classification), via `collections_for_intent()` (`common/schemas.py`). Empty or
   unrecognized-only `intent` falls back to searching all 11 collections. `tariff_section` has
   no routing entry yet — not live (`_disabled_collections` upstream). RRF fusion weight stays
   neutral (1.0/1.0) regardless of category — this was considered and explicitly rejected during
   design; don't reintroduce category-based dense/sparse weighting. See
   `docs/superpowers/specs/2026-08-13-intent-category-classification-design.md` and
   `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md`.
5. **Python 3.11, not 3.14** — `pymilvus`'s `grpcio` has no prebuilt wheel for 3.14.

## Known gotchas hit during the build (avoid repeating)

- **pydantic-settings env var matching**: a field named `chat_model_slm` looks for env var `CHAT_MODEL_SLM`, not `DEEPINFRA_CHAT_MODEL_SLM`, unless you rename the field to match. Always check `.env.example` names against the actual `Settings`/`GatewaySettings` field names — a mismatch fails validation at real runtime but tests won't catch it unless they set the exact same env vars.
- **Monkeypatch + direct imports don't mix.** `from module_a import foo` binds a name into the importing module's own namespace; `monkeypatch.setattr(module_a, "foo", fake)` only changes `module_a`'s attribute, not the already-bound reference in the importer. If you need a call site to be mockable, `import module_a` and call `module_a.foo(...)`, or patch the *consuming* module's namespace directly.
- **Test collection can fail at import time** if a module builds config at module scope (e.g. `ROLE_MODEL_MAP = build_role_model_map(get_gateway_settings())` runs when the module is imported). If `GatewaySettings` has required fields and no `.env` exists in the checkout, tests fail before they even run. `packages/model-gateway/tests/conftest.py` sets dummy env vars for this — extend it if you add new required settings fields, or the whole test file breaks on import.
- Package naming: `common`, `model_gateway`, `retrieval_api` (underscore) as Python import names; `model-gateway`, `retrieval-api` (dash) as uv package/distribution names in `pyproject.toml`. Don't mix them up.
- Every collection/field name mentioned in `common/schemas.py` (chunked-vs-not, bm25_source) was verified against the actual `data-extraction-pipeline` source code, not its docs — some of that repo's own markdown docs are stale (describe an older word-based, ruling-only chunking scheme). Trust the code-verified facts recorded in the design spec over anything you read in that repo's docs.
- **ES's `documenttypeboost`/`court_boost`/`landmarkruling` ranking boost is disabled** (`common/es_client.py::raw_search` does not call `_wrap_function_score`). Two missing/zero-value bugs in that formula were found and patched (`landmarkruling` populated on only 2.1% of the corpus, `court_boost` a real `0` on 45.8% of it — `boost_mode: "multiply"` meant either one zeroed the entire relevance score for the affected doc regardless of text match quality). Patched and verified fixed on the live index — but a 53-query eval (`evals/retrieval_cases.json`) with the *patched* formula still active passed only 21/53, versus 42/53 with the boost skipped entirely. The multiplicative boost stack still routinely outweighs real text relevance by 10-50x even fully patched — an architecture problem (`boost_mode: multiply` itself), not another missing-data instance. Don't re-enable `_wrap_function_score` (kept in `es_client.py`, unused, as a record) without redesigning the combination first — see the update note atop `docs/superpowers/specs/2026-08-11-instant-mode-es-retrieval-redesign-design.md`.

## Model selection

Do not use Opus for subagents/reviewers in this repo (drains usage quota faster than the user wants) — default to Sonnet even for tasks that would otherwise call for "the most capable available model" (e.g. final whole-branch reviews). Only escalate beyond Sonnet if the user explicitly asks for it in the moment.

## Running things

`uv sync --all-packages`, NOT bare `uv sync` — the latter can drop editable installs of workspace members (`common`, `model-gateway`, `retrieval-api`, `agents`), breaking local test collection.

`uv run pytest` from repo root aggregates all 4 packages (143 tests). Scope to one package with `uv run pytest packages/<name>/tests` if needed.

`docker compose build` / `docker compose up -d --build` from repo root.

## If you're continuing implementation work

This repo was built with `superpowers:subagent-driven-development`. Progress ledger, task briefs, reports, and review diffs live in `.superpowers/sdd/2026-08-03-retrieval-system/` (gitignored — local only, not the source of truth once merged; git history is). If resuming mid-plan, read the ledger's `progress.md` first.
