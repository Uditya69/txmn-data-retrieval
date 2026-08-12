# CLAUDE.md

Guidance for Claude Code sessions in this repo.

## What this is

Retrieval service for Taxmann caselaw. Three paths, one query: Instant (raw ES+Milvus preview), AI Mode (SLM rewrite → RRF → rerank → LLM synthesis), and Agentic search (an LLM tool-calling agent over the same ES/Milvus tools, with citation validation, served at `/ws/agent` and the `packages/agents` package). Full design: `docs/superpowers/specs/2026-08-03-retrieval-system-design.md`. Full build plan (17 tasks, TDD, each with brief/report): `docs/superpowers/plans/2026-08-03-retrieval-system.md`.

Standalone repo. No code dependency on `data-extraction-pipeline` (the sibling repo that populates Milvus/ES) — own client code here, kept in sync by hand with its `schemas/Milvus.json`/`schemas/ES.json`.

## Hard rules — do not violate

1. **`query_embed` role goes through Voyage, never DeepInfra or any other provider.** The Milvus corpus's `dense_vector` was embedded with Voyage by the ingestion pipeline. A different embed model produces vectors in a different space — cosine similarity against the corpus silently becomes meaningless, no error thrown. This is wired in `model-gateway`'s `ROLE_PROVIDER_MAP` (`config.py`) — don't "simplify" it to one provider.
2. **Milvus's `sparse_vector` is never set by client code.** It's computed server-side by a native BM25 `Function` at insert time. To *query* it, search with `anns_field="sparse_vector"` and `data=[<raw query text>]` — pass the text string directly, Milvus converts it. See `common/milvus_client.py::_search_one`. Passing a computed vector or `None` here is wrong.
3. **No raw-score ranking fusion between ES and Milvus.** `doc_id` is join-only (citation lookup, filter allowlist) by default. Don't blend ES's lexical score and Milvus's cosine/BM25-distance score directly — they're on incomparable scales. The one sanctioned exception: Instant mode's opt-in `rerank` toggle (`instant/rerank.py::rrf_merge_by_doc_id`) fuses ES + Milvus dense + Milvus sparse by *rank position* via RRF, not raw score — rank-based fusion sidesteps the incomparable-scale problem this rule exists to prevent. Don't extend raw-score blending elsewhere off the back of this exception.
4. **AI Mode searches all 7 Milvus collections every query.** No intent-based collection routing.
5. **Python 3.11, not 3.14** — `pymilvus`'s `grpcio` has no prebuilt wheel for 3.14.

## Known gotchas hit during the build (avoid repeating)

- **pydantic-settings env var matching**: a field named `chat_model_slm` looks for env var `CHAT_MODEL_SLM`, not `DEEPINFRA_CHAT_MODEL_SLM`, unless you rename the field to match. Always check `.env.example` names against the actual `Settings`/`GatewaySettings` field names — a mismatch fails validation at real runtime but tests won't catch it unless they set the exact same env vars.
- **Monkeypatch + direct imports don't mix.** `from module_a import foo` binds a name into the importing module's own namespace; `monkeypatch.setattr(module_a, "foo", fake)` only changes `module_a`'s attribute, not the already-bound reference in the importer. If you need a call site to be mockable, `import module_a` and call `module_a.foo(...)`, or patch the *consuming* module's namespace directly.
- **Test collection can fail at import time** if a module builds config at module scope (e.g. `ROLE_MODEL_MAP = build_role_model_map(get_gateway_settings())` runs when the module is imported). If `GatewaySettings` has required fields and no `.env` exists in the checkout, tests fail before they even run. `packages/model-gateway/tests/conftest.py` sets dummy env vars for this — extend it if you add new required settings fields, or the whole test file breaks on import.
- Package naming: `common`, `model_gateway`, `retrieval_api` (underscore) as Python import names; `model-gateway`, `retrieval-api` (dash) as uv package/distribution names in `pyproject.toml`. Don't mix them up.
- Every collection/field name mentioned in `common/schemas.py` (chunked-vs-not, bm25_source) was verified against the actual `data-extraction-pipeline` source code, not its docs — some of that repo's own markdown docs are stale (describe an older word-based, ruling-only chunking scheme). Trust the code-verified facts recorded in the design spec over anything you read in that repo's docs.

## Running things

`uv sync --all-packages`, NOT bare `uv sync` — the latter can drop editable installs of workspace members (`common`, `model-gateway`, `retrieval-api`, `agents`), breaking local test collection.

`uv run pytest` from repo root aggregates all 4 packages (143 tests). Scope to one package with `uv run pytest packages/<name>/tests` if needed.

`docker compose build` / `docker compose up -d --build` from repo root.

## If you're continuing implementation work

This repo was built with `superpowers:subagent-driven-development`. Progress ledger, task briefs, reports, and review diffs live in `.superpowers/sdd/2026-08-03-retrieval-system/` (gitignored — local only, not the source of truth once merged; git history is). If resuming mid-plan, read the ledger's `progress.md` first.
