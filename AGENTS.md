# AGENTS.md

Instructions for coding agents working in this repo.

## Repo

`retrieval-system` — Taxmann caselaw retrieval service. uv workspace, Python 3.11, 3 packages: `common`, `model-gateway`, `retrieval-api`. Docker-compose stack. See `README.md` for architecture/setup, `docs/superpowers/specs/2026-08-03-retrieval-system-design.md` for full design.

## Hard rules

1. `query_embed` role must resolve to Voyage, never DeepInfra or another provider — the Milvus corpus was embedded with Voyage, mismatched embeddings silently corrupt search (no error, just wrong results). See `model-gateway/src/model_gateway/config.py::build_role_provider_map`.
2. Milvus `sparse_vector` is server-computed BM25, never set client-side. Sparse queries pass `anns_field="sparse_vector"`, `data=[<raw text>]` — see `common/src/common/milvus_client.py`.
3. No ranking fusion between ES and Milvus — `doc_id` is join-only.
4. AI Mode queries all 7 Milvus collections every time, no intent-based routing.
5. Python 3.11 only (`pymilvus`'s `grpcio` has no 3.14 wheel).

## Gotchas already hit once — don't repeat

- pydantic-settings matches env vars to uppercased field names, no prefix by default. If `.env.example` has `DEEPINFRA_CHAT_MODEL_SLM` the field must be named `deepinfra_chat_model_slm`, not `chat_model_slm`.
- `from module import name` + `monkeypatch.setattr(module, "name", fake)` does NOT intercept calls made via the direct-imported name in a different module. Use `import module` and call `module.name(...)` at any call site you need to be mockable from outside.
- Module-scope config construction (e.g. building a role→model map at import time) breaks test collection if required settings aren't in the environment. Check `packages/model-gateway/tests/conftest.py` for the pattern used to work around this.

## Commands

```bash
uv sync --all-packages                               # NOT bare `uv sync` - drops editable installs of workspace members
uv run pytest                                        # aggregates all 3 packages from repo root
docker compose up -d --build                         # run the stack
```

## Conventions

- Import names: `common`, `model_gateway`, `retrieval_api` (underscore). Distribution/package names in pyproject.toml: `common`, `model-gateway`, `retrieval-api` (dash for the latter two).
- TDD throughout: every module has a paired test file, written failing-first.
- `common/schemas.py`'s collection/chunking facts were verified against `data-extraction-pipeline`'s actual source code, not its docs (some of which are stale). Don't re-derive these from that repo's markdown.
