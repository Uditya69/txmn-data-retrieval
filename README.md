# retrieval-system

Taxmann legal/tax caselaw retrieval service. Three query paths, one query:

- **Instant** — fast raw ES + Milvus preview, ~300-800ms, no rerank. Streamed over `/ws/search`.
- **AI Mode** — SLM query rewrite → Milvus RRF retrieval → cross-encoder rerank + ES citation prefetch → LLM synthesis. Runs in background, streams after Instant, also over `/ws/search`.
- **Agentic search** — an LLM tool-calling agent over the same ES/Milvus search tools, with citation validation and retry. Served at `/agent` in the frontend, streamed over its own `/ws/agent`.

Reads from the Milvus (`aic` DB, 7 collections) and Elasticsearch stores populated by the separate `data-extraction-pipeline` repo. No code dependency on that repo — own ES/Milvus client code here.

## Architecture

Two services, docker-compose:

- **`model-gateway`** (`:8001`) — the only seam that knows about LLM/embedding/rerank providers. Routes: `/v1/chat`, `/v1/embed`, `/v1/rerank`, each keyed by a `role`. DeepInfra backs `slm`/`synthesis`/`reranker`. **`query_embed` is Voyage-only** — the Milvus corpus was embedded with Voyage by the ingestion pipeline, so query embeddings must land in the same vector space. Swapping that provider is not a config change, it means re-embedding the whole corpus.
- **`retrieval-api`** (`:8000`) — FastAPI + WebSocket app. `/ws/search` dispatches Instant and AI Mode concurrently, sends Instant's result the moment it resolves, then AI Mode's `ai_mode_done`/`ai_mode_error`. `/ws/agent` runs the agentic tool-calling loop (`packages/agents`) and streams its trace + final answer.

Full design: [`docs/superpowers/specs/2026-08-03-retrieval-system-design.md`](docs/superpowers/specs/2026-08-03-retrieval-system-design.md)
Build plan: [`docs/superpowers/plans/2026-08-03-retrieval-system.md`](docs/superpowers/plans/2026-08-03-retrieval-system.md)

## Retrieval evaluation

Run the 53 corpus-backed direct/indirect/adversarial queries against ES, Milvus dense,
Milvus sparse, rewritten retrieval, RRF, and the reranker:

```bash
uv run retrieval-eval --gateway-url http://localhost:8001 \
  --langfuse-base-url http://localhost:3030
```

Use `--query Q06`, `--class indirect`, `--class adversarial`, or
`--no-langfuse` for focused/local runs. Every run writes a timestamped result
and its exact dataset snapshot under `.eval-results/`; `latest.json` and
`latest.dataset.json` mirror the newest run. Reproduce an older run with:

```bash
uv run retrieval-eval \
  --dataset .eval-results/20260806T123456Z-retrieval-eval.dataset.json \
  --run-name rerun-old-dataset \
  --gateway-url http://localhost:8001
```

## Packages

```
packages/
  common/         # config, Milvus/ES client wrappers, schema constants
  model-gateway/  # FastAPI: role -> provider/model routing (DeepInfra + Voyage adapters)
  retrieval-api/  # FastAPI + WebSocket: Instant + AI Mode + agentic orchestration
  agents/         # LLM tool-calling agent loop, search tools, citation validation
```

## Setup

```bash
uv sync --all-packages   # NOT bare `uv sync` - drops editable installs of workspace members
cp .env.example .env   # fill in MILVUS_*, ES_URI/ES_USERNAME/ES_PASSWORD/ES_INDEX, DEEPINFRA_API_KEY, VOYAGE_API_KEY
```

Run tests (aggregates all 4 packages from repo root):

```bash
uv run pytest
```

Run the stack:

```bash
docker compose up -d --build
```

`model-gateway` on `http://localhost:8001`, `retrieval-api` on `http://localhost:8000`. `retrieval-api` reaches `model-gateway` by service name (`GATEWAY_URL=http://model-gateway:8001`), overridden in `docker-compose.yml` regardless of `.env`.

## Using `/ws/search`

```json
// client sends
{"query": "your search text"}

// server sends, in order
{"type": "instant_result", "es": [...], "es_error": null, "milvus": {...}, "milvus_error": null}
{"type": "ai_mode_done", "answer": "...", "citations": {...}}
// or on AI Mode failure (Instant result already delivered, untouched):
{"type": "ai_mode_error", "error": "..."}
```

## Using `/ws/agent`

```json
// client sends
{"query": "your search text"}

// server sends zero or more trace steps, then exactly one terminal message
{"type": "ai_mode_trace", "step": "agent_tool_call", "data": {...}}
{"type": "ai_mode_trace", "step": "agent_tool_result", "data": {...}}
{"type": "agent_done", "answer": "...", "doc_ids": [...]}
// or, if citations couldn't be verified after retries:
{"type": "agent_unverifiable", "invalid_doc_ids": [...]}
// or on an unhandled pipeline error:
{"type": "agent_error", "error": "..."}
```

## Known follow-ups (not blocking, tracked in the plan's ledger)

- `dense_vector_2` (second embedder) parity — v1 queries Voyage's `dense_vector` only.
- `model-gateway` provider #2 beyond DeepInfra — adapter interface ready, not written.
- No logging on caught exceptions in the AI Mode pipeline (observability gap).
- ES/Milvus client construction in `ws.py` isn't fully inside the cleanup `try/finally` (narrow leak only if a constructor itself throws).
