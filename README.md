# retrieval-system

Taxmann legal/tax caselaw retrieval service. Two query paths, one query, one WebSocket:

- **Instant** — fast raw ES + Milvus preview, ~300-800ms, no rerank.
- **AI Mode** — SLM query rewrite → Milvus RRF retrieval → cross-encoder rerank + ES citation prefetch → LLM synthesis. Runs in background, streams after Instant.

Reads from the Milvus (`aic` DB, 7 collections) and Elasticsearch stores populated by the separate `data-extraction-pipeline` repo. No code dependency on that repo — own ES/Milvus client code here.

## Architecture

Two services, docker-compose:

- **`model-gateway`** (`:8001`) — the only seam that knows about LLM/embedding/rerank providers. Routes: `/v1/chat`, `/v1/embed`, `/v1/rerank`, each keyed by a `role`. DeepInfra backs `slm`/`synthesis`/`reranker`. **`query_embed` is Voyage-only** — the Milvus corpus was embedded with Voyage by the ingestion pipeline, so query embeddings must land in the same vector space. Swapping that provider is not a config change, it means re-embedding the whole corpus.
- **`retrieval-api`** (`:8000`) — FastAPI + WebSocket app. `/ws/search` dispatches Instant and AI Mode concurrently, sends Instant's result the moment it resolves, then AI Mode's `ai_mode_done`/`ai_mode_error`.

Full design: [`docs/superpowers/specs/2026-08-03-retrieval-system-design.md`](docs/superpowers/specs/2026-08-03-retrieval-system-design.md)
Build plan: [`docs/superpowers/plans/2026-08-03-retrieval-system.md`](docs/superpowers/plans/2026-08-03-retrieval-system.md)

## Packages

```
packages/
  common/         # config, Milvus/ES client wrappers, schema constants
  model-gateway/  # FastAPI: role -> provider/model routing (DeepInfra + Voyage adapters)
  retrieval-api/  # FastAPI + WebSocket: Instant + AI Mode orchestration
```

## Setup

```bash
uv sync
cp .env.example .env   # fill in MILVUS_*, ES_URL, DEEPINFRA_API_KEY, VOYAGE_API_KEY
```

Run tests (per package):

```bash
cd packages/common && uv run pytest
cd packages/model-gateway && uv run pytest
cd packages/retrieval-api && uv run pytest
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

## Known follow-ups (not blocking, tracked in the plan's ledger)

- `dense_vector_2` (second embedder) parity — v1 queries Voyage's `dense_vector` only.
- `model-gateway` provider #2 beyond DeepInfra — adapter interface ready, not written.
- No logging on caught exceptions in the AI Mode pipeline (observability gap).
- ES/Milvus client construction in `ws.py` isn't fully inside the cleanup `try/finally` (narrow leak only if a constructor itself throws).
