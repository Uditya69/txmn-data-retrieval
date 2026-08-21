#!/usr/bin/env bash
# One-command local dev: model-gateway + retrieval-api (uvicorn --reload) + web (vite dev server).
# Hits the remote Milvus/ES from .env directly - no docker, no rebuild needed on code changes.
set -euo pipefail
cd "$(dirname "$0")/.."

# Ports match docker-compose's host mappings so the git-tracked
# packages/web/public/env-config.js (WS_URL: ws://localhost:8010/...) works
# unmodified - no need to regenerate/dirty it for local dev.
MODEL_GATEWAY_PORT=8001
RETRIEVAL_API_PORT=8010
WEB_PORT=5173

pids=()
cleanup() {
  echo
  echo "Stopping dev services..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> model-gateway on :$MODEL_GATEWAY_PORT"
uv run uvicorn model_gateway.main:app --reload \
  --reload-dir packages/model-gateway/src --reload-dir packages/common/src \
  --port "$MODEL_GATEWAY_PORT" &
pids+=($!)

echo "==> retrieval-api on :$RETRIEVAL_API_PORT (GATEWAY_URL=http://localhost:$MODEL_GATEWAY_PORT)"
GATEWAY_URL="http://localhost:$MODEL_GATEWAY_PORT" \
  uv run uvicorn retrieval_api.main:app --reload \
  --reload-dir packages/retrieval-api/src --reload-dir packages/common/src \
  --port "$RETRIEVAL_API_PORT" &
pids+=($!)

echo "==> web on :$WEB_PORT (vite dev server, HMR)"
(
  cd packages/web
  if [ ! -d node_modules ]; then
    npm install
  fi
  npm run dev -- --port "$WEB_PORT"
) &
pids+=($!)

echo
echo "model-gateway:  http://localhost:$MODEL_GATEWAY_PORT"
echo "retrieval-api:  http://localhost:$RETRIEVAL_API_PORT"
echo "web:            http://localhost:$WEB_PORT"
echo "Ctrl-C to stop everything."

wait
