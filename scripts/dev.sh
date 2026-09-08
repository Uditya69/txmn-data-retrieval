#!/usr/bin/env bash
# One-command local dev: retrieval-api (uvicorn --reload, model-gateway included
# in-process) + web (vite dev server). Hits the remote Milvus/ES from .env directly -
# no docker, no rebuild needed on code changes.
set -euo pipefail
cd "$(dirname "$0")/.."

# Port matches docker-compose's old host mapping so the git-tracked
# packages/web/public/env-config.js (WS_URL: ws://localhost:8010/...) works
# unmodified - no need to regenerate/dirty it for local dev.
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

echo "==> retrieval-api on :$RETRIEVAL_API_PORT"
uv run uvicorn retrieval_api.main:app --reload \
  --reload-dir packages/retrieval-api/src --reload-dir packages/common/src \
  --reload-dir packages/model-gateway/src \
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
echo "retrieval-api:  http://localhost:$RETRIEVAL_API_PORT"
echo "web:            http://localhost:$WEB_PORT"
echo "Ctrl-C to stop everything."

wait
