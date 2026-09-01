#!/usr/bin/env bash
# Runs all 4 eval scripts back-to-back (each with --resume, so a re-run after a
# crash/kill just continues), then prints every result JSONL to stdout with clear
# delimiters - for the copy-paste-out-of-SSH workflow instead of scp/rsync.
#
# Usage:
#   tmux new -d -s all-evals 'bash evals/run_all.sh'   # survives disconnect
#   tmux attach -t all-evals                            # watch it / see the printed JSON later
#
# Or just run it in the foreground if you're staying connected:
#   bash evals/run_all.sh
#
# Override the gateway URL if it's not the default:
#   GATEWAY_URL=http://localhost:8001 bash evals/run_all.sh

set -uo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8001}"
mkdir -p .eval-results

echo "=== running slm_intent_eval ==="
uv run python -m retrieval_api.slm_intent_eval --gateway-url "$GATEWAY_URL" \
    --output .eval-results/slm-intent.jsonl --resume

echo "=== running collection_routing_eval ==="
uv run python -m retrieval_api.collection_routing_eval --gateway-url "$GATEWAY_URL" \
    --output .eval-results/collection-routing.jsonl --resume

echo "=== running intent_eval ==="
uv run python -m retrieval_api.intent_eval --gateway-url "$GATEWAY_URL" \
    --output .eval-results/intent-filter.jsonl --resume

echo "=== running retrieval_eval ==="
uv run python -m retrieval_api.retrieval_eval --gateway-url "$GATEWAY_URL" \
    --jsonl-output .eval-results/retrieval.jsonl --resume

echo
echo "===== ALL RESULTS (copy everything below this line) ====="
for name in slm-intent collection-routing intent-filter retrieval; do
    echo "----- BEGIN $name.jsonl -----"
    cat ".eval-results/$name.jsonl"
    echo "----- END $name.jsonl -----"
done
