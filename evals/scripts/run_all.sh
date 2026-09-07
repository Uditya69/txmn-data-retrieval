#!/usr/bin/env bash
# Runs all 4 eval scripts back-to-back (each with --resume, so a re-run after a
# crash/kill just continues), then prints every result JSONL to stdout with clear
# delimiters - for the copy-paste-out-of-SSH workflow instead of scp/rsync.
#
# Usage:
#   bash evals/scripts/run_all.sh [output-dir]
#
# output-dir defaults to .eval-results (gitignored scratch space). Pass a real path to
# write straight into a kept, dated eval-results/ folder instead - the dir is created
# if it doesn't exist:
#   bash evals/scripts/run_all.sh eval-results/2026-09-01/03-full-sweep
#
# Headless (survives SSH disconnect):
#   tmux new -d -s full-sweep 'bash evals/scripts/run_all.sh eval-results/2026-09-01/03-full-sweep'
#   tmux attach -t full-sweep   # watch it / see the printed JSON later

set -uo pipefail

OUT_DIR="${1:-.eval-results}"
mkdir -p "$OUT_DIR"

echo "=== running slm_intent_eval ==="
uv run python -m retrieval_api.slm_intent_eval \
    --output "$OUT_DIR/slm-intent.jsonl" --resume

echo "=== running collection_routing_eval ==="
uv run python -m retrieval_api.collection_routing_eval \
    --output "$OUT_DIR/collection-routing.jsonl" --resume

echo "=== running intent_eval ==="
uv run python -m retrieval_api.intent_eval \
    --output "$OUT_DIR/intent-filter.jsonl" --resume

echo "=== running retrieval_eval ==="
uv run python -m retrieval_api.retrieval_eval \
    --jsonl-output "$OUT_DIR/retrieval.jsonl" --resume

echo
echo "===== ALL RESULTS (copy everything below this line) ====="
for name in slm-intent collection-routing intent-filter retrieval; do
    echo "----- BEGIN $name.jsonl -----"
    cat "$OUT_DIR/$name.jsonl"
    echo "----- END $name.jsonl -----"
done
