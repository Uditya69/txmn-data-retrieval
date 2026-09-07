#!/usr/bin/env bash
# Runs the full reranker-cap-sweep pipeline end to end: populates the
# retrieval-eval stage cache (ES/Milvus/SLM calls happen once here), then
# sweeps rerank_cap_sweep.py's candidate caps against it. Safe to re-run -
# retrieval-eval's stage cache is content-addressed per case/model/flags, so
# a second run reuses whatever's already cached instead of redoing it.
#
# Usage (from repo root):
#   evals/scripts/run_rerank_cap_sweep.sh
#   CAPS=100,50,25,20,10,5 evals/scripts/run_rerank_cap_sweep.sh
#
# Env vars (all optional):
#   CACHE_DIR     - stage cache dir (default: .rerank-cache)
#   CAPS          - comma-separated cap values (default: rerank_cap_sweep.py's own default)
#   SLM_MODEL / RERANKER_MODEL / SPARSE - must match between runs against the same CACHE_DIR
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CACHE_DIR="${CACHE_DIR:-.rerank-cache}"
mkdir -p .eval-results

SLM_ARGS=()
[[ -n "${SLM_MODEL:-}" ]] && SLM_ARGS+=(--slm-model "$SLM_MODEL")

RERANKER_ARGS=()
[[ -n "${RERANKER_MODEL:-}" ]] && RERANKER_ARGS+=(--reranker-model "$RERANKER_MODEL")

SPARSE_ARGS=()
if [[ "${SPARSE:-}" == "true" ]]; then SPARSE_ARGS+=(--sparse); fi
if [[ "${SPARSE:-}" == "false" ]]; then SPARSE_ARGS+=(--no-sparse); fi

CAPS_ARGS=()
[[ -n "${CAPS:-}" ]] && CAPS_ARGS+=(--caps "$CAPS")

echo "== Step 1/2: populating stage cache at ${CACHE_DIR} (ES/Milvus/SLM calls) =="
uv run retrieval-eval \
  --cache-dir "$CACHE_DIR" --skip-synthesis --no-langfuse \
  "${SLM_ARGS[@]}" "${RERANKER_ARGS[@]}" "${SPARSE_ARGS[@]}" \
  2>&1 | tee .eval-results/populate.log

echo
echo "== Step 2/2: sweeping reranker caps against ${CACHE_DIR} (reranker calls only) =="
uv run python evals/scripts/rerank_cap_sweep.py \
  --cache-dir "$CACHE_DIR" \
  "${CAPS_ARGS[@]}" "${SLM_ARGS[@]}" "${RERANKER_ARGS[@]}" "${SPARSE_ARGS[@]}" \
  2>&1 | tee .eval-results/sweep.log

echo
echo "Done. Sweep summary above; full JSON at .eval-results/rerank_cap_sweep_latest.json"
