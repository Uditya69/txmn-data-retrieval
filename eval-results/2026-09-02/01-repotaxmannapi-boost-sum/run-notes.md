# Eval run — 2026-09-02, run 01

`boost_source="sum"` baseline for Task 11 of the `repotaxmannapi`-exact-replica plan.
See `evals/2026-09-01-repotaxmannapi-replica-comparison.md` for the full comparison
against run 02 (`boost_source="repotaxmannapi"`).

```bash
uv run python -m retrieval_api.retrieval_eval \
  --no-langfuse --skip-synthesis --no-rerank --no-sparse --boost --boost-source sum \
  --jsonl-output eval-results/2026-09-02/01-repotaxmannapi-boost-sum/retrieval.jsonl --resume \
  --output eval-results/2026-09-02/01-repotaxmannapi-boost-sum/retrieval-full.json \
  --run-name repotaxmannapi-boost-sum
```

`model-gateway` unreachable in this environment (`--skip-synthesis --no-rerank
--no-sparse`) — only the `es` stage is meaningful; other stages read `>50` because
their embed/intent calls errored and degraded to empty results.

| stage | passed | total |
|---|---|---|
| es | 50 | 71 |

By query class (es pass rate): direct 20/24, indirect 8/24, adversarial 22/23.
