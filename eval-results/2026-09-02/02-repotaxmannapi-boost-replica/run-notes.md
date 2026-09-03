# Eval run — 2026-09-02, run 02

`boost_source="repotaxmannapi"` run for Task 11 of the `repotaxmannapi`-exact-replica
plan (ported legacy .NET multiply-mode ES boost). See
`evals/2026-09-01-repotaxmannapi-replica-comparison.md` for the full comparison
against run 01 (`boost_source="sum"`).

```bash
uv run python -m retrieval_api.retrieval_eval \
  --no-langfuse --skip-synthesis --no-rerank --no-sparse --boost --boost-source repotaxmannapi \
  --jsonl-output eval-results/2026-09-02/02-repotaxmannapi-boost-replica/retrieval.jsonl --resume \
  --output eval-results/2026-09-02/02-repotaxmannapi-boost-replica/retrieval-full.json \
  --run-name repotaxmannapi-boost-replica
```

`model-gateway` unreachable in this environment (`--skip-synthesis --no-rerank
--no-sparse`) — only the `es` stage is meaningful; other stages read `>50` because
their embed/intent calls errored and degraded to empty results.

| stage | passed | total |
|---|---|---|
| es | 8 | 71 |

By query class (es pass rate): direct 7/24, indirect 0/24, adversarial 1/23.
