# Eval run — 2026-09-01

Ran against `CHAT_PROVIDER=local` (self-hosted qwen3, see `.env`'s `LOCAL_*` vars) via
`evals/run_all.sh` on the remote server, `--resume`d after a mid-run restart. Full
per-case detail (query, rewrite, reasoning trace) is in the sibling `.jsonl` files in
this folder — summarize any of them locally with e.g.:

```bash
uv run python -m retrieval_api.slm_intent_eval --summarize eval-results/2026-09-01/slm-intent.jsonl
```

## slm-intent.jsonl (160 cases — category + rewrite + filters graded together)

- categories: 138/160 passed (exact=56, superset=1, safe-empty=81, wrong=22)
- rewrite: 160/160 passed
- filters: 160/160 passed
- all three checks: 138/160 passed
- errors: 0

## collection-routing.jsonl (80 cases — routing only, safe-empty tolerant)

- overall: 67/80 passed (exact=24, superset=2, safe-empty=41, wrong=13)
- confident cases: 48/60 passed (exact=24, superset=2, safe-empty=22, wrong=12)
- vague cases: 19/20 passed (safe-empty=19, wrong=1)
- errors: 0

## intent-filter.jsonl (40 cases — exact-match filters + categories)

- passed: 9/40
- errors: 0
- note: this eval is strict exact-match (no containment tolerance like slm-intent's
  filter check) — a low pass rate here may reflect that strictness rather than a
  proportionally worse filter-extraction defect. Cross-check against slm-intent's
  100% filter pass rate on the same underlying extract_intent() call before treating
  this as the primary signal.

## retrieval.jsonl (71 cases — full pipeline: ES/Milvus/RRF/reranker rank, citations)

Recall@pass_at by stage:
| stage | passed | total |
|---|---|---|
| es | 47 | 71 |
| raw_dense | 67 | 71 |
| raw_sparse | 57 | 71 |
| rewritten_dense | 55 | 71 |
| rewritten_sparse | 49 | 71 |
| rrf | 55 | 71 |
| reranker | 56 | 71 |

By query class (reranker pass rate): direct 17/24, indirect 18/24, adversarial 21/23.

Citations: 45/71 valid (no hallucinated doc_id), 49/71 actually cited the gold doc.

Errors: 0.
