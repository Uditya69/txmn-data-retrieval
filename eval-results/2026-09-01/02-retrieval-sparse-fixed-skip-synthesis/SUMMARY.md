# Eval run — 2026-09-01, run 02

Retrieval-only rerun after the `MILVUS_SPARSE_ENABLED` gating fix (previous run,
`01-full-sweep-sparse-unfixed`, ran Milvus sparse unconditionally — a bug, not real
behavior). `--skip-synthesis` was passed, so citation/gold-cited stats are not
meaningful here (no synthesis call was made). Ran against `CHAT_PROVIDER=local`
(self-hosted qwen3) via:

```bash
uv run python -m retrieval_api.retrieval_eval --gateway-url http://localhost:8001 \
  --jsonl-output eval-results/2026-09-01/02-retrieval-sparse-fixed-skip-synthesis/retrieval.jsonl \
  --resume --skip-synthesis
```

## retrieval.jsonl (71 cases)

Recall@pass_at by stage:
| stage | passed | total |
|---|---|---|
| es | 48 | 71 |
| raw_dense | 67 | 71 |
| raw_sparse | 0 | 71 |
| rewritten_dense | 59 | 71 |
| rewritten_sparse | 0 | 71 |
| rrf | 59 | 71 |
| reranker | 60 | 71 |

By query class (reranker pass rate): direct 18/24, indirect 19/24, adversarial 23/23.

Citations: not computed (`--skip-synthesis`).

**9 cases errored on the ES stage** (`ConnectionTimeout: Connection timed out`):
Q09, Q14, Q16, Q22, Q53, Q56, Q58, Q59, Q62. Worth a rerun of just those ids
(`--query Q09 --query Q14 ...`) once the ES connection is stable, or investigating
whether this is a transient network blip or a recurring issue with the remote ES host.

## Comparison to run 01 (`01-full-sweep-sparse-unfixed`)

| stage | run 01 (sparse unconditional) | run 02 (sparse correctly off) |
|---|---|---|
| es | 47/71 | 48/71 |
| raw_dense | 67/71 | 67/71 |
| raw_sparse | 57/71 | 0/71 (correct — was a bug in run 01) |
| rewritten_dense | 55/71 | 59/71 |
| rewritten_sparse | 49/71 | 0/71 (correct — was a bug in run 01) |
| rrf | 55/71 | 59/71 |
| reranker | 56/71 | 60/71 |

Reranker recall actually improved slightly (56→60) with sparse correctly disabled,
though the two runs aren't a clean apples-to-apples comparison (9 cases in run 02 hit
ES timeouts and effectively didn't run their ES stage) — worth rerunning those 9 ids
before drawing a firm conclusion.
