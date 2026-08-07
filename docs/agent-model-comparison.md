# Agent chat model comparison

Tracks which DeepInfra model backs the `agent_chat` role (`DEEPINFRA_CHAT_MODEL_AGENT`
in `.env`/`.env.example`) — the model driving the tool-calling agent at `/ws/agent`
(`packages/agents`). Kept here so the comparison and its rationale survive past any
one session, instead of living only in chat history.

## Why this changed

`agent_chat` was pinned to `deepseek-ai/DeepSeek-V3` because, at the time, it was "the
only model DeepInfra explicitly documents as supporting tool calling" (see
`.env.example` history / commit `0fd823a`). That constraint is stale — essentially
every model in DeepInfra's current catalog is tagged `tools`. Once that constraint
lifted, cost-to-performance became the deciding factor, and a real bug (see below)
meant the model choice mattered less than expected for correctness.

Separately, a structural bug was found and fixed first (commit `f96aaf1`): the agent's
`search_milvus_dense`/`search_milvus_sparse` tools required the model to guess a single
Milvus collection per call, and the model often guessed wrong even when the gold
document ranked #1 in a different collection. That tools fix (search all 7 collections,
merge+sort results) was necessary before a model comparison would even be meaningful —
otherwise a "bad" result could just be the collection-guessing bug, not the model.

## Candidates considered

Pricing pulled live from `https://api.deepinfra.com/models/list` (2026-08-07):

| Model | Input $/M tokens | Output $/M tokens | Tags |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V3` (previous default) | $0.32 | $0.89 | `tools` |
| `deepseek-ai/DeepSeek-V3.1` | $0.25 | $0.95 | `reasoning`, `tools`, `can-disable-reasoning` |
| `Qwen/Qwen3-235B-A22B` (current default) | $0.18 | $0.54 | `reasoning`, `tools` |
| `deepseek-ai/DeepSeek-V4-Flash` | $0.09 | $0.18 | `reasoning`, `tools`, `can-disable-reasoning` |

Notes:
- `Qwen3-235B-A22B` is a sparse MoE — 235B total parameters, but only 22B active per
  token — which is why it's cheaper than V3 despite being "bigger" on paper.
- `DeepSeek-V4-Flash` is a newer generation than V3 and the cheapest reasoning+tools
  option found; "Flash" naming usually trades some raw capability for cost/speed, not
  yet verified against this eval set.
- `DeepSeek-V3.1`/`V3.1-Terminus` are hybrid models (`can-disable-reasoning`) - same
  lineage as the original V3 default, useful if isolating "reasoning on vs off" as the
  only variable matters more than comparing model families.

## Methodology

Run the full 53-query eval set (`evals/retrieval_cases.json`, documented query-by-query
in `docs/retrieval-eval-queries.md`) via:

```
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 \
  --run-name <model-name>-baseline \
  --no-langfuse
```

This runs every query through 8 retrieval methods (`es`, `raw_dense`, `raw_sparse`,
`rewritten_dense`, `rewritten_sparse`, `rrf`, `reranker`, `agentic`) and records the
gold-document rank per method. Only the **`agentic`** column changes between runs in
this comparison — everything else is unaffected by the `agent_chat` model swap, so it
acts as a built-in control. Pass criterion per query class (direct ≤5, indirect ≤10,
adversarial ≤20) is defined in `docs/retrieval-eval-queries.md`.

Each run writes a timestamped result file to `.eval-results/` (gitignored — local
only, not the source of truth once compared here) plus a `latest.json`/
`latest.dataset.json` pointer pair.

## Status

| Run | Model | Status |
|---|---|---|
| `qwen3-235b-a22b-baseline` | `Qwen/Qwen3-235B-A22B` | Running (started 2026-08-07 ~07:27 UTC) |
| *(queued)* | `deepseek-ai/DeepSeek-V4-Flash-0731` | Not started - queued to run after the Qwen baseline finishes. Note the `-0731` dated suffix specifically: DeepInfra's own description calls it "the official release of DeepSeek-V4-Flash, superseding the preview version, with substantially enhanced agentic capabilities" - the un-suffixed `DeepSeek-V4-Flash` is the earlier, weaker preview. |

## Results

_To be filled in once both runs complete._ Will report, per model: `agentic` pass rate
overall and per query class (direct/indirect/adversarial), plus any queries where the
model's answer was correct but hallucinated/mis-cited (a separate axis from
`retrieval_eval.py`'s recall check - it only checks whether cited `doc_id`s were
tool-retrieved and where they rank, not whether the answer text itself is accurate).

| Model | Direct pass | Indirect pass | Adversarial pass | Overall |
|---|---|---|---|---|
| Qwen/Qwen3-235B-A22B | TBD | TBD | TBD | TBD |
| DeepSeek-V3.1 | TBD | TBD | TBD | TBD |
