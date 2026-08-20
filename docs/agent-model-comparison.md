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

Run the eval set (`evals/retrieval_cases.json`, documented query-by-query in
`evals/retrieval-eval-queries.md`) via:

```
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8001 \
  --run-name <model-name>-baseline \
  --no-langfuse \
  --query Q01 --query Q02 ... # optionally scope to a subset
```

This runs every selected query through 8 retrieval methods (`es`, `raw_dense`,
`raw_sparse`, `rewritten_dense`, `rewritten_sparse`, `rrf`, `reranker`, `agentic`) and
records the gold-document rank per method. Only the **`agentic`** column changes between
runs in this comparison — everything else is unaffected by the `agent_chat` model swap,
so it acts as a built-in control **as long as Milvus stays up for the whole run** (see
Results below — it didn't, for one of these runs). Pass criterion per query class
(direct ≤5, indirect ≤10, adversarial ≤20) is defined in `evals/retrieval-eval-queries.md`.

The script only writes its result file at the very end of the full loop — killing a run
partway through loses all of it, there's no incremental save. Scope with `--query` to
keep runs short enough to actually let finish, rather than killing a long run partway
through and losing the data.

Each run writes a timestamped result file to `.eval-results/` (gitignored — local
only, not the source of truth once compared here) plus a `latest.json`/
`latest.dataset.json` pointer pair.

## A blocking bug found before the comparison was meaningful

The original full-53-query run against `deepseek-ai/DeepSeek-V4-Flash-0731` was killed
after it visibly ran far slower than the Qwen baseline. Isolating a single query
(`agents/loop.py`'s `run_agent_loop`) showed why: this model made **33+ tool calls**
chasing increasingly unrelated case names (e.g. drifting from a share-capital-reduction
question into unrelated "Jupiter Capital"/"CMS Computers" cases) with no sign of
converging, and the smoke test eventually timed out with no final answer at all.

`run_agent_loop` had no cap on tool-call rounds - a bare `while True` trusting the model
to stop on its own. That held for every model tried before this one, but is not a safe
assumption in general. Fixed in commit `89e364e`: `MAX_TOOL_ROUNDS = 8`, after which the
loop forces one final `tools=[]` call so the model must answer from whatever it already
retrieved, guaranteeing termination regardless of model behavior. All results below are
post-fix.

## Results

Two 20-query runs (`Q01`-`Q20`, pairs 1-10, direct+indirect classes) for a head-to-head,
scoped down from the full 53 per a call to keep total eval time reasonable ("20-30
queries is good enough to validate").

**Confound:** Milvus (`57.159.24.173:19530`) went down again mid-run during the Qwen
run, starting at `Q07` (`MilvusException: ... Connection refused`, confirmed still down
via direct `nc` check afterward) - see `evals/retrieval-eval-queries.md`'s existing note
on this corpus-wide outage pattern. This knocked `raw_dense`/`raw_sparse`/`rrf`/
`reranker` down to 5-6/20 for Qwen's run (versus 14-18/20 for V4-Flash's run, where
Milvus stayed up throughout) - **those four columns are not comparable between runs**.
`es` is unaffected (doesn't depend on Milvus) and `agentic` is comparable but was a
real handicap for Qwen (many of its queries had to fall back to ES-only mid-run).

| Run | Model | `es` | `raw_dense` | `raw_sparse` | `rrf` | `reranker` | **`agentic`** |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash-0731-capped-20` | DeepSeek-V4-Flash-0731 | 13/20 | 17/20 | 14/20 | 17/20 | 16/20 | **4/20** |
| `qwen3-235b-a22b-capped-20` | Qwen3-235B-A22B | 13/20 | 6/20 *(Milvus down)* | 5/20 *(Milvus down)* | 6/20 *(Milvus down)* | 6/20 *(Milvus down)* | **10/20** |

Despite the Milvus handicap, Qwen still won on `agentic` (10/20 vs 4/20) - several of
its passes (`Q07`, `Q10`, `Q12`, `Q13`, `Q19`) came from falling back to ES-only
retrieval when Milvus errored, which is itself a good resilience signal.

**Clean subset (`Q01`-`Q06`, Milvus up for both runs, no confound):**

| Query | Qwen3-235B-A22B `agentic` | V4-Flash-0731 `agentic` |
|---|---|---|
| Q01 | 1 ✅ | >50 ❌ |
| Q02 | 1 ✅ | >50 ❌ |
| Q03 | 1 ✅ | >50 ❌ |
| Q04 | 1 ✅ | >50 ❌ |
| Q05 | 1 ✅ | >50 ❌ |
| Q06 | >50 ❌ | >50 ❌ |

**Qwen 5/6 vs V4-Flash 0/6** on identical queries with no infra confound — a decisive,
unambiguous result.

## Conclusion

**Keep `Qwen/Qwen3-235B-A22B` as the `agent_chat` default.** DeepSeek-V4-Flash-0731 is
cheaper per token and its individual `/v1/chat` calls are fast (~2s), but "enhanced
agentic capabilities" translated in practice to burning far more of its tool-call budget
on unfocused exploration - even under the round cap, it converged on the right answer in
only 4/20 (20%) queries versus Qwen's 10/20 (50%, itself understated by a mid-run Milvus
outage). Cost-per-token is irrelevant if the agent needs several times as many rounds
(and therefore several times the tokens and wall-clock latency) to reach a worse answer.
No further models are queued for comparison at this time.
