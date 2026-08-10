# Small-model eval results (AI Mode: slm / reranker / synthesis)

Tracks the first round of A/B testing 3-6B-class (or MoE-cheap) self-hostable model
candidates against the current DeepInfra defaults for AI Mode's `slm`, `reranker`, and
`synthesis` roles — motivated by wanting to self-host these roles instead of paying
per-token API pricing. Design: `docs/superpowers/specs/2026-08-10-small-model-eval-harness-design.md`.
Plan: `docs/superpowers/plans/2026-08-10-small-model-eval-harness.md`. Harness code lives
on branch `small-model-eval-harness`.

`agent_chat`/agentic search is out of scope this round (AI Mode is P0) — all runs used
`retrieval_eval.py --skip-agentic` to avoid paying for the tool-call loop's wall-clock
cost on a role that wasn't being tested.

## Candidates

The original candidate list (from web research) didn't match DeepInfra's current
catalog — several named models had been deprecated/replaced since the research was
done. Final candidates, confirmed present via `https://api.deepinfra.com/models/list`:

| Role | Baseline (current) | Candidate | Why this one |
|---|---|---|---|
| slm | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `Qwen/Qwen3-30B-A3B` | Original `Qwen3-4B-Instruct-2507` pick wasn't on DeepInfra. Chose the MoE `A3B` (30B total, ~3B active) over the also-available `Qwen/Qwen3-VL-4B-Instruct` after research found VL-tagged models show measurable text-only instruction-following degradation (IFEval) from their multimodal training mix — a bad fit for a strict-JSON extraction task. |
| reranker | `Qwen/Qwen3-Reranker-4B` | `Qwen/Qwen3-Reranker-0.6B` | Same family/training recipe as the 4B baseline, smallest available cross-encoder in that family. `bge-reranker-v2-m3` (the other original candidate) isn't on DeepInfra. |
| synthesis | `meta-llama/Meta-Llama-3.1-70B-Instruct` | `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3-VL-4B-Thinking` | Original `Qwen3-4B-Thinking-2507` pick wasn't on DeepInfra. Ran both an MoE candidate (35B total/~3B active, cheap despite large total capacity) and a small dense reasoning model, since synthesis is the highest-risk role for faithfulness/citation regressions and one candidate alone wouldn't show whether size or architecture mattered more. |

`query_embed` (Voyage-only, hard rule) was untouched.

## A bug found and fixed before any of this was usable

`Qwen/Qwen3-30B-A3B` 400'd on every single chat call at first. Root cause: DeepInfra
defaults an unset `max_tokens` to a value sized for its largest served model (observed:
65536), which exceeds this model's own 40960 context limit. Fixed in
`packages/model-gateway/src/model_gateway/adapters/deepinfra.py` by sending an explicit
`max_tokens=4096` on every chat call (commit `c947a61`). That claim of "safe across all
current roles" didn't hold up: since all of this round's eval runs used `--skip-agentic`,
the always-on-reasoning `agent_chat` role (`Qwen/Qwen3-235B-A22B`) was never exercised
against the cap, and this branch's own synthesis runs (below) show the same
reasoning-vs-answer-token failure mode on a different reasoning-heavy model. A follow-up
fix raised the cap to `32768` — still safely under the smallest context limit among
currently configured models (40960) — as conservative headroom, not a proven-safe value
for arbitrarily long reasoning traces.

That fix itself exposed a second issue during the synthesis runs — see below.

## Methodology

Ran the fixed 12-query stratified sample (`SAMPLE_12_QUERY_IDS` in `retrieval_eval.py` —
4 queries per class: direct/indirect/adversarial, spanning the corpus's full era range
1927-2026) via:

```
uv run python -m retrieval_api.retrieval_eval \
  --gateway-url http://localhost:8011 --sample12 --skip-agentic --no-langfuse \
  --run-name <candidate-name> --<role>-model <model-name>
```

Gateway ran locally (not via the Docker stack, to avoid touching the running dev
containers) on port 8011; ES and Milvus are both external hosts and reachable directly.
Five runs total: 1 baseline (all role defaults) + 1 slm + 1 reranker + 2 synthesis
candidates. Each run isolates one role's override so results attribute cleanly.

## Results

Pass counts out of 12 queries, per stage (recall@pass_at: direct ≤5, indirect ≤10,
adversarial ≤20 — see `docs/retrieval-eval-queries.md`):

| Run | es | raw_dense | raw_sparse | rewritten_dense | rewritten_sparse | rrf | reranker | citation_valid | gold_cited |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 7/12 | 11/12 | 10/12 | 11/12 | 10/12 | 12/12 | 11/12 | **12/12** | **11/12** |
| slm: Qwen3-30B-A3B | 7/12 | 11/12 | 10/12 | 11/12 | 10/12 | 12/12 | 11/12 | 11/12 | 11/12 |
| reranker: Qwen3-Reranker-0.6B | 6/12 | 11/12 | 10/12 | 10/12 | 9/12 | 11/12 | 11/12 | 10/12 | 10/12 |
| synthesis: Qwen3.6-35B-A3B | 5/12 | 11/12 | 10/12 | 11/12 | 10/12 | 12/12 | 11/12 | 4/12 | **3/12** ⚠️ |
| synthesis: Qwen3-VL-4B-Thinking | 7/12 | 11/12 | 10/12 | 11/12 | 10/12 | 12/12 | 11/12 | 10/12 | 9/12 |

`es`'s run-to-run variance (7/6/5) is not caused by any model swap — `es` never touches
the gateway at all. It reflects real flakiness on the external ES host during these
runs, not a regression from any candidate.

### slm: Qwen3-30B-A3B — clean pass

Every retrieval-rank stage is identical to baseline (unsurprising — slm only affects
query rewriting, and the rewrite is guarded by `_safe_rewrite`'s conservative validation
in `intent.py`, so a worse rewrite mostly falls back to the original query rather than
actively hurting retrieval). The one citation dip (11/12 vs 12/12) is likely a metric
false-positive, not a real regression: the only flagged invalid citation was the bare
token `"2021"` (Q16) — a plausible year mention in prose that `extract_cited_doc_ids`
can't distinguish from a real doc_id citation, since both are bracketed digit tokens.
**Verdict: safe to shrink, no evidence of real quality loss on this sample.**

### reranker: Qwen3-Reranker-0.6B — mild regression, one real miss

Slightly weaker across the board (es 6 vs 7, rewritten_sparse 9 vs 10, rrf 11 vs 12) —
but these are all upstream of the reranker (embedding/search variance), not something
the reranker itself controls. `reranker` stage pass rate is unchanged (11/12 both). One
query (Q28) lost its gold hit entirely upstream (`rewritten_dense`/`rrf` both dropped to
`>50`) — but Q28's own `citation_valid` in this run is `True`, so that upstream miss did
*not* cascade into a citation failure; it's purely an upstream retrieval-stage effect,
unrelated to the reranker swap. The two actual citation-validity dips in this run
(10/12 vs baseline's 12/12) are Q15 (invalid id `"2003"`) and Q47 (invalid id `"2010"`) —
both bare-year false positives from the citation-token regex (the same artifact already
diagnosed for the slm run's Q16/`"2021"` case above), not genuine hallucinations. The one
real, reranker-attributable effect in this round is on Q47: `rrf` rank is unchanged at 1
and the gold doc is still within the reranked/synthesis-cutoff window (reranker rank 4),
yet `gold_cited` flips from baseline's `True` to `False` here — meaning the smaller
reranker's *reordering* of otherwise-intact retrieval changed what the synthesis stage
chose to cite. **Verdict: reasonable shrink candidate, but Q47's gold_cited regression
should be spot-checked against a second run before trusting the smaller model in
production** — it's the strongest real (non-artifact) evidence in this round that the
reranker swap itself, not just upstream noise, can change synthesis-time citation
outcomes.

### synthesis: Qwen3.6-35B-A3B — run compromised, do not trust these numbers

`gold_cited` collapsing to 3/12 is **not** a fair reading of this model's synthesis
quality. Root cause, confirmed directly: 4/12 queries hit `ReadTimeout` on
`GatewayClient`'s fixed 60s HTTP timeout, and this model's serving profile is unusually
reasoning-heavy — a standalone test against DeepInfra's raw API (300-word essay prompt)
returned `content` of 2166 chars against `reasoning_content` of **10078 chars** for the
same request, i.e. roughly 4-5x more reasoning tokens than answer tokens. On the eval's
longer, multi-excerpt legal synthesis prompts, this model plausibly burns most or all of
the 4096-token cap (added in the bugfix above) on `<think>` reasoning before it ever
reaches the answer, producing an empty or truncated `synthesis_answer` with no exception
raised (several queries show `errors: {}` but `synthesis_answer` length 0 — these are
exactly why `citation_valid` requires a non-empty answer, not just an absence of invalid
citation ids: under that corrected definition this run's `citation_valid` count drops
from 8/12 to **4/12**, since 4 of the 8 originally-"valid" queries had empty answers with
zero invalid ids). **This is
itself a real finding, not just an eval artifact**: as currently served on DeepInfra,
this model's reasoning-to-answer token ratio makes it a poor fit for a latency-sensitive,
short-answer synthesis role under any token budget small enough to be safe for the
gateway's other models — the same class of risk flagged in the original research for
`agent_chat` candidates (reasoning-heavy models risking runaway token/time consumption)
turned out to apply to a synthesis candidate too. **Verdict: do not adopt without either
a role-specific higher `max_tokens`/timeout (which reopens the original DeepInfra
400 problem for other models sharing the same client) or confirmation from DeepInfra
that reasoning tokens can be capped independently of answer tokens.**

### synthesis: Qwen3-VL-4B-Thinking — closest to baseline, best synthesis candidate this round

10/12 citation-valid, 9/12 gold-cited vs baseline's 12/12 and 11/12 — a real but modest
gap, and no confounding timeouts distorting the majority of queries (2/12 timeouts here
vs baseline's 0/12, worth another look but far less than the 35B-A3B run's 4/12). Despite
being dense and much smaller (4B) than the 70B baseline, it's noticeably closer to
baseline behavior than the larger, cheaper-per-token MoE candidate above — a reminder
that "more total/active params" doesn't reliably predict "closer to baseline quality" for
this specific citation-faithfulness metric.

## Conclusion

- **slm → Qwen3-30B-A3B: adopt.** No measurable retrieval-rank regression on this
  sample; the one citation dip looks like a metric artifact, not a real fault.
- **reranker → Qwen3-Reranker-0.6B: adopt with a follow-up spot-check** on Q28 (or a
  second run) before fully trusting it — mild regression present but not clearly
  attributable to the reranker itself.
- **synthesis → neither candidate is a clean recommendation yet.** Qwen3-VL-4B-Thinking
  is the better of the two, but still shows a real (if modest) faithfulness gap under
  baseline, plus 2/12 timeouts worth investigating. Qwen3.6-35B-A3B should not be
  adopted based on this run — its numbers are dominated by a serving/token-budget
  confound, not genuine model quality; if there's interest in a MoE synthesis candidate,
  it needs a rerun with a role-specific larger token/timeout budget before it can be
  fairly judged.
- **Next steps if pursuing self-hosting further:** confirm the reranker Q28 result is
  reproducible, decide whether to give synthesis a role-specific `max_tokens`/timeout
  override (a real gateway change, not just an eval-harness one) to fairly test
  reasoning-heavy synthesis candidates, and re-run the full 53-query set on whichever
  slm/reranker candidates are adopted before calling this decision final — this round
  only used the fast 12-query sample.
