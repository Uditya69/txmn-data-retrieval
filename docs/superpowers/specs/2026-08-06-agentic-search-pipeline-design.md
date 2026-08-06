# Agentic Search Pipeline — Design

Date: 2026-08-06

## Purpose

Third retrieval path, alongside Instant and AI Mode. An LLM-driven tool-calling agent that decides which search tools to call, loops until satisfied, and produces a cited answer — no hallucinated citations. Built to A/B against AI Mode on the same eval harness to see if agentic tool selection beats the fixed RRF+rerank pipeline.

Search-only tools for now (ES, Milvus dense/sparse, doc lookup). Non-search tools (calculator, statute API, etc.) are a later extension, not in scope here.

## Non-goals

- No changes to Instant or AI Mode paths.
- No new Milvus/ES collections or schema changes.
- No UI integration into the main search flow — standalone page only, for now.
- No step cap on the tool-calling loop (may be added later if needed).

## Architecture

### New package: `packages/agents`

Matches the existing `common` / `model-gateway` / `retrieval-api` package split.

- **Tool definitions** — thin wrappers around existing `common` ES/Milvus clients. No new retrieval logic; reuses hard rules already enforced there (Voyage-only `query_embed`, sparse search by raw text, no ES/Milvus score fusion).
  - `search_es(query)`
  - `search_milvus_dense(collection: enum[7 collections], query)`
  - `search_milvus_sparse(collection: enum[7 collections], query)`
  - `lookup_doc(doc_id)`

  Chose few generic tools with an enum param over one dedicated tool per collection (17+ tools) — smaller tool schema per model call, cheaper, and collection choice is just a parameter, not a distinct capability, so splitting into separate tools buys no accuracy.

- **Agent loop** — LLM-driven, tool-calling, uncapped step count. Uses a new `agent_chat` role in model-gateway's `ROLE_PROVIDER_MAP`, deliberately separate from AI Mode's synthesis role/model so the model backing this path can be swapped and compared independently.

- **Citation validator** — runs after the model emits a final (non-tool-call) answer. Extracts every cited `doc_id` from the answer and checks it against the set of `doc_id`s actually returned by tool calls during that run.
  - All citations valid → answer accepted.
  - Any invalid → rejection fed back to the model as feedback, loop retries (max 3 total attempts).
  - Still invalid after 3 attempts → pipeline returns an explicit **unverifiable-answer** state. Never returns a best-effort/unverified guess — reliability over always-answering.

### `retrieval-api` changes

New route, same process (async, same WS-streaming pattern already used for AI Mode) — no separate service needed since model calls proxy through model-gateway and the process is already async. Route drives the `packages/agents` loop and streams each tool call/step live over the WS connection as it happens.

### Frontend changes

New standalone page (separate from the main search page, same pattern as `/debug`), showing:
- Live trace of each tool call as it streams in.
- Final cited answer, or the unverifiable-answer state if validation exhausted retries.

### Eval harness

Existing retrieval-eval CLI gets a new target mode that runs the same query set through this pipeline headless (no WS) and reports metrics for comparison against AI Mode.

## Data Flow

1. User submits query on new page → WS connection opens to new `retrieval-api` route.
2. Route hands query to the agent loop.
3. Loop calls the `agent_chat` model with tool schemas + query/history. Model either calls a tool or emits a final answer.
   - Tool call → route executes it via `common` client → result streamed to UI as a trace step → fed back to model as tool output.
   - Repeats with no step cap.
4. Model emits final answer → citation validator checks it.
   - Valid → stream final answer, close connection.
   - Invalid → feed rejection reason back to model, retry (up to 3 total attempts).
   - 3rd failure → stream unverifiable-answer state, close connection.
5. Eval harness path: same loop invoked headless per query in the existing query set; results compared against AI Mode on the same metrics.

## Error Handling

- **Tool call fails** (ES/Milvus timeout, bad param, etc.) → surfaced to the model as a tool-error result, not a crash. Model can retry the call or pick a different tool. No cap on this — same uncapped loop as normal operation.
- **Model/gateway failure** (e.g. `agent_chat` model call returns 5xx or times out) → loop aborts; route streams a hard-failure state to the UI, distinct from the unverifiable-answer state (this is infra failure, not a citation problem).
- **Citation-retry exhaustion** (3 attempts) → unverifiable-answer state, per the reliability requirement above.

## Testing

- **`packages/agents` unit tests**:
  - Tool wrappers, with `common` clients mocked.
  - Loop step logic — mock `agent_chat` responses, verify it stops on a final answer, verify it doesn't infinite-loop on repeated tool-errors.
  - Citation validator — valid citation set, invalid set, and the retry-then-unverifiable path.
- **`retrieval-api` integration tests**: new route/WS, `packages/agents` loop mocked — same style as existing AI Mode WS tests.
- **Eval harness**: extend existing CLI to add this pipeline's mode; no new test framework.

## Open Questions / Future Work

- Step cap on the tool-calling loop, if latency/cost becomes an issue in practice.
- Non-search tools (calculator, statute lookup API, etc.).
- Whether this path graduates from standalone page into the main search UI as a third mode, pending eval results.
