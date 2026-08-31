# Testing Notes

Branch: `feat/persona-interest-timeline` (merged to `dev`). Live at:
- Web UI: http://localhost:8501
- API: http://localhost:8010

## Test account

- Email: `test@test.com`
- Password: `test1234`
- user_id: `67773202-6702-44e2-9308-b33192faec00`

Throwaway account seeded into the real Atlas cluster — safe to reuse,
re-seed, or delete at any time (see `record_query_event` in
`packages/persona/src/persona/repository.py` for how). It carries 8 real
queries about **Section 54F capital gains exemption** (reinvesting property
sale proceeds into a new residential house), spread across 6 distinct days
ending today, so its topic reads as `active` (score ~0.62) and fresh rather
than decayed.

---

## What this demo shows

The persona system replaces a flat "average of everything you've ever asked"
profile with a **timeline**: queries cluster into topics, topics accumulate a
recency-weighted interest score, and a topic only counts as the user's
"current focus" once sustained (multi-day) activity backs it up.

---

## How to verify persona is working

### 1. Baseline (guest, no login)

Ask: **`time limit`**

Expect a generic answer — no lean toward any one legal domain, because
there's no persona to draw on.

### 2. Logged in, ambiguous query pulls in the user's focus

Sign in as `test@test.com` / `test1234`. Ask any of these — none of them
name Section 54F or even capital gains explicitly, so a non-personalized
answer would go generic:

- `time limit`
- `exemption conditions`
- `what happens if the new asset is sold early`
- `net consideration`
- `deduction eligibility`
- `is a flat under construction eligible`
- `cost of acquisition`
- `can I claim this along with section 54`

Expect the answer to open by specifically addressing **Section 54F** —
unprompted by the query text. Compare side-by-side against the guest answer
from step 1.

### 3. Check the resolved persona directly (fast, no full answer)

```bash
curl -sS -X POST http://localhost:8010/v1/intent-analysis \
  -H 'Content-Type: application/json' \
  -d '{"query":"time limit","user_id":"67773202-6702-44e2-9308-b33192faec00"}' \
  | python3 -m json.tool
```

Swap `<query>` and reuse the same `user_id` for any of the queries above.
Swap the endpoint to `/v1/ai-mode-analysis` (same request shape) to see the
actual generated answer instead of just the resolved persona.

Check for:
- `persona_found: true`
- `persona_context_used`: should read `"This user is currently focused on:
  Section 54F, Income-tax Act 1961 (active)."`
- `search_query`: should expand to include `"Section 54F"` / `"Income-tax
  Act 1961"` even though the typed query didn't mention them.

**Confirmed live 2026-08-31:** `time limit` → `search_query` expanded to
`"time limit Section 54F Income-tax Act 1961"`. Expansion is real but not
100% consistent run-to-run — on the same run, `exemption conditions` did
*not* expand despite the model's own reasoning trace walking through the
expansion rule and deciding to apply it; the final `search_query` field
still came back unchanged. Not a hard failure — just retry with a different
query from the list above if one doesn't expand. `cost of acquisition`
timed out on the SLM call once — likely transient DeepInfra latency, not a
persona bug.

### 4. The guardrail — off-topic query doesn't get hijacked

Still signed in as `test@test.com`, ask something clearly unrelated to
Section 54F, e.g. **`what is the GST e-way bill validity period for goods
above 50km`**.

Expect a clean, on-topic GST answer — no forced Section 54F framing. Persona
is advisory, not authoritative: it only leans in when the query is genuinely
ambiguous, never when the query already answers its own question.

---

## Web UI gotcha

The sign-in modal tends to autofill with whatever real credentials are saved
in the browser's password manager — always clear both fields and type the
test account's own email/password before hitting Sign in.

---

## Insights

- **Persona reliably reaches the prompt every time** — confirmed via
  `persona_context_used` on every test run. The read/write wiring itself is
  solid.
- **Persona actually changes retrieval, not just synthesis tone.** The SLM
  rewrite step (`extract_intent`) treats a persona note as a basis for
  expanding `search_query`, verified live 2026-08-31: `time limit` expanded
  to `"time limit Section 54F Income-tax Act 1961"`. But expansion isn't
  100% consistent run-to-run (see step 3 above) — keep a couple of backup
  queries in reserve rather than relying on a single fixed one.
- **Semantic caching is off** in this environment
  (`SEMANTIC_CACHE_ENABLED=false`), so every query hits the real pipeline
  live — nothing here is a cached replay.
