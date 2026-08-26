# Testing Notes

Branch: `feat/persona-interest-timeline` (not yet merged to `dev`). Live at:
- Web UI: http://localhost:8501
- API: http://localhost:8010

All test accounts share the password `TestPassw0rd!123`. These are throwaway
synthetic accounts seeded into the real Atlas cluster for this demo — safe to
reuse, re-seed, or delete at any time.

---

## What this demo shows

The persona system replaces a flat "average of everything you've ever asked"
profile with a **timeline**: queries cluster into topics, topics accumulate a
recency-weighted interest score, and a topic only counts as the user's
"current focus" once sustained (multi-day) activity backs it up. The demo
below walks through: a guest baseline, a user with a strong single focus, a
user who visibly changed focus over time, and a safety guardrail that stops
persona from hijacking an unrelated query.

---

## Demo accounts

| # | Persona | Email | user_id |
|---|---|---|---|
| 1 | IBC specialist — one sustained, active focus | `persona-test-ibc@example.test` | `897ae25d-7e3e-4937-b46b-4528a6837376` |
| 2 | Pivoted user — moved from IBC to GST | `persona-test-pivot@example.test` | `2a01929f-e1ee-4a8a-b281-87a2d04fed7f` |
| 3 | New user — one query ever, no real signal | `persona-test-new@example.test` | `f4705985-1a6c-4ec7-abb4-b5e3e52a10b9` |
| 4 | Legacy user — pre-migration flat persona doc | `persona-test-legacy@example.test` | `6e52c740-0111-47af-8031-925a6f75a090` |
| 5 | Income-tax specialist — Section 54F capital gains exemption | `persona-test-incometax@example.test` | `b5d77256-a07f-4c16-8b3d-2cc7dcfe9e22` |

---

## Demo script

### Act 1 — Baseline (guest, no login)

Ask: **`limitation period`**

Expect a generic answer covering the Consumer Protection Act, the Merchant
Shipping Act, and the Limitation Act generally — no lean toward any one legal
domain, because there's no persona to draw on.

### Act 2 — A strong, established focus (User 1: IBC specialist)

Sign in as `persona-test-ibc@example.test`. This account has an `active`
topic (score ~0.70) built from 8 real queries about Section 7 IBC
admission/financial creditor procedure, spread across 5 distinct days.

Ask the same query: **`limitation period`**

Expect the answer to open by specifically addressing the limitation period
for a **Section 7 IBC application** — unprompted by the query text, which on
its own gives no hint of IBC. Compare side-by-side against Act 1's guest
answer.

*(Talking point: same literal query, same retrieval corpus — the only
variable is who's logged in.)*

### Act 3 — Current focus, not lifetime average (User 2: pivoted user)

Sign in as `persona-test-pivot@example.test`. This account has:
- An **old** topic (queries from 95–75 days ago, all about IBC director
  liability) that stayed at `emerging` state — real activity, but never
  sustained long enough to be trusted.
- A **current** topic (queries from the last 10 days, all about GST input
  tax credit) that's `active`.

Query the debug endpoint to see the resolved persona directly (faster than
waiting on a full answer):

```bash
curl -sS -X POST http://localhost:8010/v1/intent-analysis \
  -H 'Content-Type: application/json' \
  -d '{"query":"eligibility conditions","user_id":"2a01929f-e1ee-4a8a-b281-87a2d04fed7f"}' \
  | python3 -m json.tool
```

Expect `persona_context_used` to mention only the **current GST** topic — the
old IBC research is correctly excluded, even though it's real, logged
history. This is the core "timeline, not lifetime average" behavior.

*(Talking point: under the old flat-average design, this user would still
read as IBC-leaning today, months after they stopped researching it.)*

### Act 4 — The guardrail (User 1 again, off-topic query)

Still signed in as the IBC specialist, ask something clearly unrelated to
their persona: **`what is the GST e-way bill validity period for goods above
50km`**

Expect a clean, on-topic GST answer — no forced IBC framing. Persona is
advisory, not authoritative: it only leans in when the query is genuinely
ambiguous, never when the query already answers its own question.

### Act 5 — New user, no signal yet (User 3)

Sign in as `persona-test-new@example.test` and ask anything.

Expect `persona_context_used: ""` — identical to a guest. One query isn't
enough evidence to trust; this is the low-evidence floor working as
designed.

### Act 6 (bonus, if time) — Migrating an old user (User 4)

`persona-test-legacy@example.test` was seeded with the *old* flat persona
schema (a `category_affinity` score, no event history) to simulate a user
who signed up before this system existed. The first real query from this
account silently converts that old document into seed events under the new
model — nothing breaks, nothing is lost, the old document is left untouched
for audit purposes. Ask it a query and then check:

```bash
curl -sS -X POST http://localhost:8010/v1/intent-analysis \
  -H 'Content-Type: application/json' \
  -d '{"query":"anything","user_id":"6e52c740-0111-47af-8031-925a6f75a090"}' \
  | python3 -m json.tool
```

`topic_count` will show 2 (migrated from its old `category_affinity`), though
both start too fresh/low-scored to render context yet — that's expected;
migration seeds history, it doesn't fast-forward trust.

### Act 7 — Income tax focus, and search_query actually expanding (User 5)

Sign in as `persona-test-incometax@example.test`. This account has an
`active` topic (score ~0.75) built from 9 real queries specifically about
**Section 54F capital gains exemption** (reinvesting property sale proceeds
into a new residential house), spread across 6 distinct days.

Ask any of these — none of them name Section 54F or even capital gains
explicitly, so a non-personalized answer would go generic:

- `exemption conditions`
- `time limit`
- `what happens if the new asset is sold early`
- `limitation period`

Check via the debug endpoint:

```bash
curl -sS -X POST http://localhost:8010/v1/intent-analysis \
  -H 'Content-Type: application/json' \
  -d '{"query":"exemption conditions","user_id":"b5d77256-a07f-4c16-8b3d-2cc7dcfe9e22"}' \
  | python3 -m json.tool
```

Expect `search_query` to expand to include **"Section 54F"** / **"Income-tax
Act 1961"**, and the answer to open discussing the Section 54F reinvestment
exemption specifically — this is the cleanest of the five accounts to show
the fixed `search_query`-expansion behavior on, since Income Tax is this
platform's core, best-covered domain (unlike the pivot user's GST content,
which is comparatively thinner in the corpus).

**More income-tax queries to try against this account** (useful as a
reference set beyond just this demo — swap in for Act 2/7's ambiguous-query
slot, or use standalone to explore corpus coverage):

- `cost of acquisition` (ambiguous on its own — could mean any capital asset)
- `net consideration`
- `deduction eligibility`
- `capital gains account scheme`
- `is a flat under construction eligible`
- `can I claim this along with section 54`

---

## Fast reference — checking any account without a full answer

```bash
curl -sS -X POST http://localhost:8010/v1/intent-analysis \
  -H 'Content-Type: application/json' \
  -d '{"query":"<query>","user_id":"<user_id>"}' | python3 -m json.tool
```

Returns `persona_found`, `persona_context_used`, and `topic_count` instantly —
no retrieval or synthesis, just what persona resolved to. Swap the endpoint
to `/v1/ai-mode-analysis` (same request shape) to see the actual generated
answer instead.

## Web UI gotcha

The sign-in modal tends to autofill with whatever real credentials are saved
in the browser's password manager — always clear both fields and type the
test account's own email/password before hitting Sign in.

---

## Insights worth mentioning in the demo

- **Persona reliably reaches the prompt every time** — confirmed via
  `persona_context_used` on every test run. The read/write wiring itself is
  solid.
- **Persona now actually changes retrieval, not just synthesis tone.**
  Earlier in this branch's testing, persona only ever showed up as a
  cosmetic opening sentence — the model would acknowledge "no IBC-specific
  material found" but the actual retrieved citations were identical to a
  guest's. Root cause: the SLM rewrite step (`extract_intent`) had no
  permission to treat a persona note as a basis for expanding
  `search_query`, and separately, a bug meant topic labels could surface the
  two weakest, most generic entities instead of the useful ones (e.g. `GST`,
  `Section 17(5)`). Both are fixed now — re-verified live, `search_query`
  expansion went from 0/3 to 3/3 on the same test user after the fix. Try
  Act 3 again for a live demo of this; `search_query` in the
  `/v1/intent-analysis` response should now visibly incorporate the
  relevant Act/section for a logged-in user on an ambiguous query.
- **Three real bugs were found and fixed getting this far**: a timezone
  mismatch that crashed against real MongoDB (never caught by unit tests,
  which use in-memory fakes), a clustering bug where near-identical queries
  about the same legal topic were landing in separate topics because entity
  matching was too strict for how an SLM phrases things run to run, and the
  search_query/topic-label issue above. All fixed on this branch now.
- **Semantic caching is off** in this environment
  (`SEMANTIC_CACHE_ENABLED=false`), so every demo query hits the real
  pipeline live — nothing here is a cached replay.
