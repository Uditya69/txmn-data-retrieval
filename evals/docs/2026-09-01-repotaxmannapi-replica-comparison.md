# Eval comparison — `boost_source="sum"` vs `boost_source="repotaxmannapi"`

Task 11 of `.superpowers/sdd/2026-09-01-repotaxmannapi-exact-replica/`: real numbers
for the promote/reject decision on the ported legacy `.NET` (`repotaxmannapi`)
multiply-mode ES boost formula, wired behind `raw_search(..., boost_source: Literal["sum",
"repotaxmannapi"] = "sum")` in `packages/common/src/common/es_client.py`. Default is
unchanged (`"sum"`) — this doc records the comparison only, it does not make the
promote/reject call.

Both runs use `evals/datasets/retrieval_cases.json` (71 cases — grown since CLAUDE.md's earlier
53-query figure), scoped to the ES `es` stage only (`--skip-synthesis --no-rerank
--no-sparse`, `model-gateway` unreachable in this environment so dense/sparse/RRF/
reranker stages are not meaningful here — see each run's raw output). Branch
`feature/repotaxmannapi-exact-replica`, commit `9bc56c6` (the `retrieval_eval.py`
`--boost-source` plumbing added for this task).

```bash
# sum (existing default formula)
uv run python -m retrieval_api.retrieval_eval \
  --no-langfuse --skip-synthesis --no-rerank --no-sparse --boost --boost-source sum \
  --jsonl-output eval-results/2026-09-02/01-repotaxmannapi-boost-sum/retrieval.jsonl --resume \
  --output eval-results/2026-09-02/01-repotaxmannapi-boost-sum/retrieval-full.json \
  --run-name repotaxmannapi-boost-sum

# repotaxmannapi (ported legacy multiply-mode formula)
uv run python -m retrieval_api.retrieval_eval \
  --no-langfuse --skip-synthesis --no-rerank --no-sparse --boost --boost-source repotaxmannapi \
  --jsonl-output eval-results/2026-09-02/02-repotaxmannapi-boost-replica/retrieval.jsonl --resume \
  --output eval-results/2026-09-02/02-repotaxmannapi-boost-replica/retrieval-full.json \
  --run-name repotaxmannapi-boost-replica
```

## Result

| boost_source | es passed | total |
|---|---|---|
| `sum` (default) | 50 | 71 |
| `repotaxmannapi` | 8 | 71 |

By query class:

| class | sum pass | repotaxmannapi pass | n |
|---|---|---|---|
| direct | 20/24 | 7/24 | 24 |
| indirect | 8/24 | 0/24 | 24 |
| adversarial | 22/23 | 1/23 | 23 |

`repotaxmannapi`-mode is sharply worse on this eval set — 42 fewer passes than `sum`
mode, with 34 of those 71 cases (48%) not returning the gold `doc_id` anywhere in the
top 50 at all (`rep_rank=None`), versus far fewer such total misses under `sum`. This
lines up with CLAUDE.md's existing note on `boost_mode: "multiply"` (a different,
now-patched multiply formula routinely outweighing real text relevance by 10-50x) —
the newly-ported legacy formula reproduces the same class of problem, byte-exact to the
original .NET system's own behavior (that was the goal of the port — this is not a
bug in the port).

## Flipped queries (44 pass→fail, 2 fail→pass)

### PASS under `sum` → FAIL under `repotaxmannapi` (44 of 71)

| id | class | pass_at | sum rank | repotaxmannapi rank | query |
|---|---|---|---|---|---|
| Q04 | indirect | 10 | 5 | None | Is money received for resigning from a managing agency taxable as capital gains? |
| Q07 | direct | 5 | 1 | None | Gharda Chemicals Rule 57G Modvat invoice head office Dombivli plant |
| Q09 | direct | 5 | 1 | 6 | Ali Jawad Rizvi Indo French Biotech 1025 per cent return corporate veil |
| Q11 | direct | 5 | 1 | None | Alka Khandu Avhad section 138 141 non signatory wife joint liability cheque |
| Q12 | indirect | 10 | 6 | None | A husband issued a cheque for a jointly owed debt and it bounced. Can... |
| Q13 | direct | 5 | 1 | None | Vinodkumar Chechani section 83 rule 159 provisional attachment bank account |
| Q14 | indirect | 10 | 1 | None | Should GST authorities freeze every bank account as a routine revenue-... |
| Q15 | direct | 5 | 1 | None | Husco International 133 taxmann.com 196 article 12 India USA DTAA software |
| Q16 | indirect | 10 | 1 | None | A US parent buys off-the-shelf software seats and recovers the exact cost... |
| Q17 | direct | 5 | 1 | None | B G Shirke Airport Authority residential colony 12 percent GST Notification |
| Q21 | direct | 5 | 1 | None | Shah Mohanlal Chhotalal 10 ITC 46 Bombay section 4 bonus shares reduction |
| Q22 | indirect | 10 | 3 | None | Can the tax department disregard a court-approved reduction of share capital... |
| Q23 | adversarial | 20 | 1 | None | court aproved capital reduction bonus shars reserve can dept call it profit |
| Q24 | direct | 5 | 4 | None | J K Trust 32 ITR 535 Supreme Court managing agency charitable trust section |
| Q25 | indirect | 10 | 1 | None | Is remuneration earned by trustees from a managing agency income from... |
| Q26 | adversarial | 20 | 1 | None | managing agency itself trust property charity exemption case |
| Q31 | indirect | 10 | 1 | None | Is the income-tax penalty for concealment unconstitutional merely because... |
| Q32 | adversarial | 20 | 1 | None | 271 1 c consitutional validity hostile discrimnation confiscatory concealment |
| Q34 | indirect | 10 | 1 | None | Can a new industrial undertaking claim the backward-area deduction on... |
| Q35 | adversarial | 20 | 1 | None | 80HH scrap sale yes useless drum sale no metallic wire factory |
| Q36 | direct | 5 | 1 | None | Reliance Industries 1995 taxmann.com 569 Rule 57A 57C PTA MEG POY PSF |
| Q41 | adversarial | 20 | 3 | None | trade name training course takeover ward off competition deductible expense |
| Q42 | direct | 5 | 1 | 16 | Price Waterhouse 26 STT 291 Chennai transfer pricing certification Notification |
| Q43 | indirect | 10 | 1 | None | Does a chartered accountant's certificate issued after examining books... |
| Q44 | adversarial | 20 | 1 | None | PW transfer price certifcation audit notif 59 98 ST exemption CA Chennai |
| Q46 | indirect | 10 | 1 | None | Before trading was treated as an exempt service, could a commission agent... |
| Q47 | adversarial | 20 | 6 | None | trading not service before 1-4-2011 but no separate books pro rata cenvat |
| Q48 | direct | 5 | 1 | None | Sunil Chopra v CAPL Hotels Spa 175 taxmann.com 251 section 5(8) time value |
| Q49 | indirect | 10 | 1 | None | Can an interest-free payment with no repayment terms or contemporaneous... |
| Q50 | adversarial | 20 | 1 | None | IBC s7 loan no interest no time value no agreement recalled after 17 years |
| Q51 | direct | 5 | 1 | None | PCIT v Kross Diamonds 186 taxmann.com 345 section 69C section 105 Income tax |
| Q52 | indirect | 10 | 1 | None | Can genuine diamond purchases made through banking channels be treated... |
| Q53 | adversarial | 20 | 2 | None | 69C old act 105 new act diamond cash sale buyers unknown purchase genuine |
| Q54 | adversarial | 20 | 1 | None | contractr profit flat rate 10 pct no proof rebutal chance 1927 lahore |
| Q55 | adversarial | 20 | 3 | None | managing agency chhod diya paisa mila capital gain lagega kya agency band |
| Q56 | adversarial | 20 | 2 | None | customs board circular ignore port delay charge assessable value add s |
| Q57 | adversarial | 20 | 8 | None | modvat credit invoice head office naam factory pahuchi endorse kiya alag |
| Q60 | adversarial | 20 | 2 | None | gst dept sabhi bank account freeze kar sakta chota balance bina zarurat |
| Q61 | adversarial | 20 | 1 | None | software seat cost recover no repro rights royalty ya business income |
| Q65 | indirect | 10 | 5 | None | If an employer deducts TDS from an employee's salary but never deposits it... |
| Q66 | adversarial | 20 | 4 | None | employr tds kata deposit nahi kiya employee credit milega ya nahi recovery |
| Q67 | direct | 5 | 1 | None | Bindu Projects Co 130 taxmann.com 372 AAR Karnataka 12 percent GST railway |
| Q68 | indirect | 10 | 1 | None | What GST rate applies to a contractor building a new railway station... |
| Q69 | adversarial | 20 | 1 | None | railway station banane wale works contract pe gst rate 12 ya 18 kitna |

### FAIL under `sum` → PASS under `repotaxmannapi` (2 of 71)

| id | class | pass_at | sum rank | repotaxmannapi rank | query |
|---|---|---|---|---|---|
| Q03 | direct | 5 | 24 (fail) | 1 (pass) | 32 ITR 190 Provident Investment managing agency section 12B capital gains |
| Q33 | direct | 5 | None (fail) | 1 (pass) | ITO v Poly Tech Cable Products 11 ITD 20 section 80HH scrap metal drums |

## Raw run output

- `eval-results/2026-09-02/01-repotaxmannapi-boost-sum/` — `retrieval.jsonl`,
  `retrieval-full.json`, `retrieval-full.dataset.json`
- `eval-results/2026-09-02/02-repotaxmannapi-boost-replica/` — same files, for the
  `repotaxmannapi` run

## Caveat

`model-gateway` was not reachable in this environment, so `raw_dense`/`raw_sparse`/
`rewritten_dense`/`rewritten_sparse`/`rrf`/`reranker` all read `>50`/not-passed for
both runs — those stages don't exercise `boost_source` at all (it's an ES-only
parameter, `common/es_client.py::raw_search`), so this doesn't affect the comparison
above, but it does mean this run cannot speak to end-to-end AI Mode behavior — only to
Instant mode's raw ES ranking, which is the surface `boost_source` actually touches.
