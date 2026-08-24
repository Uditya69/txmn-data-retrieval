# Instant mode rerank on/off sample set

10 manual-testing queries for Instant mode's opt-in `rerank` toggle
(`instant/rerank.py::rrf_merge_by_doc_id`). `evals/instant_rerank_sample.json` is the
machine-readable source of truth; this document mirrors it.

Every gold `doc_id` below was pulled live from the ES index (`researchindex_aic_test`,
2026-08-20) — each one was fetched by `_doc/<id>` and its `headnotes_text`/`facts_text`/
`held_text` read directly before writing the corresponding query, so these are fresh
cases, not reused from `evals/retrieval_cases.json`.

**How to use**: run each query through Instant mode with `rerank` off, note whether the
gold `doc_id` appears in the merged ES+Milvus results; flip `rerank` on and re-run the
same query; compare. The `direct` queries are a control group (should already do fine
without rerank); the `indirect` queries are where rerank is most likely to matter, since
they carry no lexical overlap with the source document at all; `mixed` sits in between.

## Direct (citation + party name + section keywords)

| ID | Topic | Query | Gold doc_id |
|---|---|---|---|
| R01 | GST input tax credit | GMA Pinnacle Automotives State Tax Officer 161 taxmann.com 145 Kerala GSTR-2A GSTR-3B mismatch input tax credit | `101010000000353260` |
| R02 | Customs valuation | Commissioner of Customs Mumbai J D Orgochem 2008 taxmann.com 334 SC section 14 Customs Valuation DPIG Rules undervaluation | `101010000000094230` |
| R03 | DTAA royalty / PE | Oracle Systems Corporation ADIT International Taxation 62 taxmann.com 291 Delhi article 5 7 12 India USA DTAA software royalty permanent establishment | `101010000000164604` |
| R04 | IBC financial debt | Utsav Securities Timeline Buildcon 118 taxmann.com 171 NCLT New Delhi section 5(8) financial debt time value of money | `101010000000193057` |
| R05 | Penalty for concealment | Nitin Chauhan Income-tax Officer Nahan 97 taxmann.com 669 Chandigarh Tribunal section 271(1)(c) Explanation 1 concealment penalty | `101010000000183772` |

## Indirect (fact-pattern paraphrase, zero keyword overlap)

| ID | Topic | Query | Gold doc_id |
|---|---|---|---|
| R06 | GST input tax credit (paraphrase of R01) | If a buyer's input credit claim doesn't match because the supplier's own return shows a different figure, must the officer first give the buyer a chance to prove the supply actually happened before rejecting the claim outright? | `101010000000353260` |
| R07 | Customs valuation (paraphrase of R02) | When customs alleges an import was underpriced but cannot point to any comparable shipment sold higher around the same time, and the importer explains the low price through falling global prices, who wins that argument? | `101010000000094230` |
| R08 | DTAA royalty / PE (paraphrase of R03) | A foreign software company licenses copies of its software to an Indian customer and pays tax on the receipts as royalty at the DTAA rate. Can the department later reopen that assessment by claiming the income should instead be taxed as business profits through a permanent establishment? | `101010000000164604` |

## Mixed (partial keywords, no citation/party name)

| ID | Topic | Stress | Query | Gold doc_id |
|---|---|---|---|---|
| R09 | IBC financial debt (paraphrase of R04) | short, vague, no party name | NBFC gave interest free loan no proof of intent to charge interest can it still be treated as financial debt insolvency | `101010000000193057` |
| R10 | Penalty for concealment (paraphrase of R05) | typos, telegraphic | declared agri income but left out fd interst can he say he thot all deposit intrest was tax free 271 1 c penalty | `101010000000183772` |

See `evals/instant_rerank_sample.json` for `expected_collections` per query and full notes.
