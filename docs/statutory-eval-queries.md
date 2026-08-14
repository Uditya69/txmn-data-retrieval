# Statutory-data retrieval evaluation set

This is a diagnostic retrieval benchmark built from the source statutory JSON files in
`tm-dp/data` (acts, rules, articles, commentary — `tariff` excluded, its Milvus
collection is parked and not live per `CLAUDE.md`). It contains 40 queries grouped into
20 matched pairs (`evals/statutory_cases.json` is the machine-readable source of truth;
this document mirrors it). Each pair targets the same gold document twice: once with
direct lexical signals (act/rule/regulation name, section number, distinctive terms of
art) and once through an indirect paraphrase of the same content with those identifiers
stripped out. Five gold documents were picked from each of the four document types.

Unlike the case-law eval (`docs/retrieval-eval-queries.md`), this set is **Milvus-only**.
`ES_INDEX` defaults to `taxmann_caselaw` (`packages/common/src/common/config.py`) — the
Elasticsearch index holds case law, not statutory content, so there is no ES leg to
evaluate here. Each gold document lives in exactly one Milvus collection determined by
its document type: `act_section`, `rule_section`, `article_section`, or
`commentary_section` (`common/schemas.py::MILVUS_COLLECTIONS`). None of these four
collections carry a `sparse_vector` field (`SPARSE_VECTOR_COLLECTIONS` excludes them —
the ingestion pipeline dropped their BM25 `Function`), so Milvus sparse search cannot
be evaluated on this set either. **Dense (Voyage) search is the only retrieval path
these collections support**; that is the intended target of this eval, not a gap.

## Suggested evaluation protocol

- Run every query through Milvus dense search only, against the query's single
  `expected_collections` entry (and, for a realistic AI Mode run, alongside the other
  ten collections it always searches — CLAUDE.md rule 4 routes by intent category, so a
  correctly classified query should route to this same collection).
- Record the rank of the gold `doc_id` in the collection's dense result list. Use `>50`
  when absent.
- Primary metric: gold-document Recall@5 (direct) / Recall@10 (indirect) per category.
- Secondary metric: reciprocal rank (`1 / rank`).
- For a direct query, treat gold rank <= 5 as a pass and rank 6-10 as a weak pass.
- For an indirect query, treat gold rank <= 10 as a pass and rank 11-20 as a weak pass.
- Do not require the exact-citation direct query to trivially outrank the indirect one —
  a disagreement is a result to investigate, not an automatic test failure. Statutory
  section text rarely repeats its own act name or section number verbatim inside the
  chunk in the way case-law headnotes repeat a citation, so direct queries here lean more
  on distinctive terms of art than on citation strings.

## Query set

### Pair 1 — revocable transfer of assets (Income-tax Act)

Gold document:

- `doc_id`: `102120000000013242`
- Source: Income-tax Act, 1961, section 61, "Revocable transfer of assets"
- Source file: `tm-dp/data/acts/direct-tax-laws/102120000000013242.json`
- Content: all income arising from a revocable transfer of assets is chargeable to
  income-tax as the income of the transferor, not the transferee.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q01 | Direct | `Income-tax Act 1961 section 61 revocable transfer of assets` | `act_section` | Gold `doc_id` in top 5 |
| Q02 | Indirect | `If a person can revoke a transfer of assets at will, is income from those assets taxed as belonging to the transferor or the transferee?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 2 — coastal goods provisions (Customs Act)

Gold document:

- `doc_id`: `102120000000002485`
- Source: Customs Act, 1962, section 98, "Application of certain provisions of this Act
  to coastal goods, etc."
- Source file: `tm-dp/data/acts/goods-services-tax/102120000000002485.json`
- Content: extends sections 33/34/36 and 37/38 (and, by notification, more of Chapter VI
  and section 45) from imported/export goods to coastal goods and the vessels carrying
  them.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q03 | Direct | `Customs Act 1962 section 98 coastal goods sections 33 34 36 37 38 apply` | `act_section` | Gold `doc_id` in top 5 |
| Q04 | Indirect | `Which customs provisions that apply to imported and export goods are extended to vessels carrying goods along the Indian coast?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 3 — agent's remuneration on misconducted business (Indian Contract Act)

Gold document:

- `doc_id`: `102120000000004640`
- Source: Indian Contract Act, 1872, section 220, "Agent not entitled to remuneration
  for business misconducted"
- Source file: `tm-dp/data/acts/ibc/102120000000004640.json`
- Content: an agent guilty of misconduct in part of an agency business forfeits
  remuneration only for that misconducted part, illustrated by a recovery-and-investment
  example.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q05 | Direct | `Indian Contract Act 1872 section 220 agent guilty of misconduct remuneration illustration` | `act_section` | Gold `doc_id` in top 5 |
| Q06 | Indirect | `If an agent handles part of a transaction negligently and causes a loss on that part, can he still claim commission for the part of the business he managed properly?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 4 — record-keeping for insurance agents (Insurance Act)

Gold document:

- `doc_id`: `102120000000005593`
- Source: Insurance Act, 1938, section 43, "Record of insurance agents"
- Source file: `tm-dp/data/acts/fema-banking-insurance/102120000000005593.json`
- Content: insurers must record an agent's name, address, and appointment start/end
  dates, retained for five years after the agent's appointment ceases.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q07 | Direct | `Insurance Act 1938 section 43 record of insurance agents five years` | `act_section` | Gold `doc_id` in top 5 |
| Q08 | Indirect | `For how long after an insurance agent's appointment ends must the insurer keep the record of that agent's name, address and appointment dates?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 5 — determination of disputed questions (Rajasthan VAT Act)

Gold document:

- `doc_id`: `102120000000033200`
- Source: Rajasthan Value Added Tax Act, 2003, section 36, "Determination of disputed
  questions"
- Source file: `tm-dp/data/acts/goods-services-tax/102120000000033200.json`
- Content: on application, the Commissioner determines questions such as dealer status
  or whether a transaction is a taxable sale, when the question arises outside a court
  proceeding.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q09 | Direct | `Rajasthan Value Added Tax Act 2003 section 36 determination of disputed questions Commissioner` | `act_section` | Gold `doc_id` in top 5 |
| Q10 | Indirect | `Under a state VAT law, who decides whether a person is a dealer or a transaction is a taxable sale when the question arises outside any court proceeding?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 6 — accountant's report under section 147(4)(a) (Income-tax Rules)

Gold document:

- `doc_id`: `103120000000061315`
- Source: Income-tax Rules, 1962, rule 69, "Report of accountant to be furnished under
  section 147(4)(a)"
- Source file: `tm-dp/data/rules/direct-tax-laws/103120000000061315.json`
- Content: the accountant's report required under section 147(4)(a) must be in Form
  No. 35.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q11 | Direct | `Income-tax Rules rule 69 report of accountant Form 35 section 147(4)(a)` | `rule_section` | Gold `doc_id` in top 5 |
| Q12 | Indirect | `Which prescribed form must a chartered accountant's report be furnished in under the income-tax provision requiring an accountant's report for reassessment?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 7 — auction sale of goods seized for tax default

Gold document:

- `doc_id`: `103120000000013177`
- Source: rule 86, "Auction sale of seized goods for default in payment of tax,
  penalty, etc"
- Source file: `tm-dp/data/rules/goods-services-tax/103120000000013177.json`
- Content: after a section 68 seizure and unpaid section 69(2) penalty, the seizing
  authority proclaims and publicly auctions the goods, with at least 30 days' notice and
  newspaper/public-view publication.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q13 | Direct | `auction sale of seized goods rule 86 section 68 69 thirty days proclamation` | `rule_section` | Gold `doc_id` in top 5 |
| Q14 | Indirect | `If seized goods are not redeemed after a default in paying tax and penalty, what public-auction and notice process must the authority follow before selling them?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 8 — approval of CIRP costs (IBBI CIRP Regulations)

Gold document:

- `doc_id`: `103120000000054592`
- Source: IBBI (Insolvency Resolution Process for Corporate Persons) Regulations,
  regulation 31B, "Approval of committee for insolvency resolution process costs"
- Source file: `tm-dp/data/rules/ibc/103120000000054592.json`
- Content: resolution-process costs incurred before the committee's first meeting need
  committee approval at that meeting; the resolution professional also prepares a Going
  Concern Assessment Report covering income/expenditure, working capital, and value-
  erosion risks, which the committee uses to decide whether operations continue.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q15 | Direct | `IBBI CIRP Regulations regulation 31B insolvency resolution process costs committee approval going concern assessment report` | `rule_section` | Gold `doc_id` in top 5 |
| Q16 | Indirect | `Who must approve the costs incurred during a corporate insolvency resolution process before the creditors' committee holds its first meeting, and what report assesses whether the debtor should keep operating?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 9 — advertisements by insurance intermediaries (IRDAI Regulations)

Gold document:

- `doc_id`: `103120000000046127`
- Source: IRDAI (Insurance Advertisements and Disclosure) Regulations, 2021,
  regulation 8, "Advertisements by insurance intermediaries"
- Source file: `tm-dp/data/rules/fema-banking-insurance/103120000000046127.json`
- Content: intermediaries authorised under the IRDA Act to solicit insurance business
  may advertise or solicit insurance through advertisements.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q17 | Direct | `IRDAI Insurance Advertisements and Disclosure Regulations 2021 regulation 8 insurance intermediaries advertisements` | `rule_section` | Gold `doc_id` in top 5 |
| Q18 | Indirect | `Are insurance intermediaries authorised to solicit insurance business also permitted to advertise that business?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 10 — multi-family office services (IFSCA Fund Management Regulations)

Gold document:

- `doc_id`: `103120000000058733`
- Source: IFSCA (Fund Management) Regulations, 2025, regulation 81, "Multi-Family
  Office"
- Source file: `tm-dp/data/rules/company-and-sebi/103120000000058733.json`
- Content: a Fund Management Entity may provide services to a multi-family office
  under a portfolio management agreement; the Authority may specify additional
  conditions.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q19 | Direct | `IFSCA Fund Management Regulations 2025 regulation 81 multi-family office portfolio management agreement FME` | `rule_section` | Gold `doc_id` in top 5 |
| Q20 | Indirect | `Can a fund management entity in an IFSC provide services to a multi-family office under a portfolio management agreement?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 11 — Ind-AS impact on the IT sector (article)

Gold document:

- `doc_id`: `105010000000013692`
- Source: "IND-AS Bottom-line Impact – Information Technology Sector: Case Study" by
  Vinayak Pai V., `[2016] 73 taxmann.com 249`
- Source file: `tm-dp/data/articles/account-audit/105010000000013692.json`
- Content: analyzes the financial-statement impact of Ind-AS convergence on IT-sector
  companies through a case study, part of the second phase of Ind-AS adoption for
  unlisted companies.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q21 | Direct | `Vinayak Pai IND-AS Bottom-line Impact Information Technology Sector Case Study 73 taxmann.com 249` | `article_section` | Gold `doc_id` in top 5 |
| Q22 | Indirect | `What is the bottom-line financial impact of transitioning to Ind-AS on companies in the information technology sector, illustrated through a case study?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 12 — mediation for operational creditors before section 9 filing (article)

Gold document:

- `doc_id`: `105010000000024721`
- Source: "Streamlining Insolvency: IBBI proposes Mediation for Operational Creditors
  before filing Section 9 Applications" by Parth Chourikar, `[2024] 168 taxmann.com 82`
- Source file: `tm-dp/data/articles/ibc/105010000000024721.json`
- Content: covers IBBI's November 2024 discussion paper proposing mediation as a
  preliminary step for operational creditors before filing a section 9 insolvency
  application.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q23 | Direct | `Parth Chourikar IBBI mediation operational creditors Section 9 applications 168 taxmann.com 82` | `article_section` | Gold `doc_id` in top 5 |
| Q24 | Indirect | `Has the insolvency regulator proposed making mediation a mandatory preliminary step for operational creditors before they can file an insolvency application against a corporate debtor?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 13 — year-end transfer-pricing true-ups in the UAE (article)

Gold document:

- `doc_id`: `105010000000026000`
- Source: "Don't Let Margins Miss the Arm's length Mark: True-Up Your Transfer Pricing
  in UAE" by Mohit Gupta, `[2025] 170 taxmann.com 291`
- Source file: `tm-dp/data/articles/transfer-pricing/105010000000026000.json`
- Content: explains why UAE MNEs newly subject to Corporate Tax should review
  related-party transactions at fiscal year-end and make true-up/true-down adjustments
  to meet the arm's-length standard.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q25 | Direct | `Mohit Gupta true-up transfer pricing UAE arm's length year-end adjustment 170 taxmann.com 291` | `article_section` | Gold `doc_id` in top 5 |
| Q26 | Indirect | `For UAE multinational businesses newly subject to Corporate Tax, why should related-party transactions be reviewed at fiscal year-end to align actual results with the arm's-length standard?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 14 — TCS on foreign remittances and the LRS (article)

Gold document:

- `doc_id`: `105010000000023024`
- Source: "TCS on foreign transactions: Banking Sector calls for extended deadline
  amid 'Internal System' upgrade obstacles", `[2023] 151 taxmann.com 453`
- Source file: `tm-dp/data/articles/fema-banking-insurance/105010000000023024.json`
- Content: after international credit-card spending was brought under the Liberalised
  Remittance Scheme, banks sought a deadline extension before implementing the 20% TCS
  rate, citing internal-systems upgrade obstacles.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q27 | Direct | `TCS foreign transactions banking sector extended deadline internal system upgrade LRS 151 taxmann.com 453` | `article_section` | Gold `doc_id` in top 5 |
| Q28 | Indirect | `Why did banks seek more time before implementing the higher TCS rate on foreign remittances after credit card spending abroad was brought under the Liberalised Remittance Scheme?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 15 — PRAVAAH portal for regulatory approvals (article)

Gold document:

- `doc_id`: `105010000000024038`
- Source: "Unveiling PRAVAAH: A New Era in Indian Financial Regulation",
  `[2024] 163 taxmann.com 25`
- Source file: `tm-dp/data/articles/fema-banking-insurance/105010000000024038.json`
- Content: RBI Governor Shaktikanta Das introduced the PRAVAAH portal (plus the Retail
  Direct Mobile App and a FinTech Repository) to speed up regulatory-approval
  applications.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q29 | Direct | `PRAVAAH RBI Shaktikanta Das Retail Direct Mobile App FinTech Repository 163 taxmann.com 25` | `article_section` | Gold `doc_id` in top 5 |
| Q30 | Indirect | `What new RBI portal was launched to simplify and speed up regulatory approval applications from banks and financial institutions?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 16 — undervalued transactions in personal bankruptcy (commentary)

Gold document:

- `doc_id`: `107010000000375193`
- Source: commentary on section 164, Insolvency and Bankruptcy Code, 2016 —
  "Undervalued transactions"
- Source file: `tm-dp/data/commentary/ibc/107010000000375193.json`
- Content: a bankruptcy trustee may apply to challenge an undervalued transaction the
  bankrupt entered into within two years before the bankruptcy application, if it also
  caused the bankruptcy process to trigger.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q31 | Direct | `undervalued transaction bankruptcy trustee section 164 Insolvency Code 2016 two years` | `commentary_section` | Gold `doc_id` in top 5 |
| Q32 | Indirect | `Under personal insolvency law, when can a bankruptcy trustee challenge a transaction the bankrupt entered into at less than fair value before the bankruptcy application was filed?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 17 — discharge order at end of moratorium (commentary)

Gold document:

- `doc_id`: `107010000000375163`
- Source: commentary on section 92, Insolvency and Bankruptcy Code, 2016 — "Discharge
  order"
- Source file: `tm-dp/data/commentary/ibc/107010000000375163.json`
- Content: the resolution professional files a final list of qualifying debts before
  the moratorium ends; the Adjudicating Authority then passes a discharge order
  releasing the debtor from those debts.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q33 | Direct | `discharge order resolution professional qualifying debts section 92 Insolvency Code 2016 moratorium` | `commentary_section` | Gold `doc_id` in top 5 |
| Q34 | Indirect | `At the end of the moratorium period in a personal insolvency process, who issues the order discharging the debtor from the qualifying debts on the final list?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 18 — possession of foreign currency under FEMA (commentary)

Gold document:

- `doc_id`: `107010000000350510`
- Source: commentary on the Foreign Exchange Management (Possession and Retention of
  Foreign Currency) Regulations, 2015 — "Restrictions on holding currency"
- Source file: `tm-dp/data/commentary/fema-banking-insurance/107010000000350510.json`
- Content: the possession/retention restrictions apply only to physical currency
  (regulation 2(ii)), not to foreign currency held in permissible accounts with
  authorised-dealer banks.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q35 | Direct | `Foreign Exchange Management Possession and Retention of Foreign Currency Regulations 2015 regulation 2(ii) physical possession` | `commentary_section` | Gold `doc_id` in top 5 |
| Q36 | Indirect | `Do FEMA's foreign-currency possession rules restrict foreign currency held in permitted bank accounts with authorised dealers, or only physical cash holdings?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 19 — the colourable-legislation doctrine (commentary)

Gold document:

- `doc_id`: `107010000000340808`
- Source: commentary — "Colourable legislation", citing *Asstt. Director of
  Inspection v. A.B. Shanthi* [2002] 255 ITR 258 (SC), *Ashok Kumar v. Union of India*
  [1991] 3 SCC 498, and *K.C. Gajapati Narayan Deo v. State of Orissa* AIR 1953 SC 375
- Source file: `tm-dp/data/commentary/company-and-sebi/107010000000340808.json`
- Content: explains colourable legislation — a law camouflaged to appear within the
  legislature's competence while indirectly, covertly, or disguisedly transgressing it.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q37 | Direct | `colourable legislation K C Gajapati Narayan Deo State of Orissa Shanthi Ashok Kumar Union of India` | `commentary_section` | Gold `doc_id` in top 5 |
| Q38 | Indirect | `What legal doctrine describes a law that appears to be within a legislature's competence but is actually an indirect or disguised attempt to exceed its constitutional power?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 20 — hydraulic jacks under Delhi VAT entry 84 (commentary)

Gold document:

- `doc_id`: `107010000000358135`
- Source: commentary — "Hydraulic Jack: Whether covered in Entry 84 (sub-entry 211(d)
  of DVAT Act, 2004?", citing a section 84 clarification by the Delhi Commissioner of
  Trade & Taxes
- Source file: `tm-dp/data/commentary/goods-services-tax/107010000000358135.json`
- Content: hydraulic jacks fall within "Tools and Dies" under Entry 84, sub-entry
  211(d) of the Third Schedule to the DVAT Act, 2004, and are taxable at 4%.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q39 | Direct | `Hydraulic Jack Entry 84 sub-entry 211(d) DVAT Act 2004 tools and dies clarification` | `commentary_section` | Gold `doc_id` in top 5 |
| Q40 | Indirect | `Under Delhi VAT, are hydraulic jacks taxed as industrial tools under the 'tools and dies' entry, and at what rate?` | `commentary_section` | Gold `doc_id` in top 10 |

## Result capture template

| Query | Milvus dense rank | Notes |
|---|---:|---|
| Q01 | | |
| Q02 | | |
| Q03 | | |
| Q04 | | |
| Q05 | | |
| Q06 | | |
| Q07 | | |
| Q08 | | |
| Q09 | | |
| Q10 | | |
| Q11 | | |
| Q12 | | |
| Q13 | | |
| Q14 | | |
| Q15 | | |
| Q16 | | |
| Q17 | | |
| Q18 | | |
| Q19 | | |
| Q20 | | |
| Q21 | | |
| Q22 | | |
| Q23 | | |
| Q24 | | |
| Q25 | | |
| Q26 | | |
| Q27 | | |
| Q28 | | |
| Q29 | | |
| Q30 | | |
| Q31 | | |
| Q32 | | |
| Q33 | | |
| Q34 | | |
| Q35 | | |
| Q36 | | |
| Q37 | | |
| Q38 | | |
| Q39 | | |
| Q40 | | |

## Reading failures

- Direct query fails but indirect query on the same pair succeeds: the chunk's dense
  embedding captures the substantive content well but the citation/act-name tokens
  themselves aren't semantically distinctive — expected more often here than in case law,
  since these collections have no sparse/BM25 path to catch exact-string queries.
- Both members of a pair fail: first verify the gold `doc_id` is actually present in the
  target Milvus collection (`common/milvus_client.py`) before inferring a ranking defect.
- A query consistently returns plausible-looking but wrong-collection hits: check that
  the query was searched against the collection matching its own category, and that
  `intent` category classification (if testing AI Mode end-to-end) routed to the right
  collection per `common/schemas.py::collections_for_intent()`.
- Dense results are broadly plausible but consistently miss these gold documents: verify
  `query_embed` resolved to Voyage, per the same caution as the case-law eval — a
  provider mismatch returns valid-looking vectors while silently destroying retrieval
  quality.
