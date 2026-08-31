# Statutory-data retrieval evaluation set

This is a diagnostic retrieval benchmark built from the source statutory JSON files in
`tm-dp/data` (acts, rules, articles, commentary — `tariff` excluded, its Milvus
collection is parked and not live per `CLAUDE.md`). It contains 80 queries grouped into
40 matched pairs (`evals/statutory_cases.json` is the machine-readable source of truth;
this document mirrors it). Each pair targets the same gold document twice: once with
direct lexical signals (act/rule/regulation name, section number, distinctive terms of
art) and once through an indirect paraphrase of the same content with those identifiers
stripped out. Ten gold documents were picked from each of the four document types (Pairs
1-20 are the original set; Pairs 21-40 were added in the same style, five more per
category, with `gold_doc_ids` mined from the live ES/Milvus corpus).

Unlike the case-law eval (`evals/retrieval-eval-queries.md`), this set is **Milvus-only**.
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

### Pair 21 — capital-gain exemption on sale of agricultural land (Income-tax Act, amending provision)

Gold document:

- `doc_id`: `102120000000010383`
- Source: Finance Act amendment clause, "Amendment of section 54B" (Income-tax Act, 1961)
- Content: amends section 54B(1) to extend the agricultural-land capital-gain exemption
  from "the assessee or a parent of his" to "the assessee being an individual or his
  parent, or a Hindu undivided family", effective 1 April 2013 — broadening eligibility to
  HUFs.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q41 | Direct | `amendment of section 54B Income-tax Act sale of agricultural land capital gain exemption` | `act_section` | Gold `doc_id` in top 5 |
| Q42 | Indirect | `If I sell agricultural land and reinvest the proceeds in new farmland, is the capital gain exempt under a specific Income-tax Act section, and how is that different from the residential-house exemption under section 54F?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 22 — TDS on payments to sub-contractors (Income-tax Act, amending provision)

Gold document:

- `doc_id`: `102120000000009662`
- Source: Finance Act amendment clause, "Amendment of section 194C" (Income-tax Act, 1961)
- Content: inserts a proviso into section 194C after sub-section (2), effective 1 June
  2002, bringing individuals/HUFs whose turnover exceeds the section 44AB tax-audit
  threshold within the obligation to deduct TDS on payments to sub-contractors.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q43 | Direct | `amendment of section 194C Income-tax Act TDS payments to contractors sub-section (2)` | `act_section` | Gold `doc_id` in top 5 |
| Q44 | Indirect | `Was there a change to the TDS rule that applies when a company pays a sub-contractor, and what proviso got added?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 23 — mode of repayment of loans/deposits (Income-tax Act, amending provision)

Gold document:

- `doc_id`: `102120000000009688`
- Source: Finance Act amendment clause substituting section 269T, "Mode of repayment of
  certain loans or deposits" (Income-tax Act, 1961)
- Content: substitutes a new section 269T, effective 1 June 2002, requiring repayment of
  a loan or deposit of Rs. 20,000 or more (including accrued interest, or aggregated
  across loans/deposits with the same branch) to be made only by account-payee cheque or
  account-payee bank draft (or by crediting the payee's account at the same branch).

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q45 | Direct | `substitution of section 269T Income-tax Act mode of repayment of certain deposits` | `act_section` | Gold `doc_id` in top 5 |
| Q46 | Indirect | `Which section restricts how a company or firm can repay a deposit above a threshold in cash?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 24 — arm's length price computation (Income-tax Act, amending provision)

Gold document:

- `doc_id`: `102120000000009625`
- Source: Finance Act amendment clause, "Amendment of section 92C" (Income-tax Act, 1961)
- Content: substitutes the proviso to section 92C(2) to introduce the tolerance-band
  concept for arm's length price (arithmetical mean of prices under the most appropriate
  method, or a price within 5% of that mean at the assessee's option), and amends the
  second proviso to sub-section (4) to cover income "deductible" and not just "deducted".

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q47 | Direct | `amendment of section 92C Income-tax Act arm's length price transfer pricing proviso` | `act_section` | Gold `doc_id` in top 5 |
| Q48 | Indirect | `What provision governs how the arm's length price is computed for related-party transactions, and was the tolerance-band proviso amended?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 25 — deduction for contribution to pension scheme (Income-tax Act, amending provision)

Gold document:

- `doc_id`: `102120000000009806`
- Source: Finance Act amendment clause inserting new section 80CCD, "Deduction in respect
  of contribution to pension scheme of Central Government" (Income-tax Act, 1961)
- Content: inserts section 80CCD after section 80CCC, allowing a Central Government
  employee (joining on or after 1 January 2004) a deduction for amounts paid or deposited
  into a notified pension-scheme account, up to 10% of salary, with a matching deduction
  for the Government's own contribution — the statutory basis for the NPS deduction.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q49 | Direct | `insertion of new section 80CCD Income-tax Act deduction pension scheme` | `act_section` | Gold `doc_id` in top 5 |
| Q50 | Indirect | `Which section lets an employee claim a deduction for contributing to a notified pension scheme like NPS?` | `act_section` | Gold `doc_id` in top 10 |

### Pair 26 — notifying a recognised association for the speculative-transaction exception (Income-tax Rules)

Gold document:

- `doc_id`: `103120000000009655`
- Source: Income-tax Rules, 1962, rule 6DDD, "Notification of a recognised association
  for the purposes of clause (e) of the proviso to clause (5) of section 43"
- Content: an association seeking notification (to bring its members' derivative trades
  within the section 43(5) proviso exception, i.e. out of "speculative transaction")
  applies to the Member (Income Tax), CBDT, with FMC trading approval, its rules/bye-laws,
  and confirmation of the rule 6DDC conditions; the Central Government then notifies or
  rejects the application within four months of the month-end.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q51 | Direct | `rule 6DDD notification of recognised association clause (e) proviso clause (5) section 43` | `rule_section` | Gold `doc_id` in top 5 |
| Q52 | Indirect | `What is the procedure for an association to get itself notified as recognised for the purposes of the speculative transaction exception?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 27 — fair market value of property other than immovable property (Income-tax Rules)

Gold document:

- `doc_id`: `103120000000024961`
- Source: Income-tax Rules, 1962, rule 11UA, "Determination of fair market value"
- Content: for section 56 purposes, prescribes valuation methods for jewellery,
  archaeological collections/art, and other specified property (invoice value if bought
  from a registered dealer on the valuation date; registered-valuer report otherwise for
  higher-value items) — the sibling provision to rule 11UB, which values immovable
  property.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q53 | Direct | `Income-tax Rule 11UA determination of fair market value of property other than immovable property section 56` | `rule_section` | Gold `doc_id` in top 5 |
| Q54 | Indirect | `How is the fair market value of unquoted shares or property (not land/building) worked out for section 56 purposes, and is that the same rule that values immovable property under 11UB?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 28 — transactions requiring PAN to be quoted (Income-tax Rules)

Gold document:

- `doc_id`: `103120000000007541`
- Source: Income-tax Rules, 1962, rule 114B, "Transactions in relation to which permanent
  account number is to be quoted in all documents for the purpose of clause (c) of
  sub-section (5) of section 139A"
- Content: sets out (in a table) the categories of transactions — property/vehicle sales,
  bank deposits and account opening, hotel/foreign-travel payments, and other high-value
  dealings — that require quoting PAN.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q55 | Direct | `rule 114B transactions requiring permanent account number to be quoted section 139A(5)(c)` | `rule_section` | Gold `doc_id` in top 5 |
| Q56 | Indirect | `Which transactions require quoting PAN on the documents, under the Income-tax Rules?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 29 — certificate for claiming DTAA relief (Income-tax Rules)

Gold document:

- `doc_id`: `103120000000007365`
- Source: Income-tax Rules, 1962, rule 21AB, "Certificate for claiming relief under an
  agreement referred to in sections 90 and 90A"
- Content: for sections 90(5)/90A(5), a non-resident assessee must furnish prescribed
  information (status, nationality/place of incorporation, foreign tax-ID, residency
  period, foreign address) in Form No. 10F, supplementing the tax-residency certificate
  under section 90(4)/90A(4), unless that certificate already contains the information.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q57 | Direct | `rule 21AB certificate for claiming relief under agreement sections 90 90A tax residency` | `rule_section` | Gold `doc_id` in top 5 |
| Q58 | Indirect | `What certificate does a non-resident need to claim DTAA relief, and which rule prescribes its form?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 30 — FEMA scheme for ADR/GDR issuance ("Euro Issue")

Gold document:

- `doc_id`: `103120000000000010`
- Source: FEMA circular/scheme text, "Euro Issue" — guidelines for Indian companies
  issuing GDRs, FCCBs and ordinary shares to international investors via the Depository
  Receipt Mechanism (per the 12 November 1993 notification)
- Content: sets out the restrictive policy on FCCBs (as external debt until conversion),
  treatment of Euro Issues as direct foreign investment requiring FIPB clearance above
  51%, and the one-issue-per-company-per-year limit; downstream provisions cover
  custodian/company-secretary verification with NSDL/CDSL of aggregate non-resident
  holding against the sectoral investment cap.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q59 | Direct | `FEMA ADR GDR scheme custodian verification with company secretary NSDL CDSL total cap breach` | `rule_section` | Gold `doc_id` in top 5 |
| Q60 | Indirect | `Under FEMA, who verifies whether the overall sectoral investment cap is being breached when ADRs/GDRs are issued to non-residents?` | `rule_section` | Gold `doc_id` in top 10 |

### Pair 31 — launch of the faceless assessment scheme (article)

Gold document:

- `doc_id`: `105010000000018368`
- Source: Taxmann Advisory & Research Team (Income Tax) article analysing the "Transparent
  Taxation – Honoring the Honest" platform, incorporating CBDT Notification 60/2020 dated
  13-08-2020
- Content: covers the Prime Minister's 13 August 2020 launch of Faceless Assessment,
  Faceless Appeal, and the Taxpayers' Charter — replacing manual scrutiny/best-judgment/
  income-escaping/search assessments (sections 143(3)/144/147/153A) with an e-governed,
  no-human-interface process.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q61 | Direct | `faceless assessment scheme launch August 2020 Transparent Taxation Prime Minister taxmann advisory` | `article_section` | Gold `doc_id` in top 5 |
| Q62 | Indirect | `What scheme did the government launch to remove human interface between the tax officer and the assessee during scrutiny?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 32 — ESOP cost as deductible business expenditure (article)

Gold document:

- `doc_id`: `105010000000001884`
- Source: "ESOP Benefit - An Employee Welfare Expenditure Allowable as Business
  Expenditure" by V. Prabhakar, `[2004] 136 Taxman 61 (Art.)`
- Content: argues the expenditure/loss a company incurs on ESOP/ESOS benefits granted to
  employees (i.e. shares issued below intrinsic/market value) is an employee-welfare
  measure allowable as business expenditure, not capital in nature — drawing on cases
  allowing welfare-store losses and jubilee-gift expenditure as business expenditure.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q63 | Direct | `ESOP benefit employee welfare expenditure allowable business expenditure V Prabhakar tax literature` | `article_section` | Gold `doc_id` in top 5 |
| Q64 | Indirect | `Is the cost of an employee stock option scheme treated as a deductible business expense for the company?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 33 — amalgamation and minimum alternate tax (article)

Gold document:

- `doc_id`: `105010000000002700`
- Source: "Amalgamation Put under MAT!" by Chythanya K.K., `[2004] 141 Taxman 104 (Art.)`
- Content: examines whether an amalgamated/merged company can carry forward the
  amalgamating company's business loss or unabsorbed depreciation while computing book
  profit for section 115JB MAT purposes, drawing on ICAI Accounting Standards and
  *Apollo Tyres Ltd. v. CIT* [2002] 255 ITR 273.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q65 | Direct | `amalgamation put under MAT Chythanya minimum alternate tax taxation of companies article` | `article_section` | Gold `doc_id` in top 5 |
| Q66 | Indirect | `When two companies amalgamate, does the resulting entity get pulled into minimum alternate tax on the scheme?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 34 — Vivad se Vishwas Act preamble and objectives (article)

Gold document:

- `doc_id`: `105010000000017382`
- Source: article by J.V. Kodhandapani (FCA) and Venkatesh K. Pani (Advocate) on the
  Direct Tax Vivad se Vishwas Act, 2020, opening with the Act's preamble/statement of
  objects and reasons
- Content: explains that the Finance Minister introduced the Act to address the rising
  pendency of tax appeals (appeals filed outpacing appeals disposed, locking up large
  disputed-tax arrears) by letting taxpayers settle pending direct-tax disputes for a
  reduced amount.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q67 | Direct | `Direct Tax Vivad se Vishwas Act 2020 preamble statement of objects and reasons finance minister` | `article_section` | Gold `doc_id` in top 5 |
| Q68 | Indirect | `What scheme let taxpayers settle pending direct tax disputes by paying a reduced amount, and why was it introduced?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 35 — equalisation levy and the OECD digital-economy work (article)

Gold document:

- `doc_id`: `105010000000013171`
- Source: article by Vineet Sodhani (CA) and Deepshikha Sodhani (CA) on Union Budget
  2016-17 (presented 29-2-2016), covering the equalisation levy on digital transactions
- Content: introduces the equalisation levy on payments to non-residents for specified
  digital services, situating it against the OECD's BEPS Action 1 work on the digital
  economy and India's enhanced-observer engagement with the OECD's technical advisory
  work.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q69 | Direct | `equalisation levy OECD technical advisory group India enhanced observer status digital economy` | `article_section` | Gold `doc_id` in top 5 |
| Q70 | Indirect | `What levy did India introduce on digital transactions with foreign companies, and how does that relate to OECD's work on the digital economy?` | `article_section` | Gold `doc_id` in top 10 |

### Pair 36 — validity of acts of the board of directors (Companies Act commentary)

Gold document:

- `doc_id`: `107010000000334155`
- Source: commentary on section 290, Companies Act — "Validity of the acts of the board
  of directors [Section 290]"
- Content: explains that section 290's validation provision (an application of the
  doctrine of indoor management — a director's acts remain valid even if the appointment
  is later found defective/terminated) is an exception, not the rule, and does not cover
  a total absence of appointment or a fraudulent usurpation of authority, citing
  *M. Moorthy v. Drivers & Conductors Bus Service (P.) Ltd.* (1991) 71 Comp. Cas. 136
  (Mad.) and *Col. Kuldip Singh Dhillon v. Paragaon Utility Financiers (P.) Ltd.* (1988)
  64 Comp. Cas. 19 (Punj. & Har.).

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q71 | Direct | `validity of acts of board of directors section 290 Companies Act validation provision exception not rule` | `commentary_section` | Gold `doc_id` in top 5 |
| Q72 | Indirect | `If a company later discovers a director's appointment was defective, are the board decisions that director took part in still valid?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 37 — jurisdiction of the Company Court (Companies Act commentary)

Gold document:

- `doc_id`: `107010000000333765`
- Source: commentary — "Administration of Law and Justice: Company Court", under the
  Companies Act
- Content: explains the Company Court is primarily the High Court of the state where the
  company's registered office sits (or a jurisdictional District Court where the Central
  Government has conferred that jurisdiction under section 10(1)(b)), and traces the
  subsequent shift of company-law adjudication to the NCLT/NCLAT, replacing the Company
  Law Board, with appeal from NCLAT lying to the Supreme Court on questions of law.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q73 | Direct | `Company Court primarily High Court of state registered office administration of law and justice winding up` | `commentary_section` | Gold `doc_id` in top 5 |
| Q74 | Indirect | `Which court has jurisdiction over a company's winding-up petition, and is that the same court hearing NCLT/NCLAT matters?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 38 — SEBI's power to issue directions under section 12A (SEBI commentary)

Gold document:

- `doc_id`: `107010000000334000`
- Source: commentary — "SEBI's power to issue directions", on section 12A (inserted by
  the Securities Laws (Amendment) Act, 2004, effective 12-10-2004)
- Content: explains SEBI's sweeping power under section 12A to direct stock
  exchanges/clearing corporations/agencies/persons in the securities market, or listed
  companies, in the interest of orderly market development or investor protection —
  exercisable only after SEBI makes or causes an inquiry.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q75 | Direct | `SEBI power to issue directions section 12A Securities Laws Amendment Act 2004 sweeping power` | `commentary_section` | Gold `doc_id` in top 5 |
| Q76 | Indirect | `What power lets SEBI issue directions to protect investors under section 12A - and is that the same section 12A that governs a charitable trust's income-tax registration?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 39 — taxable event and supply by a court receiver (GST commentary)

Gold document:

- `doc_id`: `107010000000371807`
- Source: commentary — "Taxable event in GST", covering the meaning of 'taxable event'
  and 'supply' under Article 366(12A) of the Constitution, citing *Goodyear India Ltd.
  v. State of Haryana* (1990) 76 STC 71 (SC) and *State of Kerala v. Alex George* (2004)
  AIR SCW 6552
- Content: a large commentary section explaining that GST's taxable event is 'supply of
  goods or services or both' (not sale or manufacture); within its broader discussion it
  covers when activities carried on by a court-appointed receiver running a business
  count as taxable supply under Schedule III/CGST Act.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q77 | Direct | `court receiver GST supply Schedule III CGST Act business activities liable to pay tax` | `commentary_section` | Gold `doc_id` in top 5 |
| Q78 | Indirect | `If a court-appointed receiver runs a business, do the activities count as a taxable supply under GST?` | `commentary_section` | Gold `doc_id` in top 10 |

### Pair 40 — rectification of the register of members (Companies Act commentary)

Gold document:

- `doc_id`: `107010000000372159`
- Source: commentary — "Rectification of Register of Members", on section 59, Companies
  Act, 2013 (corresponding to section 111(4) of the 1956 Act)
- Content: explains that a person wrongly entered in, or removed from, the register of
  members (or where entry of membership status is unnecessarily delayed) can seek
  rectification from the NCLT, on application by the aggrieved person, any member, or the
  company itself; also covers rectification where a transfer of securities violated SCRA,
  the SEBI Act, or the Companies Act — e.g. where allotment consideration (a share-payment
  cheque) was dishonoured.

| ID | Class | User query | Collection | Pass criterion |
|---|---|---|---|---|
| Q79 | Direct | `rectification of register deleting name of members dishonoured cheque share allotment company law` | `commentary_section` | Gold `doc_id` in top 5 |
| Q80 | Indirect | `Can a company remove a member's name from its share register if the cheque for the shares bounced?` | `commentary_section` | Gold `doc_id` in top 10 |

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
| Q41 | | |
| Q42 | | |
| Q43 | | |
| Q44 | | |
| Q45 | | |
| Q46 | | |
| Q47 | | |
| Q48 | | |
| Q49 | | |
| Q50 | | |
| Q51 | | |
| Q52 | | |
| Q53 | | |
| Q54 | | |
| Q55 | | |
| Q56 | | |
| Q57 | | |
| Q58 | | |
| Q59 | | |
| Q60 | | |
| Q61 | | |
| Q62 | | |
| Q63 | | |
| Q64 | | |
| Q65 | | |
| Q66 | | |
| Q67 | | |
| Q68 | | |
| Q69 | | |
| Q70 | | |
| Q71 | | |
| Q72 | | |
| Q73 | | |
| Q74 | | |
| Q75 | | |
| Q76 | | |
| Q77 | | |
| Q78 | | |
| Q79 | | |
| Q80 | | |

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
