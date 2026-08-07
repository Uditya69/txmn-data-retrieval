# Case-law retrieval evaluation set

This is a diagnostic retrieval benchmark built from the source case-law JSON files in
`/Users/uditya/dev/taxmann/data-extraction-pipeline/data`. It contains 53 queries grouped
into 21 matched pairs (`evals/retrieval_cases.json` is the machine-readable source of
truth; this document mirrors it). Pairs 1-10 target the same gold document twice: once
with direct lexical signals and once through an indirect factual or legal paraphrase.
Pairs 11-21 add a third leg per pair, `adversarial` - a noisy variant of the same fact
pattern (typos, telegraphic/Hinglish phrasing, acronym-only queries, or a bare compressed
fact contrast) that stresses retrieval under degraded query quality rather than clean
paraphrase.

The `expected strongest` column is a hypothesis, not ground truth. Its purpose is to make
failures interpretable when comparing Elasticsearch, Milvus sparse BM25, and Milvus dense
Voyage retrieval. Elasticsearch and Milvus results must be evaluated independently;
`doc_id` is only the join key and scores must not be fused across those systems.

## Suggested evaluation protocol

- Run every query independently through raw ES, Milvus sparse, and Milvus dense.
- Record the rank of the gold `doc_id` in each result list. Use `>50` when absent.
- For Milvus, also record which of the seven collections returned the first gold hit.
- Primary metric: gold-document Recall@5 and Recall@10 per retriever and query class.
- Secondary metric: reciprocal rank (`1 / rank`) and first-hit collection.
- Do not require the predicted strongest retriever to win. A disagreement is a result to
  investigate, not an automatic test failure.
- For a direct query, treat gold rank <= 5 as a pass and rank 6-10 as a weak pass.
- For an indirect query, treat gold rank <= 10 as a pass and rank 11-20 as a weak pass.
- For an adversarial query (pairs 11-21 only - typos, telegraphic/Hinglish phrasing,
  acronym-only, or bare fact contrast), treat gold rank <= 20 as a pass. There is no weak
  pass band; the query is deliberately degraded, so anything outside top 20 is a fail.
- All seven Milvus collections should be queried every time, in accordance with system
  behavior. `relevant collections` below means where a useful gold hit is most expected,
  not which collections should be routed.

## Query set

### Pair 1 — unexplained flat-rate best-judgment assessment

Gold document:

- `doc_id`: `101010000000039445`
- Case: *Rai Bahadur L. Panna Lal v. Commissioner of Income-tax*
- Citation: `[1927] 2 ITC 432 (Lahore)`
- Source: `data/1927/101010000000039445.json`
- Corpus evidence: A contractor's accounts for the relevant year could not be separated.
  The tax authority applied a standard 10 percent profit rate without disclosing its basis
  or giving the assessee a chance to rebut it; the court held that a question of law arose.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q01 | Direct | `Rai Bahadur L Panna Lal 2 ITC 432 Lahore standard rate 10 per cent assessment` | `metadata`, `headnotes`, `held` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q02 | Indirect | `Can a tax officer estimate a contractor's profit at a flat percentage when accounts for one year cannot be separated, without revealing the material used or allowing rebuttal?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: Q01 has rare party/citation tokens; Q02 removes them while preserving
the factual pattern and ratio.

### Pair 2 — compensation for giving up a managing agency

Gold document:

- `doc_id`: `101010000000078684`
- Case: *Commissioner of Income-tax v. Provident Investment Co. Ltd.*
- Citation: `[1957] 32 ITR 190 (SC)`
- Source: `data/1957/101010000000078684.json`
- Corpus evidence: The company changed an agreement and resigned its managing agency
  instead of transferring it. Compensation for that relinquishment was held not to be a
  capital gain because the transaction was neither a sale nor a transfer.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q03 | Direct | `32 ITR 190 Provident Investment managing agency section 12B capital gains` | `metadata`, `headnotes` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q04 | Indirect | `Is money received for resigning from a managing agency taxable as capital gains when the agency itself was never sold or transferred?` | `held`, `ruling`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: tests historical section 12B language against the modern capital-gains
concept and a semantic description of relinquishment.

### Pair 3 — CBEC circular and demurrage in customs valuation

Gold document:

- `doc_id`: `101010000000080330`
- Case: *Commissioner of Customs v. Indian Oil Corporation Ltd.*
- Citation: `[2004] 136 Taxman 491 (SC)`
- Source: `data/2004/101010000000080330.json`
- Corpus evidence: While a Board circular excluding demurrage from assessable value
  remained operative, Revenue was bound by it and could not argue that it contradicted
  the statute.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q05 | Direct | `Commissioner of Customs Indian Oil 136 Taxman 491 demurrage section 14 151A` | `metadata`, `headnotes` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q06 | Indirect | `Can the customs department ignore its own still-operative Board circular and add port delay charges to the assessable value of imported goods?` | `digest`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: Q06 replaces the corpus term “demurrage” with “port delay charges,” a
strong sparse-versus-dense discrimination test.

### Pair 4 — Modvat credit on an invoice addressed to head office

Gold document:

- `doc_id`: `101010000000113817`
- Case: *Gharda Chemicals Ltd. v. Commissioner of Central Excise, Mumbai*
- Citation: `2004 taxmann.com 889 (Mumbai - CESTAT)`
- Source: `data/2004/101010000000113817.json`
- Corpus evidence: Credit was admissible where the invoice was raised in the name of the
  head office, the Dombivli plant received the goods, and the head office endorsed the
  invoice to that plant.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q07 | Direct | `Gharda Chemicals Rule 57G Modvat invoice head office Dombivli plant` | `metadata`, `headnotes` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q08 | Indirect | `May a factory claim excise input credit when the supplier's invoice names the company's head office but the goods reached the factory and the office endorsed the invoice to it?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: tests legacy “Modvat” terminology versus the modern paraphrase “input
credit,” without turning the query into a GST question.

### Pair 5 — court-ordered investigation behind a fraudulent corporate veil

Gold document:

- `doc_id`: `101010000000017665`
- Case: *Ali Jawad Ameerhasan Rizvi v. Indo French Biotech Enterprises Ltd.*
- Citation: `[1998] 17 SCL 183 (Bombay)`
- Source: `data/1998/101010000000017665.json`
- Corpus evidence: Investors alleged that a company promising a 1025 percent return had
  caused the collected money to disappear. The court could direct investigation and
  search and seizure despite the absence of an express statutory provision.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q09 | Direct | `Ali Jawad Rizvi Indo French Biotech 1025 per cent return corporate veil` | `metadata`, `headnotes`, `facts` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q10 | Indirect | `Can a court order investigators to trace and seize a company's property when many investors appear to have been defrauded through an investment scheme, even if no statute expressly grants that power?` | `facts`, `held`, `ruling` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: long fact-pattern query with few exact case identifiers; useful for
testing dense retrieval over facts and operative holdings.

### Pair 6 — cheque liability of a non-signatory spouse

Gold document:

- `doc_id`: `101010000000198659`
- Case: *Alka Khandu Avhad v. Amar Syamprasad Mishra*
- Citation: `[2021] 128 taxmann.com 252 (SC)`
- Source: `data/2021/101010000000198659.json`
- Corpus evidence: A wife who neither signed the dishonoured cheque nor maintained a joint
  bank account could not be prosecuted under section 138 merely because the debt was
  jointly owed; section 141 concerning companies did not apply to individuals.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q11 | Direct | `Alka Khandu Avhad section 138 141 non signatory wife joint liability cheque` | `metadata`, `headnotes`, `held` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q12 | Indirect | `A husband issued a cheque for a jointly owed debt and it bounced. Can his wife also be criminally prosecuted when she did not sign it and the bank account was not joint?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: distinguishes liability for joint debt from the statutory requirement
that the accused draw the cheque on an account maintained by that person.

### Pair 7 — routine provisional attachment of nearly empty GST accounts

Gold document:

- `doc_id`: `101010000000197847`
- Case: *Vinodkumar Murlidhar Chechani v. State of Gujarat*
- Citation: `[2021] 123 taxmann.com 329 (Gujarat)`
- Source: `data/2021/101010000000197847.json`
- Corpus evidence: Provisional attachment under GST is a drastic measure requiring
  application of mind. Attaching two accounts containing only about Rs. 22,000 caused
  undue hardship, so those attachments were quashed.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q13 | Direct | `Vinodkumar Chechani section 83 rule 159 provisional attachment bank accounts 22000` | `metadata`, `headnotes`, `facts` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q14 | Indirect | `Should GST authorities freeze every bank account as a routine revenue-protection step when the accounts hold only a small balance and there is no demonstrated need for such a drastic measure?` | `held`, `ruling`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: Q14 omits party, section, rule, court, citation, and exact monetary
amount while retaining the ratio.

### Pair 8 — software licences recharged at cost under India-USA DTAA

Gold document:

- `doc_id`: `101010000000317237`
- Case: *Husco International Inc. v. ACIT (IT), Pune*
- Citation: `[2021] 133 taxmann.com 196 (Pune - Trib.)`
- Source: `data/2021/101010000000317237.json`
- Corpus evidence: A US company bought software licences and recharged some to its Indian
  entity at cost without markup. Limited rights to install and use copyrighted articles,
  without a right to copy, did not produce royalty; absent a PE, the fee was not taxable
  as business profits.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q15 | Direct | `Husco International 133 taxmann.com 196 article 12 India USA DTAA software royalty PE` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q16 | Indirect | `A US parent buys off-the-shelf software seats and recovers the exact cost from its Indian affiliate without giving any reproduction rights. Is that receipt royalty or taxable business income in India if the parent has no local establishment?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: tests semantic matching between “software seats/off-the-shelf” and the
corpus language “licences/copyrighted articles,” plus “local establishment” versus PE.

### Pair 9 — GST rate for an airport authority staff colony

Gold document:

- `doc_id`: `101010000000316021`
- Case: *B.G. Shirke Constructions Technology (P.) Ltd., In re*
- Citation: `[2021] 130 taxmann.com 199 (AAR - Karnataka)`
- Source: `data/2021/101010000000316021.json`
- Corpus evidence: Construction of a residential colony for Airport Authority of India
  staff and employees was held taxable at 12 percent under entry 3(vi)(c) of Notification
  No. 11/2017-Central Tax (Rate).

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q17 | Direct | `B G Shirke Airport Authority residential colony 12 percent GST Notification 11/2017 entry 3(vi)(c)` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q18 | Indirect | `What GST rate applies when a contractor builds staff housing for employees of the Airports Authority of India?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: a short natural-language query with a clear answer embedded in the
facts and holding; also tests “staff housing” versus “residential colony.”

### Pair 10 — unreliable segment data in transfer-pricing comparables

Gold document:

- `doc_id`: `101010000000327286`
- Case: *Dimension Data India Ltd. v. Additional Commissioner of Income-tax*
- Citation: `[2021] 128 taxmann.com 489 (Mumbai - Trib.)`
- Source: `data/2021/101010000000327286.json`
- Corpus evidence: A company not purely engaged in IT-enabled services, whose segmental
  results were unreliable, was not a valid comparable. A company outsourcing its work was
  also not comparable with an assessee performing services itself.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q19 | Direct | `Dimension Data India section 92C ITES comparables unreliable segmental results outsourcing` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q20 | Indirect | `For an arm's-length analysis, should a mixed-service company with untrustworthy segment accounts be compared with an IT-enabled-services provider, especially when one outsources the work and the other performs it in-house?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |

Diagnostic intent: concept-heavy query expected to benefit from dense similarity while
still retaining a few useful sparse anchors such as “arm's-length” and “outsources.”

### Pair 11 — court-approved capital reduction distributing a capitalised reserve

Gold document:

- `doc_id`: `101010000000015863`
- Case: *Shah Mohanlal Chhotalal v. Commissioner of Income-tax*
- Citation: `1936 10 ITC 46 (Bombay)`
- Source: `data/1936/101010000000015863.json`
- Corpus evidence: A company capitalised its reserve, bought shares in another company, then
  reduced its own capital by returning those shares to shareholders, with the reduction
  confirmed by a District Court. Revenue treated the whole transaction as illusory and the
  shares as taxable profit; the court held the department could not go behind the court-
  confirmed capital reduction, and the share value was not taxable income.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q21 | Direct | `Shah Mohanlal Chhotalal 10 ITC 46 Bombay section 4 bonus shares reduction of capital` | `metadata`, `headnotes`, `held` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q22 | Indirect | `Can the tax department disregard a court-approved reduction of share capital and treat shares distributed to shareholders out of a capitalised reserve as taxable profits?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q23 | Adversarial | `court aproved capital reduction bonus shars reserve can dept call it profit 1930s bombay` | `headnotes`, `facts`, `held` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q23 tests robustness to typos (“aproved”, “shars”) and telegraphic
compression of the same fact pattern as Q22, with no citation or party-name anchors.

### Pair 12 — managing agency income held as charitable trust property

Gold document:

- `doc_id`: `101010000000079607`
- Case: *J.K. Trust v. Commissioner of Income-tax, Excess Profits-tax*
- Citation: `[1957] 32 ITR 535 (SC)`
- Source: `data/1957/101010000000079607.json`
- Corpus evidence: Trustees who had settled funds on charities were later appointed managing
  agents of a company under the trust deed and claimed the managing-agency income was
  exempt as income from property held for charitable purposes. Revenue argued it was mere
  remuneration for services, not property income. The court held “property” was broad
  enough to include the managing agency, and the trustees’ option to surrender the agency
  did not disqualify it as trust property.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q24 | Direct | `J K Trust 32 ITR 535 Supreme Court managing agency charitable trust section 4(3)(i)` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q25 | Indirect | `Is remuneration earned by trustees from a managing agency income from property held for charitable purposes, even though they may surrender the agency?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q26 | Adversarial | `managing agency itself trust property charity exemption case` | `headnotes`, `held`, `case_summary` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q26 strips every identifier down to a bare eight-word topic phrase,
testing whether retrieval still surfaces the gold case from concept words alone.

### Pair 13 — domestic enquiry proceeding despite a pending criminal case

Gold document:

- `doc_id`: `101010000000411060`
- Case: *J.K. Cotton Spinning & Weaving Co. Ltd. v. The Workmen*
- Citation: `1965 taxmann.com 60 (SC)` (`1965 11 FLR 27 (SC)`)
- Source: `data/1965/101010000000411060.json`
- Corpus evidence: The question was whether an employer must await the outcome of a pending
  criminal case or appeal against a workman before proceeding with a domestic disciplinary
  enquiry. The court held natural justice does not require the employer to wait, and also
  limited an Industrial Tribunal’s power to re-examine a domestic tribunal’s factual
  findings.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q27 | Direct | `J K COTTON SPINNING AND WEAVING CO LTD v WORKMEN 1965 domestic enquiry criminal case pending` | `metadata`, `headnotes` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q28 | Indirect | `Must an employer postpone a domestic disciplinary enquiry until the employee's related criminal case or appeal has been decided?` | `held`, `ruling`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q29 | Adversarial | `criminal case pending hai toh company domestic enquiry rokna padega kya workman` | `headnotes`, `held`, `ruling` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q29 mixes Hindi/English (Hinglish) phrasing over the same fact pattern
as Q28, testing tokenization and embedding robustness to code-mixed colloquial queries.

### Pair 14 — constitutional challenge to the concealment-penalty provision

Gold document:

- `doc_id`: `101010000000031979`
- Case: *Rahimbhai Karimbhai Nagriwala v. B.B. Patel*
- Citation: `[1974] 97 ITR 660 (Gujarat)`
- Source: `data/1974/101010000000031979.json`
- Corpus evidence: The concealment-penalty provision, section 271(1)(c), was challenged as
  violating articles 14 and 19(1)(f) of the Constitution. The court held the State has wide
  discretion in selecting objects of taxation, the section did not single out a particular
  class of taxpayer, and the penalty was deterrent rather than confiscatory.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q30 | Direct | `Rahimbhai Karimbhai Nagriwala 97 ITR 660 section 271(1)(c) article 14 article 19 concealment penalty` | `metadata`, `headnotes`, `held` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q31 | Indirect | `Is the income-tax penalty for concealment unconstitutional merely because the State penalises some classes of taxpayers differently, or because the penalty is strongly deterrent?` | `held`, `case_summary`, `digest` | Milvus dense | Gold `doc_id` in top 10 |
| Q32 | Adversarial | `271 1 c consitutional validity hostile discrimnation confiscatory conceal income penalty gujrat` | `headnotes`, `held`, `metadata` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q32 injects misspellings into legal terms of art (“consitutional”,
“discrimnation”, “gujrat”) to test tolerance for noisy legal vocabulary.

### Pair 15 — backward-area deduction on scrap metal versus useless drums

Gold document:

- `doc_id`: `101010000000069794`
- Case: *Income-tax Officer v. Poly Tech Cable & Products (P.) Ltd.*
- Citation: `[1985] 11 ITD 20 (Hyderabad)`
- Source: `data/1985/101010000000069794.json`
- Corpus evidence: A manufacturer of metallic wires produced scrap metal as manufacturing
  waste and separately sold water-storage drums that had become useless. The section 80HH
  backward-area deduction was allowed on the scrap-metal profit as an incidental
  manufacturing byproduct, but denied on the unrelated drum-sale profit.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q33 | Direct | `ITO v Poly Tech Cable Products 11 ITD 20 section 80HH scrap metal drums backward area` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q34 | Indirect | `Can a new industrial undertaking claim the backward-area deduction on profits from manufacturing scrap as well as on proceeds from discarded water-storage drums?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q35 | Adversarial | `80HH scrap sale yes useless drum sale no metallic wire factory` | `headnotes`, `held`, `digest` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q35 compresses the case’s two-outcome holding (allowed for scrap,
disallowed for drums) into a terse yes/no contrast without full sentence structure.

### Pair 16 — Modvat credit on inputs cleared at a nil duty rate

Gold document:

- `doc_id`: `101010000000130933`
- Case: *Reliance Industries Ltd. v. Collector of Central Excise, Bombay*
- Citation: `1995 taxmann.com 569 (CEGAT - Mumbai)` (`1995 78 ELT 595`)
- Source: `data/1995/101010000000130933.json`
- Corpus evidence: A Modvat credit dispute over inputs PTA and MEG used to manufacture
  POY/PSF (polyester yarn and fibre) cleared at a nil duty rate under Rule 191B/191BB. The
  court held Modvat credit was not deniable and the nil-rate clearance did not trigger a
  Rule 57-C disallowance.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q36 | Direct | `Reliance Industries 1995 taxmann.com 569 Rule 57A 57C PTA MEG POY PSF Modvat nil rate` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q37 | Indirect | `Can input credit be denied for raw materials used to make polyester yarn and fibre merely because the finished products were cleared at a nil duty rate for export manufacture?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q38 | Adversarial | `PTA MEG to POY PSF nil rate 191B 191BB modvat allowed?` | `headnotes`, `held`, `metadata` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q38 is acronym-only, with none of the expanded chemical/product names
(purified terephthalic acid, mono ethylene glycol, polyester oriented yarn, polyester
staple fibre), testing whether retrieval depends on acronym expansion.

### Pair 17 — taking over a competing training business to ward off competition

Gold document:

- `doc_id`: `101010000000072121`
- Case: *Vinod Kothari Consultants Ltd. v. Deputy Commissioner of Income-tax*
- Citation: `[2004] 91 ITD 153 (Kolkata)`
- Source: `data/2004/101010000000072121.json`
- Corpus evidence: A financial-services training company took over a similarly-placed
  competitor’s business to eliminate competition and claimed the payment as revenue
  expenditure. The court held the payment was for taking over a business and warding off
  competition, so it was capital expenditure, not deductible under section 37(1).

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q39 | Direct | `Vinod Kothari Consultants 91 ITD 153 Kolkata takeover competing training business section 37(1)` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q40 | Indirect | `Is the price paid to take over a competitor's financial-services training business and eliminate competition a revenue expense or capital expenditure?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q41 | Adversarial | `trade name training course takeover ward off competition deductible expense case Kolkata` | `headnotes`, `held`, `ruling` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q41 includes “trade name” and “deductible expense” as mild distractor
phrasing not literal to the corpus text, testing resilience to near-miss vocabulary.

### Pair 18 — chartered accountant's transfer-pricing certificate as a taxable service

Gold document:

- `doc_id`: `101010000000003961`
- Case: *Price Waterhouse v. Commissioner of Service Tax*
- Citation: `[2010] 26 STT 291 (Chennai)` (`2010 19 STR 63`)
- Source: `data/2010/101010000000003961.json`
- Corpus evidence: A practising chartered accountant issued a certificate following a
  transfer-pricing audit; Revenue taxed it as an auditing/certification service and denied
  exemption under Notification No. 59/98-ST. The court held that since the certificate
  followed a transfer-pricing audit and verification of books precedes the certificate, it
  fell within “audit/certification services” and was taxable.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q42 | Direct | `Price Waterhouse 26 STT 291 Chennai transfer pricing certification Notification 59/98-ST chartered accountant service tax` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q43 | Indirect | `Does a chartered accountant's certificate issued after examining books for a transfer-pricing audit amount to taxable audit or certification service?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q44 | Adversarial | `PW transfer price certifcation audit notif 59 98 ST exemption CA Chennai` | `headnotes`, `held`, `metadata` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q44 abbreviates the party name to initials (“PW”) and misspells
“certification” as “certifcation,” testing tolerance to abbreviation and spelling noise
together.

### Pair 19 — pro-rata Cenvat credit for a commission agent without separate accounts

Gold document:

- `doc_id`: `101010000000175656`
- Case: *Ruchika Global Interlinks v. Customs, Excise & Service Tax Appellate Tribunal, Chennai*
- Citation: `[2017] 82 taxmann.com 480 (Madras)`
- Source: `data/2017/101010000000175656.json`
- Corpus evidence: For April 2006-2008, before Rule 2(e) was amended with effect from
  1-4-2011 to bring trading within “service,” a commission agent also engaged in trading
  did not maintain separate accounts for taxable and non-taxable services. The court held
  trading was not taxable as a service in that period, but since no separate accounts were
  kept, credit could only be taken pro rata under Rule 6(3)(c), so full credit was rightly
  denied.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q45 | Direct | `Ruchika Global Interlinks 82 taxmann.com 480 Rule 6(3)(c) Cenvat trading commission agent separate accounts` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q46 | Indirect | `Before trading was treated as an exempt service, could a commission agent engaged in both trading and taxable services take the entire input-service credit without maintaining separate accounts?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q47 | Adversarial | `trading not service before 1-4-2011 but no separate books pro rata cenvat credit Madras` | `headnotes`, `held`, `ruling` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: Q47 anchors only on the statutory amendment date (1-4-2011) rather than
the case name or citation, testing whether a bare date/rule combination surfaces the gold
document.

### Pair 20 — interest-free disbursal recalled seventeen years later under the IBC

Gold document:

- `doc_id`: `101010000000398668`
- Case: *Sunil Chopra v. CAPL Hotels and Spa (P.) Ltd.*
- Citation: `2025 175 taxmann.com 251 (NCLT - New Delhi)`
- Source: `data/2025/101010000000398668.json`
- Corpus evidence: A claimed financial creditor sought to recall an interest-free disbursal
  made with no stipulated interest, repayment terms, or contemporaneous loan documentation,
  nearly seventeen years after disbursal. The court held that absent the time-value-of-money
  element the disbursal was not a “financial debt” under section 5(8) IBC, and the section
  7 application was in any event time-barred.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q48 | Direct | `Sunil Chopra v CAPL Hotels Spa 175 taxmann.com 251 section 5(8) time value money 17 years limitation` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q49 | Indirect | `Can an interest-free payment with no repayment terms or contemporaneous loan documents qualify as financial debt under the IBC, especially when recalled seventeen years later?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q50 | Adversarial | `IBC s7 loan no interest no time value no agreement recalled after 17 yrs financial debt?` | `headnotes`, `held`, `ruling` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: tests both recency (a 2025 NCLT ruling) and telegraphic section
shorthand (“s7”) against the fully spelled-out statutory phrasing used in Q49.

### Pair 21 — unexplained-expenditure addition on genuine diamond purchases

Gold document:

- `doc_id`: `101010000000419651`
- Case: *Principal Commissioner of Income-tax v. Kross Diamonds (P.) Ltd.*
- Citation: `2026 186 taxmann.com 345 (Delhi)` (`2026 311 Taxman 40`)
- Source: `data/2026/101010000000419651.json`
- Corpus evidence: A diamond trader’s genuine import purchases, made through banking
  channels, were added back as unexplained expenditure under section 69C because the
  trader had made a large volume of cash sales to buyers who could not later be verified.
  The court held that since the purchases were genuine and properly documented, the source
  of funds was not unexplained merely because buyer identities in the cash sales could not
  be verified, and deleted the addition.

| ID | Class | User query | Relevant collections | Expected strongest | Pass criterion |
|---|---|---|---|---|---|
| Q51 | Direct | `PCIT v Kross Diamonds 186 taxmann.com 345 section 69C section 105 Income-tax Act 2025 cash sales 97 crores` | `metadata`, `headnotes`, `digest` | ES / Milvus sparse | Gold `doc_id` in top 5 |
| Q52 | Indirect | `Can genuine diamond purchases made through banking channels be treated as unexplained expenditure only because the stock was later sold for cash to buyers whose identities could not be verified?` | `facts`, `held`, `case_summary` | Milvus dense | Gold `doc_id` in top 10 |
| Q53 | Adversarial | `69C old act 105 new act diamond cash sale buyers unknown purchase genuine Delhi 2026` | `headnotes`, `held`, `metadata` | Milvus dense (robustness hypothesis) | Gold `doc_id` in top 20 |

Diagnostic intent: tests the section 69C (1961 Act) to section 105 (2025 Act) crosswalk
under a compressed, half-citation-style query.

## Result capture template

Copy this table for each evaluation run. `Gold collection` is the collection containing
the highest-ranked hit for the gold document.

| Query | ES rank | Milvus sparse rank | Sparse gold collection | Milvus dense rank | Dense gold collection | RRF rank | Notes |
|---|---:|---:|---|---:|---|---:|---|
| Q01 | | | | | | | |
| Q02 | | | | | | | |
| Q03 | | | | | | | |
| Q04 | | | | | | | |
| Q05 | | | | | | | |
| Q06 | | | | | | | |
| Q07 | | | | | | | |
| Q08 | | | | | | | |
| Q09 | | | | | | | |
| Q10 | | | | | | | |
| Q11 | | | | | | | |
| Q12 | | | | | | | |
| Q13 | | | | | | | |
| Q14 | | | | | | | |
| Q15 | | | | | | | |
| Q16 | | | | | | | |
| Q17 | | | | | | | |
| Q18 | | | | | | | |
| Q19 | | | | | | | |
| Q20 | | | | | | | |
| Q21 | | | | | | | |
| Q22 | | | | | | | |
| Q23 | | | | | | | |
| Q24 | | | | | | | |
| Q25 | | | | | | | |
| Q26 | | | | | | | |
| Q27 | | | | | | | |
| Q28 | | | | | | | |
| Q29 | | | | | | | |
| Q30 | | | | | | | |
| Q31 | | | | | | | |
| Q32 | | | | | | | |
| Q33 | | | | | | | |
| Q34 | | | | | | | |
| Q35 | | | | | | | |
| Q36 | | | | | | | |
| Q37 | | | | | | | |
| Q38 | | | | | | | |
| Q39 | | | | | | | |
| Q40 | | | | | | | |
| Q41 | | | | | | | |
| Q42 | | | | | | | |
| Q43 | | | | | | | |
| Q44 | | | | | | | |
| Q45 | | | | | | | |
| Q46 | | | | | | | |
| Q47 | | | | | | | |
| Q48 | | | | | | | |
| Q49 | | | | | | | |
| Q50 | | | | | | | |
| Q51 | | | | | | | |
| Q52 | | | | | | | |
| Q53 | | | | | | | |

## Reading failures

- Direct query succeeds in ES and Milvus sparse but paired indirect query fails in dense:
  inspect Voyage query embedding, vector field selection, and semantic chunk quality.
- Direct query succeeds in ES but fails in Milvus sparse: inspect the collection's BM25
  source text and tokenization; never construct `sparse_vector` client-side.
- Direct query succeeds in Milvus sparse but fails in ES: inspect ES field mapping,
  analyzers, field boosts, and whether the identifying evidence exists in indexed fields.
- Gold document appears in a Milvus collection but its best passage is in the wrong
  section: inspect extraction and chunk boundary classification.
- Both members of a pair fail everywhere: first verify ingestion/index presence by exact
  `doc_id`; do not infer a ranking defect from a missing document.
- Dense results are broadly plausible but consistently miss these gold documents: verify
  that `query_embed` resolved to Voyage. A provider mismatch can return valid vectors and
  silently destroy retrieval quality.

