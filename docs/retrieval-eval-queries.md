# Case-law retrieval evaluation set

This is a diagnostic retrieval benchmark built from the source case-law JSON files in
`/Users/uditya/dev/taxmann/data-extraction-pipeline/data`. It contains 20 queries grouped
into 10 matched pairs. Each pair targets the same gold document once with direct lexical
signals and once through an indirect factual or legal paraphrase.

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

