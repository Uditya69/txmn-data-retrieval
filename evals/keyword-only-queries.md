# Keyword-only probe: ES BM25 exact-match vs Milvus dense

`keyword_only_cases.json` is the machine-readable source of truth (regenerated
live from ES by `evals/keyword_only_probe.py` — do not hand-edit); this file
mirrors it for humans. Unlike the other eval sets, this one is intentionally
disposable/regenerable rather than hand-curated.

## Why this set exists

The other datasets (`retrieval_cases.json`, `statutory_cases.json`) use
multi-word queries — case names, fact patterns, distinctive terms of art —
which give a dense embedding enough signal to work with. This set asks the
opposite question: what happens on a **bare 1-3 word statutory reference**
(`section 55`, `rule 6`, `80HH`) with no other context? That's exactly what
BM25 is built for (literal term/heading match) and exactly what a dense
embedding struggles with — the query carries almost no distinguishing
semantic content, so cosine similarity has little to grab onto among
thousands of near-duplicate "Section - N" / "Rule - N" chunks.

## Gold definition (different from the other datasets)

These queries are inherently ambiguous — many docs share the literal heading
(e.g. "Section - 54" appears verbatim across multiple statutes/editions), and
ES ties the top few within noise on score. So **gold is the top-3 ES-ranked
`doc_id`s** for that exact query (confirmed live against ES on 2026-08-25),
not a single hand-verified doc. Any of the three counts as correct.

## Result (2026-08-25, live run)

Milvus dense-only search (query_embed → Voyage, straight `dense_vector`
search across all 11 collections — no ES, no sparse, no rerank, no RRF):
**0/8 hit `pass_at=5`.**

| ID | Query | ES top-1 doc_id | ES top-1 heading | Milvus dense rank |
|---|---|---|---|---|
| K01 | `section 54` | 102120000000080277 | Section - 54 | >20 |
| K02 | `section 55` | 102120000000010223 | Section - 55 | >20 |
| K03 | `section 43B` | 102120000000049876 | Section - 43B | 40 |
| K04 | `section 61` | 102120000000009238 | Section - 61 | >20 |
| K05 | `section 271` | 107010000000362901 | SECTION 271 | >20 |
| K06 | `rule 6` | 103120000000051010 | Rule - 6 | >20 |
| K07 | `article 14` | 105010000000016313 | [2019] 103 taxmann.com 384 (Article) | >20 |
| K08 | `80HH` | 102120000000054401 | Section - 80HH | >20 |

`>20` means the gold doc did not appear anywhere in the merged, deduped
Milvus-dense result list (up to 20 hits × 11 collections) for that query.

## Caveat on K07 (`article 14`)

The ES top hits for `article 14` are **commentary articles** (content type
"Article" — a magazine-style write-up), not the Constitution's Article 14.
The live corpus overloads the word "article" between two unrelated senses
(constitutional article vs. commentary-article content type), and BM25 picks
the commentary sense here purely on term frequency. This is a corpus/query
ambiguity, not a retrieval bug — flagged so it isn't misread as a second
independent data point for "dense misses obvious things."

## Regenerating

```
uv run python evals/keyword_only_probe.py --gateway-url http://localhost:8001
```

Rewrites `keyword_only_cases.json`, `keyword_only_results.json`, and
`keyword_only_results.csv`. Requires the model-gateway container running
(`docker compose up -d model-gateway`) and live ES/Milvus connectivity from
`.env`.
