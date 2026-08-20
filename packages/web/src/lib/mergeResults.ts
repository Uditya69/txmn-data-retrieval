export interface EsHit {
  doc_id: string
  score: number
  heading: string
  subheading: string
}

export interface MilvusHit {
  chunk_id: string
  doc_id: string
  text: string
  score: number
}

export type MilvusByCollection = Record<string, MilvusHit[]>

export type CardSource = 'es' | 'milvus_dense' | 'milvus_sparse' | 'reranked'

export interface MergedCard {
  doc_id: string
  source: CardSource
  collection?: string
  score: number
  snippet: string
  heading?: string
}

// Backend rows in the rerank toggle's fused list keep whichever source's row shape
// won the doc_id (es_client.raw_search's heading/subheading, or milvus_client's
// chunk_id/text) - see instant/rerank.py::rrf_merge_by_doc_id. Normalized here so
// the UI doesn't need to care which source contributed a given card.
export interface RerankedHit {
  doc_id: string
  // Only rows that went through the cross-encoder carry rerank_score - RRF-only
  // (rerank toggle off) rows keep rrf_score, and RRF-off/rerank-off rows keep
  // whichever `score` ES/Milvus gave them. See instant/rerank.py::rerank_instant_results.
  rerank_score?: number
  rrf_score?: number
  score?: number
  heading?: string
  subheading?: string
  text?: string
}

export function mapRerankedResults(reranked: RerankedHit[] | null | undefined): MergedCard[] {
  return (reranked ?? []).map((hit) => ({
    doc_id: hit.doc_id,
    source: 'reranked',
    score: hit.rerank_score ?? hit.rrf_score ?? hit.score ?? 0,
    heading: hit.heading,
    snippet: hit.subheading ?? hit.text ?? '',
  }))
}

function bestPerDocId(
  milvus: MilvusByCollection | null | undefined,
  source: CardSource,
  seen: Set<string>,
): MergedCard[] {
  const bestByDocId = new Map<string, MergedCard>()
  for (const [collection, hits] of Object.entries(milvus ?? {})) {
    for (const hit of hits) {
      if (seen.has(hit.doc_id)) continue
      const existing = bestByDocId.get(hit.doc_id)
      if (!existing || hit.score > existing.score) {
        bestByDocId.set(hit.doc_id, { doc_id: hit.doc_id, source, collection, score: hit.score, snippet: hit.text })
      }
    }
  }
  return [...bestByDocId.values()]
}

/**
 * Presentation merge, not a ranking fusion: ES and Milvus scores live in
 * different spaces and are never blended or re-ranked against each other.
 * ES cards keep their own order; Milvus dense and sparse cards (each
 * deduped to their best score per doc_id within their own source) are
 * appended after, in their own first-seen order. Dense and sparse are kept
 * as independent sources, so a doc_id matched by both appears twice - this
 * lets dev mode filter/inspect each retriever's contribution separately.
 */
export function mergeResults(
  es: EsHit[] | null | undefined,
  milvusDense: MilvusByCollection | null | undefined,
  milvusSparse?: MilvusByCollection | null,
): MergedCard[] {
  const cards: MergedCard[] = []
  const seen = new Set<string>()

  for (const hit of es ?? []) {
    if (seen.has(hit.doc_id)) continue
    seen.add(hit.doc_id)
    cards.push({ doc_id: hit.doc_id, source: 'es', score: hit.score, heading: hit.heading, snippet: hit.subheading })
  }

  cards.push(...bestPerDocId(milvusDense, 'milvus_dense', seen))
  cards.push(...bestPerDocId(milvusSparse, 'milvus_sparse', seen))
  return cards
}
