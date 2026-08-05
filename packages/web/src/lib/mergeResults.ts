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

export interface MergedCard {
  doc_id: string
  source: 'es' | 'milvus'
  collection?: string
  score: number
  snippet: string
  heading?: string
}

/**
 * Presentation merge, not a ranking fusion: ES and Milvus scores live in
 * different spaces and are never blended or re-ranked against each other.
 * ES cards keep their own order; Milvus-only cards (deduped to their best
 * score per doc_id across collections) are appended after, in their own
 * first-seen order.
 */
export function mergeResults(
  es: EsHit[] | null | undefined,
  milvus: MilvusByCollection | null | undefined,
): MergedCard[] {
  const cards: MergedCard[] = []
  const seen = new Set<string>()

  for (const hit of es ?? []) {
    if (seen.has(hit.doc_id)) continue
    seen.add(hit.doc_id)
    cards.push({ doc_id: hit.doc_id, source: 'es', score: hit.score, heading: hit.heading, snippet: hit.subheading })
  }

  const bestMilvusByDocId = new Map<string, MergedCard>()
  for (const [collection, hits] of Object.entries(milvus ?? {})) {
    for (const hit of hits) {
      if (seen.has(hit.doc_id)) continue
      const existing = bestMilvusByDocId.get(hit.doc_id)
      if (!existing || hit.score > existing.score) {
        bestMilvusByDocId.set(hit.doc_id, {
          doc_id: hit.doc_id,
          source: 'milvus',
          collection,
          score: hit.score,
          snippet: hit.text,
        })
      }
    }
  }

  cards.push(...bestMilvusByDocId.values())
  return cards
}
