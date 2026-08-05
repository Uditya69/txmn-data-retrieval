import { useMemo } from 'react'
import { mergeResults, type MergedCard } from '../lib/mergeResults'
import { parseCitations } from '../lib/citations'
import type { InstantResult, AiModeResult } from '../api/useSearch'
import DocumentCard from './DocumentCard'
import styles from './DocumentsFeed.module.css'

export interface DocumentsFeedProps {
  instant: InstantResult | null
  aiMode: AiModeResult | null
  devMode: boolean
  highlightedDocId: string | null
  onOpenDocument?: (docId: string) => void
}

function computeRelevance(cards: MergedCard[]): number[] {
  if (cards.length === 0) return []
  const scores = cards.map((c) => c.score)
  const min = Math.min(...scores)
  const max = Math.max(...scores)
  if (max === min) return cards.map(() => 100)
  return scores.map((s) => Math.round(((s - min) / (max - min)) * 100))
}

const MAX_RENDERED_CARDS = 20

export default function DocumentsFeed({ instant, aiMode, devMode, highlightedDocId, onOpenDocument }: DocumentsFeedProps) {
  const cards = useMemo(
    () =>
      mergeResults(instant?.es, instant?.milvus)
        .sort((a, b) => b.score - a.score)
        .slice(0, MAX_RENDERED_CARDS),
    [instant],
  )
  const relevance = useMemo(() => computeRelevance(cards), [cards])
  const citationCounts = useMemo(() => {
    if (!aiMode?.ok || !aiMode.answer) return new Map<string, number>()
    return new Map(parseCitations(aiMode.answer).citations.map((c) => [c.doc_id, c.count]))
  }, [aiMode])

  if (instant === null) {
    return <p className={styles.placeholder}>Search to see documents.</p>
  }
  if (cards.length === 0) {
    return <p className={styles.placeholder}>No results found.</p>
  }

  return (
    <section className={styles.feed}>
      <div className={styles.header}>
        <h2>Documents</h2>
        <span className={styles.count}>{cards.length}</span>
        <span className={styles.sort}>Relevance</span>
      </div>
      <ul className={styles.list}>
        {cards.map((card, index) => (
          <DocumentCard
            key={card.doc_id}
            card={card}
            citedCount={citationCounts.get(card.doc_id) ?? 0}
            citation={aiMode?.ok ? aiMode.citations?.[card.doc_id] : undefined}
            relevance={relevance[index]}
            devMode={devMode}
            highlighted={card.doc_id === highlightedDocId}
            onOpenDocument={onOpenDocument}
          />
        ))}
      </ul>
    </section>
  )
}
