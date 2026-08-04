// src/components/OverviewCard.tsx
import { useMemo, useState } from 'react'
import { parseCitations } from '../lib/citations'
import type { AiModeResult } from '../api/useSearch'
import styles from './OverviewCard.module.css'

export interface OverviewCardProps {
  aiMode: AiModeResult | null
  loading: boolean
  onCitationClick: (docId: string) => void
}

export default function OverviewCard({ aiMode, loading, onCitationClick }: OverviewCardProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false)
  const parsed = useMemo(() => {
    if (!aiMode?.ok || !aiMode.answer) return null
    return parseCitations(aiMode.answer)
  }, [aiMode])

  if (loading && !aiMode) {
    return (
      <section className={styles.card} aria-busy="true" data-testid="overview-loading">
        <h2>Overview</h2>
        <div className={styles.skeleton} />
      </section>
    )
  }

  if (!aiMode) {
    return null
  }

  if (!aiMode.ok) {
    return (
      <section className={styles.card}>
        <h2>Overview</h2>
        <p className={styles.error}>AI Mode is currently unavailable: {aiMode.error}</p>
      </section>
    )
  }

  return (
    <section className={styles.card}>
      <h2>Overview</h2>
      <p className={styles.answer}>
        {parsed?.segments.map((segment, index) =>
          segment.type === 'text' ? (
            <span key={index}>{segment.text}</span>
          ) : (
            <span key={index} className={styles.citationGroup}>
              {segment.numbers.map((n, i) => (
                <sup key={i} className={styles.pill}>
                  {n}
                </sup>
              ))}
            </span>
          ),
        )}
      </p>
      {parsed && parsed.citations.length > 0 && (
        <div className={styles.chipRow}>
          {parsed.citations.map((citation) => (
            <button
              key={citation.doc_id}
              type="button"
              className={styles.chip}
              onClick={() => onCitationClick(citation.doc_id)}
            >
              {citation.number}. {citation.doc_id} ({citation.count})
            </button>
          ))}
        </div>
      )}
      <button type="button" className={styles.reasoningToggle} onClick={() => setReasoningOpen((open) => !open)}>
        {reasoningOpen ? 'Hide' : 'Show'} detailed reasoning
      </button>
      {reasoningOpen && <p className={styles.reasoningPlaceholder}>Reasoning trace coming soon.</p>}
    </section>
  )
}
