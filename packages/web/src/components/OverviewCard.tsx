// src/components/OverviewCard.tsx
import { useMemo, useState } from 'react'
import { parseCitations } from '../lib/citations'
import { groupIntoParagraphs, renderInlineText, splitPlainTextIntoParagraphs } from '../lib/richText'
import type { AiModeResult } from '../api/useSearch'
import { extractPartyName } from './DocumentCard'
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
  const paragraphs = useMemo(() => (parsed ? groupIntoParagraphs(parsed.segments) : []), [parsed])
  const reasoning = aiMode?.ok ? aiMode.reasoning : null

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
      {paragraphs.map((paragraph, pIndex) => (
        <p key={pIndex} className={styles.answer}>
          {paragraph.map((segment, index) =>
            segment.type === 'text' ? (
              renderInlineText(segment.text, `${pIndex}-${index}`)
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
      ))}
      {parsed && parsed.citations.length > 0 && (
        <div className={styles.chipRow}>
          {parsed.citations.map((citation) => {
            const label = extractPartyName(aiMode.ok ? aiMode.citations?.[citation.doc_id] : undefined) ?? citation.doc_id
            return (
              <button
                key={citation.doc_id}
                type="button"
                className={styles.chip}
                onClick={() => onCitationClick(citation.doc_id)}
              >
                {citation.number}. {label} ({citation.count})
              </button>
            )
          })}
        </div>
      )}
      {reasoning && (
        <>
          <button type="button" className={styles.reasoningToggle} onClick={() => setReasoningOpen((open) => !open)}>
            {reasoningOpen ? 'Hide' : 'Show'} detailed reasoning
          </button>
          {reasoningOpen &&
            splitPlainTextIntoParagraphs(reasoning).map((paragraph, index) => (
              <p key={index} className={styles.reasoning}>
                {renderInlineText(paragraph, `reasoning-${index}`)}
              </p>
            ))}
        </>
      )}
    </section>
  )
}
