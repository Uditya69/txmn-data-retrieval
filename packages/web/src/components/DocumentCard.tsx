// src/components/DocumentCard.tsx
import type { MergedCard } from '../lib/mergeResults'
import type { AiModeCitation } from '../api/useSearch'
import styles from './DocumentCard.module.css'

export interface DocumentCardProps {
  card: MergedCard
  citedCount: number
  citation?: AiModeCitation
  relevance: number
  devMode: boolean
  highlighted?: boolean
  onOpenDocument?: (docId: string) => void
}

function truncate(text: string, length: number): string {
  if (text.length <= length) return text
  return `${text.slice(0, length).trimEnd()}…`
}

export function extractPartyName(citation?: AiModeCitation): string | null {
  // fetch_citations() returns ES's raw nested _source shape (confirmed
  // against packages/common/tests/test_es_client.py), e.g.
  // { otherinfo: { partyname: ... } } - NOT a flat "otherinfo.partyname" key.
  const otherinfo = citation?.otherinfo
  const raw =
    otherinfo && typeof otherinfo === 'object' ? (otherinfo as Record<string, unknown>).partyname : undefined
  if (!raw) return null
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) {
    const names = raw
      .map((entry) => (entry && typeof entry === 'object' && 'name' in entry ? String((entry as { name: unknown }).name) : null))
      .filter((name): name is string => Boolean(name))
    return names.length > 0 ? names.join(' vs. ') : null
  }
  if (typeof raw === 'object' && 'name' in (raw as Record<string, unknown>)) {
    return String((raw as { name: unknown }).name)
  }
  return null
}

export default function DocumentCard({
  card,
  citedCount,
  citation,
  relevance,
  devMode,
  highlighted = false,
  onOpenDocument,
}: DocumentCardProps) {
  const partyName = extractPartyName(citation)
  const title = partyName ?? truncate(card.snippet, 80)

  return (
    <li
      id={`document-${card.doc_id}`}
      className={highlighted ? `${styles.card} ${styles.highlighted}` : styles.card}
      onClick={onOpenDocument ? () => onOpenDocument(card.doc_id) : undefined}
      style={onOpenDocument ? { cursor: 'pointer' } : undefined}
    >
      <div className={styles.headerRow}>
        <span className={styles.typeBadge}>Case Law</span>
        {citedCount > 0 && <span className={styles.citedBadge}>Cited {citedCount}</span>}
      </div>
      <h3 className={styles.title}>{title}</h3>
      {title !== card.snippet && <p className={styles.snippet}>{card.snippet}</p>}
      <div className={styles.footerRow}>
        <div className={styles.relevance}>
          <div className={styles.relevanceBar} style={{ width: `${relevance}%` }} />
          <span>{relevance}</span>
        </div>
        {devMode && (
          <span className={styles.devBadge}>
            {card.source === 'es' ? 'ES' : `Milvus:${card.collection}`} · score {card.score.toFixed(3)}
          </span>
        )}
      </div>
    </li>
  )
}
