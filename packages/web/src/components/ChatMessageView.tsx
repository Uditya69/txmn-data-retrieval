import { useMemo, useState } from 'react'
import type { ChatMessage, ChatMode, ResultState } from '../types'
import { mergeResults, type CardSource } from '../lib/mergeResults'
import { parseCitations } from '../lib/citations'
import { groupIntoParagraphs, renderInlineText } from '../lib/richText'
import TracePanel from './TracePanel'

const SOURCE_FILTERS: { source: CardSource; label: string }[] = [
  { source: 'es', label: 'ES' },
  { source: 'milvus_dense', label: 'Milvus dense' },
  { source: 'milvus_sparse', label: 'Milvus sparse' },
]

type Props = {
  message: ChatMessage
  devMode: boolean
  onOpenDocument: (docId: string) => void
}

const SCROLL_PANE = 'max-h-[65vh] overflow-y-auto'

function LoadingDots() {
  return (
    <span className="inline-flex gap-1 items-center h-5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full animate-bounce"
          style={{ background: 'var(--text-faint)', animationDelay: `${i * 120}ms` }}
        />
      ))}
    </span>
  )
}

function TraceSection({ result, onOpenDocument }: { result: ResultState | undefined; onOpenDocument: (docId: string) => void }) {
  if (!result || result.traceSteps.length === 0) return null
  return (
    <details className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border-soft)' }}>
      <summary className="text-xs font-medium uppercase tracking-wider cursor-pointer" style={{ color: 'var(--text-faint)' }}>
        Trace ({result.traceSteps.length})
      </summary>
      <div className="mt-2">
        <TracePanel steps={result.traceSteps} onOpenDocument={onOpenDocument} />
      </div>
    </details>
  )
}

function InstantPane({ result, devMode, onOpenDocument }: { result: ResultState | undefined; devMode: boolean; onOpenDocument: (docId: string) => void }) {
  const instant = result?.instant
  const allCards = useMemo(
    () => mergeResults(instant?.es, instant?.milvus, instant?.milvus_sparse),
    [instant],
  )
  const [activeSources, setActiveSources] = useState<Set<CardSource>>(
    () => new Set(SOURCE_FILTERS.map((f) => f.source)),
  )
  const cards = devMode ? allCards.filter((card) => activeSources.has(card.source)) : allCards

  function toggleSource(source: CardSource) {
    setActiveSources((prev) => {
      const next = new Set(prev)
      if (next.has(source)) next.delete(source)
      else next.add(source)
      return next
    })
  }

  return (
    <div className={`flex-1 min-w-0 rounded-2xl px-4 py-3 ${SCROLL_PANE}`} style={{ background: 'var(--surface-raised)' }}>
      <div className="flex items-center gap-2 mb-3">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--text-faint)' }} />
        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
          Instant matches
        </span>
      </div>

      {devMode && instant && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {SOURCE_FILTERS.map(({ source, label }) => {
            const active = activeSources.has(source)
            return (
              <button
                key={source}
                onClick={() => toggleSource(source)}
                className="text-xs px-2 py-0.5 rounded-full uppercase tracking-wide transition-colors duration-150 cursor-pointer"
                style={{
                  background: active ? 'var(--accent-soft)' : 'var(--surface)',
                  color: active ? 'var(--accent)' : 'var(--text-faint)',
                  border: '1px solid var(--border-soft)',
                }}
              >
                {label}
              </button>
            )
          })}
        </div>
      )}

      {!instant && <LoadingDots />}
      {instant && cards.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>No matches.</p>
      )}
      {cards.length > 0 && (
        <div className="flex flex-col gap-2">
          {cards.map((card, index) => (
            <button
              key={`${card.source}-${card.doc_id}-${index}`}
              onClick={() => onOpenDocument(card.doc_id)}
              className="w-full text-left rounded-lg p-3 transition-colors duration-150 cursor-pointer"
              style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>
                  {card.heading ?? card.doc_id}
                </span>
                {devMode && (
                  <span className="text-xs shrink-0 font-mono" style={{ color: 'var(--text-faint)' }}>
                    {card.score.toFixed(3)}
                  </span>
                )}
              </div>
              {devMode && (
                <div className="flex items-center gap-2 mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                  <span className="uppercase tracking-wide px-1.5 py-0.5 rounded" style={{ background: 'var(--surface-raised)' }}>
                    {card.source === 'es'
                      ? 'ES'
                      : `Milvus ${card.source === 'milvus_dense' ? 'dense' : 'sparse'}:${card.collection}`}
                  </span>
                  <span className="font-mono truncate">{card.doc_id}</span>
                </div>
              )}
              <p className="text-sm mt-2 line-clamp-3" style={{ color: 'var(--text-muted)' }}>{card.snippet}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function renderAnswer(answer: string, knownDocIds: Set<string>) {
  const parsed = parseCitations(answer, knownDocIds)
  const paragraphs = groupIntoParagraphs(parsed.segments)
  return { paragraphs, citations: parsed.citations }
}

// heading/subheading come from the citations dict the backend sends (classic mode
// only - AI Mode's fetch_citations pulls them straight from ES, see schemas.py's
// MASTERINFO_CITATION_FIELDS). Agent mode has no per-doc metadata, only doc_ids, so
// cards there fall back to showing the doc_id itself.
type CitationMeta = { heading?: string; subheading?: string }

function CitedDocsStrip({
  citations,
  metaByDocId,
  onOpenDocument,
}: {
  citations: { doc_id: string; number: number }[]
  metaByDocId: Record<string, CitationMeta>
  onOpenDocument: (docId: string) => void
}) {
  if (citations.length === 0) return null
  return (
    <div className="flex gap-2 overflow-x-auto pb-3 mb-3" style={{ borderBottom: '1px solid var(--border-soft)' }}>
      {citations.map((cite) => {
        const meta = metaByDocId[cite.doc_id]
        const title = meta?.heading || cite.doc_id
        return (
          <button
            key={cite.doc_id}
            onClick={() => onOpenDocument(cite.doc_id)}
            className="text-left shrink-0 w-48 rounded-lg p-2.5 transition-colors duration-150 cursor-pointer"
            style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span
                className="h-4 min-w-4 px-1 flex items-center justify-center rounded text-[10px] font-semibold"
                style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}
              >
                {cite.number}
              </span>
              <span className="text-xs font-mono truncate" style={{ color: 'var(--text-faint)' }}>{cite.doc_id}</span>
            </div>
            <p className="text-xs leading-snug line-clamp-3" style={{ color: 'var(--text)' }}>{title}</p>
          </button>
        )
      })}
    </div>
  )
}

function AnswerPane({
  mode,
  result,
  devMode,
  onOpenDocument,
}: {
  mode: ChatMode
  result: ResultState | undefined
  devMode: boolean
  onOpenDocument: (docId: string) => void
}) {
  const status = result?.status ?? 'loading'
  const answerText = mode === 'classic' ? (result?.aiMode?.ok ? result.aiMode.answer : '') : result?.agent?.ok ? result.agent.answer : ''
  const errorText = mode === 'classic' ? (result?.aiMode && !result.aiMode.ok ? result.aiMode.error : null) : result?.agent && !result.agent.ok ? result.agent.error : null
  const docIds = mode === 'agent' && result?.agent?.ok ? result.agent.docIds : []
  // DB-sourced doc_id allowlist for this answer - citations dict (classic) and
  // docIds (agent) both come from ES/Milvus fetches, never from the LLM's own text.
  const knownDocIds = useMemo(() => {
    if (mode === 'agent') return new Set(docIds)
    return new Set(result?.aiMode?.ok ? Object.keys(result.aiMode.citations) : [])
  }, [mode, docIds, result])
  const metaByDocId = useMemo(() => {
    if (mode === 'agent' || !result?.aiMode?.ok) return {}
    const citations = result.aiMode.citations
    return Object.fromEntries(
      Object.keys(citations).map((docId) => {
        const meta = citations[docId] as CitationMeta
        return [docId, { heading: meta?.heading, subheading: meta?.subheading }]
      }),
    )
  }, [mode, result])
  const { paragraphs, citations } = answerText ? renderAnswer(answerText, knownDocIds) : { paragraphs: [], citations: [] }

  return (
    <div className={`flex-1 min-w-0 rounded-2xl px-4 py-3 ${SCROLL_PANE}`} style={{ background: 'var(--surface-raised)' }}>
      <div className="flex items-center gap-2 mb-2">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: 'var(--accent)', boxShadow: status === 'loading' ? '0 0 8px var(--accent)' : 'none' }}
        />
        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--accent)' }}>
          {mode === 'agent' ? 'Agent' : 'Answer'}
        </span>
      </div>

      {status === 'loading' && !answerText && <LoadingDots />}

      {errorText && (
        <p className="text-sm rounded-lg p-3" style={{ color: 'var(--danger)', background: 'var(--danger-soft)' }}>
          {errorText}
        </p>
      )}

      {answerText && (
        <div>
          <CitedDocsStrip citations={citations} metaByDocId={metaByDocId} onOpenDocument={onOpenDocument} />
          {paragraphs.map((paragraph, pIndex) => (
            <p key={pIndex} className="text-[15px] leading-relaxed whitespace-pre-wrap mb-3" style={{ color: 'var(--text)' }}>
              {paragraph.map((segment, index) =>
                segment.type === 'text' ? (
                  renderInlineText(segment.text, `${pIndex}-${index}`)
                ) : (
                  <span key={index}>
                    {segment.numbers.map((n, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          const cite = citations.find((c) => c.number === n)
                          if (cite) onOpenDocument(cite.doc_id)
                        }}
                        className="inline-flex items-center justify-center h-5 min-w-5 px-1 mx-0.5 rounded text-xs font-semibold align-middle cursor-pointer transition-colors duration-150"
                        style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}
                      >
                        {n}
                      </button>
                    ))}
                  </span>
                ),
              )}
            </p>
          ))}
        </div>
      )}

      {devMode && <TraceSection result={result} onOpenDocument={onOpenDocument} />}
    </div>
  )
}

export function ChatMessageView({ message, devMode, onOpenDocument }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl px-4 py-2.5 text-[15px]" style={{ background: 'var(--accent-strong)', color: 'var(--accent-ink)' }}>
          {message.text}
        </div>
      </div>
    )
  }

  const result = message.results[message.activeMode]

  if (message.activeMode === 'classic') {
    return (
      <div className="flex justify-start w-full">
        <div className="w-full flex gap-4 min-w-0">
          <InstantPane result={result} devMode={devMode} onOpenDocument={onOpenDocument} />
          <AnswerPane mode="classic" result={result} devMode={devMode} onOpenDocument={onOpenDocument} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] w-full">
        <AnswerPane mode="agent" result={result} devMode={devMode} onOpenDocument={onOpenDocument} />
      </div>
    </div>
  )
}
