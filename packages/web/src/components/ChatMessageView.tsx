import { useEffect, useMemo, useState, type MouseEvent } from 'react'
import type { ChatMessage, ChatMode, ResultState } from '../types'
import { mergeResults, mapRerankedResults, type CardSource, type MilvusByCollection } from '../lib/mergeResults'
import { parseCitations } from '../lib/citations'
import { groupIntoParagraphs, renderInlineText } from '../lib/richText'
import { highlightMatches } from '../lib/highlight'
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

// No inner height cap and no overflow-y-auto here on purpose - a fixed-height
// box always shows a hard-edge cutoff (plus its own scrollbar) once content
// exceeds it. Panes grow with their content instead; the page itself is the
// only scroll container, so scrolling down never hits a visible stop point
// before the real end of the content.
const SCROLL_PANE = ''

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

function CopyTraceButton({ traceSteps }: { traceSteps: ResultState['traceSteps'] }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = (e: MouseEvent) => {
    // Stop the click from also toggling the parent <details> open/closed -
    // the button lives inside <summary>, whose default behavior is exactly that.
    e.preventDefault()
    e.stopPropagation()
    navigator.clipboard.writeText(JSON.stringify(traceSteps, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="text-xs font-medium normal-case tracking-normal"
      style={{ color: 'var(--text-faint)', marginLeft: '0.5rem' }}
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function TraceSection({ result, onOpenDocument }: { result: ResultState | undefined; onOpenDocument: (docId: string) => void }) {
  if (!result || result.traceSteps.length === 0) return null
  return (
    <details className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border-soft)' }}>
      <summary className="text-xs font-medium uppercase tracking-wider cursor-pointer" style={{ color: 'var(--text-faint)' }}>
        Trace ({result.traceSteps.length})
        <CopyTraceButton traceSteps={result.traceSteps} />
      </summary>
      <div className="mt-2">
        <TracePanel steps={result.traceSteps} onOpenDocument={onOpenDocument} />
      </div>
    </details>
  )
}

// Instant mode pulls up to ~20 ES hits plus deduped best-per-doc hits across 7
// Milvus collections x2 retrievers - that can be 50+ cards. Paginating instead of
// dumping them all into one ever-growing column keeps the pane a fixed, predictable
// size instead of turning the whole page into a multi-thousand-pixel scroll.
const PAGE_SIZE = 10

function InstantPane({ result, devMode, onOpenDocument, query }: { result: ResultState | undefined; devMode: boolean; onOpenDocument: (docId: string) => void; query: string }) {
  const instant = result?.instant
  const isReranked = Boolean(instant?.reranked)
  const allCards = useMemo(
    () => (isReranked ? mapRerankedResults(instant?.reranked) : mergeResults(instant?.es, instant?.milvus, instant?.milvus_sparse)),
    [instant, isReranked],
  )
  const [activeSources, setActiveSources] = useState<Set<CardSource>>(
    () => new Set(SOURCE_FILTERS.map((f) => f.source)),
  )
  const [page, setPage] = useState(0)
  useEffect(() => setPage(0), [instant])
  const [lookupId, setLookupId] = useState('')
  // Deliberately built from `instant` directly, not `allCards`: mergeResults dedupes a
  // doc_id to whichever source claims it FIRST (ES, then Milvus dense, then sparse - see
  // its own comment), so a doc present in both ES and Milvus sparse only gets an `allCards`
  // entry tagged 'es'. Looking up rank against `allCards` would then silently report
  // "not found" for Milvus sparse even though that retriever genuinely returned it - exactly
  // the kind of wrong answer this lookup exists to prevent. Each source's own rank is
  // recomputed independently here instead, ES in its already-rank-ordered position, Milvus
  // dense/sparse deduped to best-score-per-doc_id across their 7 collections and re-sorted -
  // the honest "which rank is this doc_id at within this retriever, on its own" answer, not
  // filtered by whether some other retriever already claimed it first for display purposes.
  const lookup = useMemo(() => {
    const target = lookupId.trim()
    if (!target) return null
    if (isReranked) {
      const idx = (instant?.reranked ?? []).findIndex((h) => h.doc_id === target)
      return { reranked: idx === -1 ? null : idx + 1 }
    }
    const rankInEs = (() => {
      const idx = (instant?.es ?? []).findIndex((h) => h.doc_id === target)
      return idx === -1 ? null : idx + 1
    })()
    const rankInMilvus = (byCollection: MilvusByCollection | null | undefined) => {
      const bestScoreByDocId = new Map<string, number>()
      for (const hits of Object.values(byCollection ?? {})) {
        for (const hit of hits) {
          const prev = bestScoreByDocId.get(hit.doc_id)
          if (prev === undefined || hit.score > prev) bestScoreByDocId.set(hit.doc_id, hit.score)
        }
      }
      const ranked = [...bestScoreByDocId.entries()].sort((a, b) => b[1] - a[1])
      const idx = ranked.findIndex(([docId]) => docId === target)
      return idx === -1 ? null : idx + 1
    }
    return {
      bySource: {
        es: rankInEs,
        milvus_dense: rankInMilvus(instant?.milvus),
        milvus_sparse: rankInMilvus(instant?.milvus_sparse),
      } as Record<CardSource, number | null>,
    }
  }, [lookupId, instant, isReranked])
  const cards = devMode && !isReranked ? allCards.filter((card) => activeSources.has(card.source)) : allCards
  const pageCount = Math.max(1, Math.ceil(cards.length / PAGE_SIZE))
  const clampedPage = Math.min(page, pageCount - 1)
  const pageCards = cards.slice(clampedPage * PAGE_SIZE, clampedPage * PAGE_SIZE + PAGE_SIZE)

  function toggleSource(source: CardSource) {
    setActiveSources((prev) => {
      const next = new Set(prev)
      if (next.has(source)) next.delete(source)
      else next.add(source)
      return next
    })
    setPage(0)
  }

  return (
    <div className={`flex-1 min-w-0 rounded-2xl px-4 py-3 ${SCROLL_PANE}`} style={{ background: 'var(--surface-raised)' }}>
      <div className="flex items-center gap-2 mb-3">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--text-faint)' }} />
        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
          Instant matches
        </span>
      </div>

      {devMode && instant && !isReranked && (
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

      {devMode && instant && (
        <div className="mb-3">
          <input
            type="text"
            value={lookupId}
            onChange={(e) => setLookupId(e.target.value)}
            placeholder="Check doc_id rank…"
            aria-label="Check doc_id rank"
            className="text-xs w-full px-2 py-1.5 rounded-lg font-mono"
            style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)', color: 'var(--text)' }}
          />
          {lookup && (
            <p className="text-xs mt-1.5 font-mono" style={{ color: 'var(--text-faint)' }}>
              {isReranked
                ? lookup.reranked
                  ? `rank #${lookup.reranked}`
                  : 'not found'
                : SOURCE_FILTERS.map(({ source, label }) => {
                    const rank = lookup.bySource?.[source]
                    return `${label}: ${rank ? `#${rank}` : '—'}`
                  }).join('   ')}
            </p>
          )}
        </div>
      )}

      {!instant && <LoadingDots />}
      {instant && cards.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>No matches.</p>
      )}
      {cards.length > 0 && (
        <div className="flex flex-col gap-2">
          {pageCards.map((card, index) => (
            <button
              key={`${card.source}-${card.doc_id}-${index}`}
              onClick={() => onOpenDocument(card.doc_id)}
              className="w-full text-left rounded-lg p-3 transition-colors duration-150 cursor-pointer"
              style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>
                  {card.heading ? highlightMatches(card.heading, query) : card.doc_id}
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
                      : card.source === 'reranked'
                        ? 'Reranked'
                        : `Milvus ${card.source === 'milvus_dense' ? 'dense' : 'sparse'}:${card.collection}`}
                  </span>
                  <span className="font-mono truncate">{card.doc_id}</span>
                </div>
              )}
              <p className="text-sm mt-2 line-clamp-3" style={{ color: 'var(--text-muted)' }}>{highlightMatches(card.snippet, query)}</p>
            </button>
          ))}
        </div>
      )}

      {cards.length > PAGE_SIZE && (
        <div className="flex items-center justify-between mt-3 pt-3" style={{ borderTop: '1px solid var(--border-soft)' }}>
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={clampedPage === 0}
            className="text-xs px-3 py-1.5 rounded-full font-medium cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-150"
            style={{ background: 'var(--surface)', color: 'var(--text-muted)', border: '1px solid var(--border-soft)' }}
          >
            Prev
          </button>
          <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
            Page {clampedPage + 1} of {pageCount} · {cards.length} matches
          </span>
          <button
            onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
            disabled={clampedPage >= pageCount - 1}
            className="text-xs px-3 py-1.5 rounded-full font-medium cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-150"
            style={{ background: 'var(--surface)', color: 'var(--text-muted)', border: '1px solid var(--border-soft)' }}
          >
            Next
          </button>
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
    <div
      className="flex gap-2 overflow-x-auto pb-3 mb-3"
      style={{
        borderBottom: '1px solid var(--border-soft)',
        // Fades the trailing edge instead of hard-cutting the last card mid-width
        // when the row overflows - a visual cue that there's more to scroll to.
        maskImage: 'linear-gradient(to right, black calc(100% - 24px), transparent)',
        WebkitMaskImage: 'linear-gradient(to right, black calc(100% - 24px), transparent)',
      }}
    >
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
          <InstantPane result={result} devMode={devMode} onOpenDocument={onOpenDocument} query={message.question} />
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
