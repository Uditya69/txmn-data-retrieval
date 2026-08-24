// src/components/TracePanel.tsx
import { useState } from 'react'
import type { MouseEvent, ReactNode } from 'react'
import type { TraceStep } from '../api/useSearch'
import styles from './TracePanel.module.css'

const STEP_LABELS: Record<string, string> = {
  query_correction: 'Query correction',
  query_analysis: 'Query analysis',
  classifier: 'Classifier',
  intent: 'Intent',
  filters_resolved: 'Filters resolved',
  es_search: 'ES search',
  milvus_dense: 'Milvus dense search',
  milvus_sparse: 'Milvus sparse search',
  rrf_merge: 'RRF merge',
  ai_milvus_dense: 'Milvus dense search',
  ai_milvus_sparse: 'Milvus sparse search',
  ai_rrf_merge: 'RRF merge',
  rerank: 'Rerank',
  instant_reranked: 'Instant rerank',
  synthesis_prompt: 'Synthesis prompt',
}

const ORIGIN_LABELS: Record<string, string> = {
  es: 'ES',
  milvus_dense: 'Milvus dense',
  milvus_sparse: 'Milvus sparse',
}

const STEP_DESCRIPTIONS: Record<string, string> = {
  query_correction: 'Fixes likely typos in court/journal/legal-term names before anything else runs.',
  query_analysis: 'Breaks the query into the chunks actually sent to Elasticsearch (sections, citations, free text).',
  classifier: 'Picks KEYWORD (structural, ES only), INTENT (semantic, Milvus only), or HYBRID (both, RRF-fused).',
  intent: 'Classifies which legal categories the query is about, used to pick which Milvus collections get searched.',
  filters_resolved: 'Resolves any doc_id filters requested by the query.',
  es_search: 'Elasticsearch lexical (keyword) search results.',
  milvus_dense: 'Milvus dense (semantic embedding) search results.',
  milvus_sparse: 'Milvus BM25 sparse search results.',
  ai_milvus_dense: 'Milvus dense (semantic embedding) search results.',
  ai_milvus_sparse: 'Milvus BM25 sparse search results.',
  rrf_merge: 'Merges ES + Milvus results by rank position (RRF) — never by comparing raw scores.',
  ai_rrf_merge: 'Merges ES + Milvus results by rank position (RRF) — never by comparing raw scores.',
  rerank: 'Cross-encoder reranking of the merged candidates.',
  instant_reranked: 'Final result list for Instant mode after fusion/reranking.',
  synthesis_prompt: 'The prompt sent to the LLM to synthesize the final answer.',
}

function summarize(step: TraceStep): string {
  const d = step.data as Record<string, any>
  switch (step.step) {
    case 'query_correction': {
      const corrections = d.corrections ?? []
      return corrections.length === 0
        ? 'no corrections'
        : `${corrections.length} correction(s): ${corrections.map((c: any) => `"${c.original}" -> "${c.corrected}"`).join(', ')}`
    }
    case 'query_analysis': {
      const chunkCount = (d.chunks ?? []).length
      return `shape: ${d.shape} — ${chunkCount} chunk${chunkCount === 1 ? '' : 's'}`
    }
    case 'classifier': {
      const skipped = [!d.plan?.es && 'ES', !d.plan?.milvus && 'Milvus'].filter(Boolean)
      const skippedText = skipped.length ? `, skipped: ${skipped.join(', ')}` : ''
      // return `${d.label} (${(d.confidence * 100).toFixed(1)}% confidence)${skippedText}`
      return `${d.label} ${skippedText}`
    }
    case 'intent':
      return `"${d.query}" -> "${d.search_query}" (${(d.intent ?? []).join(', ')})`
    case 'filters_resolved':
      return `${d.doc_id_count} doc(s) matched`
    case 'es_search':
      return `${(d.hits ?? []).length} hit(s)`
    case 'milvus_dense':
    case 'milvus_sparse':
    case 'ai_milvus_dense':
    case 'ai_milvus_sparse': {
      const collections = d.collections ?? []
      const total = collections.reduce((sum: number, c: any) => sum + c.hit_count, 0)
      return `${collections.length} collections, ${total} hits`
    }
    case 'rrf_merge':
    case 'ai_rrf_merge':
      return `${d.candidate_count} candidates merged`
    case 'rerank': {
      const capped = d.total_candidates !== undefined && d.total_candidates > d.considered_count
      const from = capped ? ` (capped from ${d.total_candidates})` : ''
      return `${d.considered_count} considered${from}, top ${d.top_chunks?.length ?? 0} kept`
    }
    case 'instant_reranked':
      return `${(d.hits ?? []).length} result(s)`
    case 'synthesis_prompt':
      return `${(d.prompt ?? '').length} chars`
    default:
      return ''
  }
}

function TruncatedHitList({
  hits,
  onOpenDocument,
}: {
  hits: Array<Record<string, any>>
  onOpenDocument?: (docId: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? hits : hits.slice(0, 5)
  const remaining = hits.length - visible.length

  return (
    <>
      <ul className={styles.hitList}>
        {visible.map((hit, i) => {
          const score = hit.score ?? hit.rrf_score ?? hit.rerank_score
          const description = hit.heading
            ? `${hit.heading}${hit.subheading ? ` — ${hit.subheading}` : ''}`
            : hit.text_preview ?? hit.text
          return (
            <li key={hit.chunk_id ?? hit.doc_id ?? i}>
              {hit.origin && <span className={styles.originBadge}>{ORIGIN_LABELS[hit.origin] ?? hit.origin}</span>}
              [{score?.toFixed?.(4) ?? score}]{' '}
              {onOpenDocument ? (
                <button type="button" className={styles.docLink} onClick={() => onOpenDocument(hit.doc_id)}>
                  {hit.doc_id}
                </button>
              ) : (
                hit.doc_id
              )}
              : {description}
            </li>
          )
        })}
      </ul>
      {remaining > 0 && (
        <button type="button" className={styles.showMore} onClick={() => setExpanded(true)}>
          Show {remaining} more
        </button>
      )}
    </>
  )
}

type QueryChunk = { text: string; proximity: number; type: string; alt_text: string | null }

function ChunkList({ chunks }: { chunks: QueryChunk[] }) {
  return (
    <ul className={styles.hitList}>
      {chunks.map((c, i) => (
        <li key={`${c.text}-${i}`}>
          <strong>"{c.text}"</strong> <em>({c.type}, slop {c.proximity})</em>
          {c.alt_text && <> — alt: <code>"{c.alt_text}"</code></>}
        </li>
      ))}
    </ul>
  )
}

function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = (e: MouseEvent) => {
    // Lives inside a <summary> in every call site - stop the click from also
    // toggling the parent <details> open/closed (its default behavior).
    e.preventDefault()
    e.stopPropagation()
    navigator.clipboard.writeText(getText())
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <button type="button" className={styles.copyButton} onClick={handleCopy}>
      {copied ? 'Copied ✓' : 'Copy'}
    </button>
  )
}

// Mirrors retrieval_api/instant/search.py's _ES_LIMIT - the real raw_search() call this
// preview stands in for always searches with this size, so the request body shown/copied
// here must carry it too, or pasting it into Elasticvue/curl runs a different search than
// what Instant mode actually sent.
const _ES_LIMIT = 20

function QueryAnalysisBody({ data }: { data: Record<string, any> }) {
  // Same build_query_preview() output the /v1/query-analysis endpoint returns - see its
  // own docstring (common/es_client.py) - so this must never show a different breakdown
  // than what raw_search actually sent to ES.
  const chunks: QueryChunk[] = data.chunks ?? []
  // The full request body ES itself expects (query + size) - copyable straight into
  // Elasticvue's REST console or a curl -d payload, not just the bare query clause.
  const esRequestBody = data.es_query ? { query: data.es_query, size: _ES_LIMIT } : null
  return (
    <>
      {data.expanded_query && (
        <p className={styles.summary}>
          Synonym-expanded: <code>{data.expanded_query}</code>
        </p>
      )}
      <ChunkList chunks={chunks} />
      {esRequestBody && (
        <details className={styles.details}>
          <summary className={styles.detailsSummaryRow}>
            <span className={styles.detailsSummary}>Show ES query</span>
            <CopyButton getText={() => JSON.stringify(esRequestBody, null, 2)} />
          </summary>
          <pre className={styles.promptBlock}>{JSON.stringify(esRequestBody, null, 2)}</pre>
        </details>
      )}
    </>
  )
}

function ClassifierBody({ data, chunks }: { data: Record<string, any>; chunks: QueryChunk[] | null }) {
  return (
    <>
      <p className={styles.summary}>
        Boost profile: <code>{data.label}</code>
        {data.auto_route && (
          <> — routed: ES {data.plan?.es ? 'yes' : 'no'}, Milvus {data.plan?.milvus ? 'yes' : 'no'}, fuse {data.plan?.fuse ? 'yes' : 'no'}</>
        )}
      </p>
      {chunks && chunks.length > 0 && (
        <>
          <p className={styles.summary}>Terms this decision was based on:</p>
          <ChunkList chunks={chunks} />
        </>
      )}
    </>
  )
}

const COLLAPSIBLE_LABELS: Record<string, string> = {
  synthesis_prompt: 'Show prompt',
}

function collapsibleLabel(step: string): string {
  return COLLAPSIBLE_LABELS[step] ?? 'Show results'
}

function StepBody({
  step, onOpenDocument, precedingChunks,
}: {
  step: TraceStep
  onOpenDocument?: (docId: string) => void
  precedingChunks: QueryChunk[] | null
}) {
  const d = step.data as Record<string, any>
  if (step.step === 'query_analysis') {
    return <QueryAnalysisBody data={d} />
  }
  if (step.step === 'classifier') {
    return <ClassifierBody data={d} chunks={precedingChunks} />
  }

  let body: ReactNode = null
  if (step.step === 'es_search') {
    body = <TruncatedHitList hits={d.hits ?? []} onOpenDocument={onOpenDocument} />
  } else if (
    step.step === 'milvus_dense' || step.step === 'milvus_sparse' ||
    step.step === 'ai_milvus_dense' || step.step === 'ai_milvus_sparse'
  ) {
    body = (
      <>
        {(d.collections ?? []).map((c: any) => (
          <div key={c.name}>
            <strong>{c.name}</strong> ({c.hit_count})
            <TruncatedHitList hits={c.top_hits ?? []} onOpenDocument={onOpenDocument} />
          </div>
        ))}
      </>
    )
  } else if (step.step === 'rrf_merge' || step.step === 'ai_rrf_merge') {
    body = <TruncatedHitList hits={d.top_candidates ?? []} onOpenDocument={onOpenDocument} />
  } else if (step.step === 'rerank') {
    body = <TruncatedHitList hits={d.top_chunks ?? []} onOpenDocument={onOpenDocument} />
  } else if (step.step === 'instant_reranked') {
    body = <TruncatedHitList hits={d.hits ?? []} onOpenDocument={onOpenDocument} />
  } else if (step.step === 'synthesis_prompt') {
    body = <pre className={styles.promptBlock}>{d.prompt}</pre>
  }

  if (body === null) return null

  return (
    <details className={styles.details}>
      <summary className={styles.detailsSummary}>{collapsibleLabel(step.step)}</summary>
      {body}
    </details>
  )
}

export interface TracePanelProps {
  steps: TraceStep[]
  onOpenDocument?: (docId: string) => void
}

export default function TracePanel({ steps, onOpenDocument }: TracePanelProps) {
  if (steps.length === 0) {
    return <p className={styles.placeholder}>No trace yet — run a query to see it here.</p>
  }

  let lastChunks: QueryChunk[] | null = null

  return (
    <div className={styles.panel}>
      {steps.map((step, index) => {
        if (step.step === 'query_analysis') {
          lastChunks = (step.data as Record<string, any>).chunks ?? null
        }
        return (
          <section key={`${step.step}-${index}`} className={styles.card}>
            <h3>{STEP_LABELS[step.step] ?? step.step}</h3>
            {STEP_DESCRIPTIONS[step.step] && (
              <p className={styles.description}>{STEP_DESCRIPTIONS[step.step]}</p>
            )}
            <p className={styles.summary}>{summarize(step)}</p>
            <StepBody step={step} onOpenDocument={onOpenDocument} precedingChunks={lastChunks} />
          </section>
        )
      })}
    </div>
  )
}
