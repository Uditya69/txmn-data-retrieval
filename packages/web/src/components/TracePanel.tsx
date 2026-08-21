// src/components/TracePanel.tsx
import { useState } from 'react'
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
      return `${d.label} (${(d.confidence * 100).toFixed(1)}% confidence)${skippedText}`
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

function QueryAnalysisBody({ data }: { data: Record<string, any> }) {
  // Same build_query_preview() output the /v1/query-analysis endpoint returns - see its
  // own docstring (common/es_client.py) - so this must never show a different breakdown
  // than what raw_search actually sent to ES.
  const chunks: Array<{ text: string; proximity: number; type: string; alt_text: string | null }> = data.chunks ?? []
  return (
    <>
      {data.expanded_query && (
        <p className={styles.summary}>
          Synonym-expanded: <code>{data.expanded_query}</code>
        </p>
      )}
      <ul className={styles.hitList}>
        {chunks.map((c, i) => (
          <li key={`${c.text}-${i}`}>
            <strong>"{c.text}"</strong> <em>({c.type}, slop {c.proximity})</em>
            {c.alt_text && <> — alt: <code>"{c.alt_text}"</code></>}
          </li>
        ))}
      </ul>
      {/* {data.es_query && (
        <details>
          <summary className={styles.summary} style={{ cursor: 'pointer' }}>Raw ES query</summary>
          <pre className={styles.hitList}>{JSON.stringify(data.es_query, null, 2)}</pre>
        </details>
      )} */}
    </>
  )
}

function StepBody({ step, onOpenDocument }: { step: TraceStep; onOpenDocument?: (docId: string) => void }) {
  const d = step.data as Record<string, any>
  if (step.step === 'query_analysis') {
    return <QueryAnalysisBody data={d} />
  }
  if (step.step === 'es_search') {
    return <TruncatedHitList hits={d.hits ?? []} onOpenDocument={onOpenDocument} />
  }
  if (
    step.step === 'milvus_dense' || step.step === 'milvus_sparse' ||
    step.step === 'ai_milvus_dense' || step.step === 'ai_milvus_sparse'
  ) {
    return (
      <>
        {(d.collections ?? []).map((c: any) => (
          <div key={c.name}>
            <strong>{c.name}</strong> ({c.hit_count})
            <TruncatedHitList hits={c.top_hits ?? []} onOpenDocument={onOpenDocument} />
          </div>
        ))}
      </>
    )
  }
  if (step.step === 'rrf_merge' || step.step === 'ai_rrf_merge') {
    return <TruncatedHitList hits={d.top_candidates ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'rerank') {
    return <TruncatedHitList hits={d.top_chunks ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'instant_reranked') {
    return <TruncatedHitList hits={d.hits ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'synthesis_prompt') {
    return <pre className={styles.promptBlock}>{d.prompt}</pre>
  }
  return null
}

export interface TracePanelProps {
  steps: TraceStep[]
  onOpenDocument?: (docId: string) => void
}

export default function TracePanel({ steps, onOpenDocument }: TracePanelProps) {
  if (steps.length === 0) {
    return <p className={styles.placeholder}>No trace yet — run a query to see it here.</p>
  }

  return (
    <div className={styles.panel}>
      {steps.map((step, index) => (
        <section key={`${step.step}-${index}`} className={styles.card}>
          <h3>{STEP_LABELS[step.step] ?? step.step}</h3>
          <p className={styles.summary}>{summarize(step)}</p>
          <StepBody step={step} onOpenDocument={onOpenDocument} />
        </section>
      ))}
    </div>
  )
}
