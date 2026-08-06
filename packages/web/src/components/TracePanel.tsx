// src/components/TracePanel.tsx
import { useState } from 'react'
import type { TraceStep } from '../api/useSearch'
import styles from './TracePanel.module.css'

const STEP_LABELS: Record<string, string> = {
  intent: 'Intent',
  filters_resolved: 'Filters resolved',
  es_search: 'ES search',
  milvus_dense: 'Milvus dense search',
  milvus_sparse: 'Milvus sparse search',
  rrf_merge: 'RRF merge',
  rerank: 'Rerank',
  synthesis_prompt: 'Synthesis prompt',
  agent_tool_call: 'Agent tool call',
  agent_tool_result: 'Agent tool result',
  agent_citation_rejected: 'Citation rejected — retrying',
  agent_answer: 'Agent answer',
}

function summarize(step: TraceStep): string {
  const d = step.data as Record<string, any>
  switch (step.step) {
    case 'intent':
      return `"${d.query}" -> "${d.rewritten_query}" (${d.intent})`
    case 'filters_resolved':
      return `${d.doc_id_count} doc(s) matched`
    case 'es_search':
      return `${(d.hits ?? []).length} hit(s)`
    case 'milvus_dense':
    case 'milvus_sparse': {
      const collections = d.collections ?? []
      const total = collections.reduce((sum: number, c: any) => sum + c.hit_count, 0)
      return `${collections.length} collections, ${total} hits`
    }
    case 'rrf_merge':
      return `${d.candidate_count} candidates merged`
    case 'rerank':
      return `${d.considered_count} considered, top ${d.top_chunks?.length ?? 0} kept`
    case 'synthesis_prompt':
      return `${(d.prompt ?? '').length} chars`
    case 'agent_tool_call':
      return `${d.name}(${JSON.stringify(d.arguments)})`
    case 'agent_tool_result': {
      if (d.result?.error) return `error: ${d.result.error}`
      if (d.result?.citation !== undefined) return d.result.citation ? 'citation found' : 'citation not found'
      const rows = d.result?.rows ?? []
      return `${rows.length} row(s)`
    }
    case 'agent_citation_rejected':
      return `attempt ${d.attempt}: invalid doc_id(s) ${(d.invalid_doc_ids ?? []).join(', ')}`
    case 'agent_answer':
      return `${(d.doc_ids ?? []).length} doc(s) cited`
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

function StepBody({ step, onOpenDocument }: { step: TraceStep; onOpenDocument?: (docId: string) => void }) {
  const d = step.data as Record<string, any>
  if (step.step === 'es_search') {
    return <TruncatedHitList hits={d.hits ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'milvus_dense' || step.step === 'milvus_sparse') {
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
  if (step.step === 'rrf_merge') {
    return <TruncatedHitList hits={d.top_candidates ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'rerank') {
    return <TruncatedHitList hits={d.top_chunks ?? []} onOpenDocument={onOpenDocument} />
  }
  if (step.step === 'synthesis_prompt') {
    return <pre className={styles.hitList}>{d.prompt}</pre>
  }
  if (step.step === 'agent_tool_result' && d.result?.rows) {
    return <TruncatedHitList hits={d.result.rows} />
  }
  return null
}

export interface TracePanelProps {
  steps: TraceStep[]
  onOpenDocument?: (docId: string) => void
}

export default function TracePanel({ steps, onOpenDocument }: TracePanelProps) {
  if (steps.length === 0) {
    return <p className={styles.placeholder}>No trace yet — run an AI Mode query to see it here.</p>
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
