// src/components/TracePanel.tsx
import { useState } from 'react'
import type { TraceStep } from '../api/useSearch'
import styles from './TracePanel.module.css'

const STEP_LABELS: Record<string, string> = {
  intent: 'Intent',
  filters_resolved: 'Filters resolved',
  milvus_dense: 'Milvus dense search',
  milvus_sparse: 'Milvus sparse search',
  rrf_merge: 'RRF merge',
  rerank: 'Rerank',
  synthesis_prompt: 'Synthesis prompt',
}

function summarize(step: TraceStep): string {
  const d = step.data as Record<string, any>
  switch (step.step) {
    case 'intent':
      return `"${d.query}" -> "${d.rewritten_query}" (${d.intent})`
    case 'filters_resolved':
      return `${d.doc_id_count} doc(s) matched`
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
    default:
      return ''
  }
}

function TruncatedHitList({ hits }: { hits: Array<Record<string, any>> }) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? hits : hits.slice(0, 5)
  const remaining = hits.length - visible.length

  return (
    <>
      <ul className={styles.hitList}>
        {visible.map((hit, i) => (
          <li key={hit.chunk_id ?? i}>
            [{hit.score?.toFixed?.(3) ?? hit.score}] {hit.doc_id}: {hit.text_preview ?? hit.text}
          </li>
        ))}
      </ul>
      {remaining > 0 && (
        <button type="button" className={styles.showMore} onClick={() => setExpanded(true)}>
          Show {remaining} more
        </button>
      )}
    </>
  )
}

function StepBody({ step }: { step: TraceStep }) {
  const d = step.data as Record<string, any>
  if (step.step === 'milvus_dense' || step.step === 'milvus_sparse') {
    return (
      <>
        {(d.collections ?? []).map((c: any) => (
          <div key={c.name}>
            <strong>{c.name}</strong> ({c.hit_count})
            <TruncatedHitList hits={c.top_hits ?? []} />
          </div>
        ))}
      </>
    )
  }
  if (step.step === 'rrf_merge') {
    return <TruncatedHitList hits={d.top_candidates ?? []} />
  }
  if (step.step === 'rerank') {
    return <TruncatedHitList hits={d.top_chunks ?? []} />
  }
  if (step.step === 'synthesis_prompt') {
    return <pre className={styles.hitList}>{d.prompt}</pre>
  }
  return null
}

export interface TracePanelProps {
  steps: TraceStep[]
}

export default function TracePanel({ steps }: TracePanelProps) {
  if (steps.length === 0) {
    return <p className={styles.placeholder}>No trace yet — run an AI Mode query to see it here.</p>
  }

  return (
    <div className={styles.panel}>
      {steps.map((step, index) => (
        <section key={`${step.step}-${index}`} className={styles.card}>
          <h3>{STEP_LABELS[step.step] ?? step.step}</h3>
          <p className={styles.summary}>{summarize(step)}</p>
          <StepBody step={step} />
        </section>
      ))}
    </div>
  )
}
