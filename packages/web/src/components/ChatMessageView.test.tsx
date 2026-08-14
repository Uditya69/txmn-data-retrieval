import { describe, expect, it } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ChatMessageView } from './ChatMessageView'
import type { ChatMessage, ResultState } from '../types'

function assistantMessage(instant: ResultState['instant']): ChatMessage {
  return {
    id: 'm1',
    role: 'assistant',
    question: 'q',
    activeMode: 'classic',
    results: {
      classic: { status: 'done', instant, aiMode: null, traceSteps: [] },
    },
  }
}

describe('ChatMessageView doc_id rank lookup (dev mode only)', () => {
  const instant: ResultState['instant'] = {
    es: [
      { doc_id: 'd1', score: 5, heading: 'h1', subheading: 's1' },
      { doc_id: 'd2', score: 4, heading: 'h2', subheading: 's2' },
    ],
    es_error: null,
    milvus: { ruling: [{ chunk_id: 'c1', doc_id: 'd3', text: 't3', score: 0.9 }] },
    milvus_sparse: { ruling: [{ chunk_id: 'c2', doc_id: 'd1', text: 't1', score: 12 }] },
    milvus_error: null,
  }

  it('is not rendered outside dev mode', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.queryByLabelText('Check doc_id rank')).not.toBeInTheDocument()
  })

  it('reports the rank within each source for a doc_id present in some of them', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.change(screen.getByLabelText('Check doc_id rank'), { target: { value: 'd1' } })

    expect(screen.getByText(/ES: #1/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus dense: —/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus sparse: #1/)).toBeInTheDocument()
  })

  it('reports "—" for every source when the doc_id is not present anywhere', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.change(screen.getByLabelText('Check doc_id rank'), { target: { value: 'does-not-exist' } })

    expect(screen.getByText(/ES: —/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus dense: —/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus sparse: —/)).toBeInTheDocument()
  })

  it('shows nothing extra when the input is empty', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={true} onOpenDocument={() => {}} />)
    expect(screen.queryByText(/ES: /)).not.toBeInTheDocument()
  })

  it('reports a single reranked rank when the rerank toggle is on', () => {
    const reranked: ResultState['instant'] = {
      ...instant,
      reranked: [
        { doc_id: 'd2', rerank_score: 0.99, heading: 'h2' },
        { doc_id: 'd3', rerank_score: 0.5, heading: 'h3' },
      ],
    }
    render(<ChatMessageView message={assistantMessage(reranked)} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.change(screen.getByLabelText('Check doc_id rank'), { target: { value: 'd3' } })

    expect(screen.getByText('rank #2')).toBeInTheDocument()
  })
})
