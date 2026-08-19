import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ChatMessageView } from './ChatMessageView'
import { hydrateStoredMessages } from '../api/useConversations'
import type { ChatMessage, ResultState } from '../types'

describe('ChatMessageView with a reopened (server-hydrated) conversation', () => {
  it('renders a hydrated assistant message without crashing', () => {
    // Regression test: a reopened conversation's server response is flat
    // {role, text} records with no `results` field - ChatMessageView does
    // `message.results[message.activeMode]` unconditionally, which used to
    // crash before loadConversation hydrated the shape.
    const [, assistantMessage] = hydrateStoredMessages('conv-1', [
      { role: 'user', text: 'what is section 80HH' },
      { role: 'assistant', text: 'Section 80HH provides a deduction...' },
    ])

    render(<ChatMessageView message={assistantMessage} devMode={false} onOpenDocument={() => {}} />)

    expect(screen.getByText(/Section 80HH provides a deduction/)).toBeInTheDocument()
  })
})

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

describe('TraceSection copy button', () => {
  function messageWithTrace(status: 'loading' | 'done' = 'done'): ChatMessage {
    return {
      id: 'm2',
      role: 'assistant',
      question: 'q',
      activeMode: 'classic',
      results: {
        classic: {
          status,
          instant: null,
          aiMode: null,
          traceSteps: [{ step: 'intent', data: { query: 'q', search_query: 'q', intent: ['acts'] } }],
        },
      },
    }
  }

  it('copies the trace steps as JSON without toggling the details panel open state', () => {
    const writeText = vi.fn()
    Object.assign(navigator, { clipboard: { writeText } })

    render(<ChatMessageView message={messageWithTrace()} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.click(screen.getByText('Copy'))

    expect(writeText).toHaveBeenCalledWith(
      JSON.stringify([{ step: 'intent', data: { query: 'q', search_query: 'q', intent: ['acts'] } }], null, 2),
    )
  })

  it('shows "Copied" feedback briefly after clicking', () => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn() } })

    render(<ChatMessageView message={messageWithTrace()} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.click(screen.getByText('Copy'))

    expect(screen.getByText('Copied ✓')).toBeInTheDocument()
  })

  it('is disabled and does not copy while the response is still loading', () => {
    const writeText = vi.fn()
    Object.assign(navigator, { clipboard: { writeText } })

    render(<ChatMessageView message={messageWithTrace('loading')} devMode={true} onOpenDocument={() => {}} />)

    const button = screen.getByText('Copy')
    expect(button).toBeDisabled()

    fireEvent.click(button)

    expect(writeText).not.toHaveBeenCalled()
  })
})
