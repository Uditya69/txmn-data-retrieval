import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
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

  it('reports a single reranked rank when the RRF toggle is on', () => {
    const reranked: ResultState['instant'] = {
      ...instant,
      reranked: [
        { doc_id: 'd2', rrf_score: 0.99, heading: 'h2' },
        { doc_id: 'd3', rrf_score: 0.5, heading: 'h3' },
      ],
    }
    render(<ChatMessageView message={assistantMessage(reranked)} devMode={true} onOpenDocument={() => {}} />)

    fireEvent.change(screen.getByLabelText('Check doc_id rank'), { target: { value: 'd3' } })

    expect(screen.getByText('rank #2')).toBeInTheDocument()
  })
})

describe('ChatMessageView result card — doc_id and enriched metadata', () => {
  const instant: ResultState['instant'] = {
    es: [{ doc_id: 'd1', score: 5, heading: 'h1', subheading: 's1' }],
    es_error: null, milvus: null, milvus_sparse: null, milvus_error: null,
    doc_meta: {
      d1: {
        category: 'Direct Tax Laws', group: 'Case Laws',
        judge: ['V.K. KHANNA'], party: ['Commissioner of Income-tax'],
        date: '1987-03-31T00:00:00', viewcount: 70,
      },
    },
  }

  it('shows doc_id even outside dev mode', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('d1')).toBeInTheDocument()
  })

  it('shows act_name even outside dev mode (mirrors the reference product\'s always-visible Act-name label)', () => {
    const withActName: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { ...instant.doc_meta!.d1, act_name: 'Companies Act, 2013' } },
    }
    render(<ChatMessageView message={assistantMessage(withActName)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('Companies Act, 2013')).toBeInTheDocument()
  })

  it('omits the act_name line when absent', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.queryByText('Companies Act, 2013')).not.toBeInTheDocument()
  })

  it('shows tariff_name even outside dev mode when act_name is absent', () => {
    const withTariffName: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { ...instant.doc_meta!.d1, tariff_name: 'GST Tariff for Goods - Edition 42 Goods' } },
    }
    render(<ChatMessageView message={assistantMessage(withTariffName)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('GST Tariff for Goods - Edition 42 Goods')).toBeInTheDocument()
  })

  it('shows commentary_topic even outside dev mode when act_name/tariff_name are absent', () => {
    const withCommentaryTopic: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { ...instant.doc_meta!.d1, commentary_topic: 'Tax Audit' } },
    }
    render(<ChatMessageView message={assistantMessage(withCommentaryTopic)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('Tax Audit')).toBeInTheDocument()
  })

  it('shows an Unreported badge even outside dev mode when is_unreported is true', () => {
    const withUnreported: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { ...instant.doc_meta!.d1, is_unreported: true } },
    }
    render(<ChatMessageView message={assistantMessage(withUnreported)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('Unreported')).toBeInTheDocument()
  })

  it('omits the Unreported badge when is_unreported is absent', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.queryByText('Unreported')).not.toBeInTheDocument()
  })

  it('hides judge, party, date, and viewcount outside dev mode (internal debug detail, not a production card field)', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.queryByText(/V.K. KHANNA/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Commissioner of Income-tax/)).not.toBeInTheDocument()
    expect(screen.queryByText(/1987-03-31/)).not.toBeInTheDocument()
    expect(screen.queryByText(/70 views/)).not.toBeInTheDocument()
  })

  it('shows judge, party, date, and viewcount in dev mode when present on doc_meta', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={true} onOpenDocument={() => {}} />)
    expect(screen.getByText(/V.K. KHANNA/)).toBeInTheDocument()
    expect(screen.getByText(/Commissioner of Income-tax/)).toBeInTheDocument()
    expect(screen.getByText(/1987-03-31/)).toBeInTheDocument()
    expect(screen.getByText(/70/)).toBeInTheDocument()
  })

  it('omits judge/party lines entirely for a doc with no such doc_meta fields, even in dev mode', () => {
    const noExtras: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { category: 'Acts', group: 'Acts' } },
    }
    render(<ChatMessageView message={assistantMessage(noExtras)} devMode={true} onOpenDocument={() => {}} />)
    expect(screen.queryByText(/V.K. KHANNA/)).not.toBeInTheDocument()
  })

  it('does not render a stray "0" when one paired field is an empty array and its pair is absent', () => {
    const emptyArrayCase: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { category: 'Acts', group: 'Acts', party: [] } }, // judge absent, party present-but-empty
    }
    render(<ChatMessageView message={assistantMessage(emptyArrayCase)} devMode={true} onOpenDocument={() => {}} />)
    // The bug rendered a stray "0" text node from `undefined || 0` short-circuiting `&&`.
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })
})

describe('TraceSection routes query_correction to the Instant pane, not the Answer pane', () => {
  function messageWithBothPaneSteps(): ChatMessage {
    return {
      id: 'm3',
      role: 'assistant',
      question: 'q',
      activeMode: 'classic',
      results: {
        classic: {
          status: 'done',
          instant: null,
          aiMode: null,
          traceSteps: [
            { step: 'query_correction', data: { original: 'q', corrected: 'q', corrections: [] } },
            { step: 'intent', data: { query: 'q', search_query: 'q', intent: ['acts'] } },
          ],
        },
      },
    }
  }

  it('shows the Query correction card inside the Instant pane, not the Answer pane', () => {
    render(<ChatMessageView message={messageWithBothPaneSteps()} devMode={true} onOpenDocument={() => {}} />)

    for (const summary of screen.getAllByText(/^Trace \(/)) {
      fireEvent.click(summary)
    }

    const instantPane = screen.getByText('Instant matches').closest('div')!.parentElement!
    const answerPane = screen.getByText('Answer').closest('div')!.parentElement!

    expect(within(instantPane).getByRole('heading', { level: 3, name: 'Query correction' })).toBeInTheDocument()
    expect(within(answerPane).queryByRole('heading', { level: 3, name: 'Query correction' })).not.toBeInTheDocument()
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

describe('InstantPane pagination — hidden by default', () => {
  function manyEsHitsInstant(): ResultState['instant'] {
    const manyEsHits = Array.from({ length: 25 }, (_, i) => ({ doc_id: `d${i}`, score: 1, heading: `h${i}`, subheading: '' }))
    return { es: manyEsHits, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null }
  }

  it('Next button calls onFetchPage with page 2 when paginationEnabled is true', () => {
    const onFetchPage = vi.fn()
    render(
      <ChatMessageView
        message={assistantMessage(manyEsHitsInstant())} devMode={false} onOpenDocument={() => {}}
        paginationEnabled={true} onFetchPage={onFetchPage} currentPage={1}
      />,
    )
    fireEvent.click(screen.getByText('Next'))
    expect(onFetchPage).toHaveBeenCalledWith(2)
  })

  // Regression test for C1 (2026-09-02 final review): the server page must advance
  // monotonically (2, 3, 4...) as Next is clicked repeatedly, driven by the controlled
  // `currentPage` prop (mirroring App.tsx's real `instantPage` state) - not oscillate back
  // to 2 forever because of InstantPane's own internal, self-resetting `page` slice index.
  it('Next advances the server page 2, 3, 4 across repeated clicks, driven by currentPage, never oscillating back', () => {
    const onFetchPage = vi.fn()
    const { rerender } = render(
      <ChatMessageView
        message={assistantMessage(manyEsHitsInstant())} devMode={false} onOpenDocument={() => {}}
        paginationEnabled={true} onFetchPage={onFetchPage} currentPage={1}
      />,
    )

    fireEvent.click(screen.getByText('Next'))
    expect(onFetchPage).toHaveBeenNthCalledWith(1, 2)

    // App.tsx would re-render with the new server page once results for page 2 land -
    // simulated here by bumping currentPage, same as the real controlled-prop flow.
    rerender(
      <ChatMessageView
        message={assistantMessage(manyEsHitsInstant())} devMode={false} onOpenDocument={() => {}}
        paginationEnabled={true} onFetchPage={onFetchPage} currentPage={2}
      />,
    )
    fireEvent.click(screen.getByText('Next'))
    expect(onFetchPage).toHaveBeenNthCalledWith(2, 3)

    rerender(
      <ChatMessageView
        message={assistantMessage(manyEsHitsInstant())} devMode={false} onOpenDocument={() => {}}
        paginationEnabled={true} onFetchPage={onFetchPage} currentPage={3}
      />,
    )
    fireEvent.click(screen.getByText('Next'))
    expect(onFetchPage).toHaveBeenNthCalledWith(3, 4)
  })

  it('does not require onFetchPage when paginationEnabled is false (default, unchanged behavior)', () => {
    render(<ChatMessageView message={assistantMessage(manyEsHitsInstant())} devMode={false} onOpenDocument={() => {}} />)
    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText(/Page 2 of/)).toBeInTheDocument() // client-side slice still works exactly as before
  })
})
