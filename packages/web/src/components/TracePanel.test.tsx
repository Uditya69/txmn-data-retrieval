import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import TracePanel from './TracePanel'
import type { TraceStep } from '../api/useSearch'

describe('TracePanel', () => {
  it('shows a placeholder when there are no steps yet', () => {
    render(<TracePanel steps={[]} />)
    expect(screen.getByText(/no trace yet/i)).toBeInTheDocument()
  })

  it('renders one card per step, in arrival order, with a summary line', () => {
    const steps: TraceStep[] = [
      { step: 'intent', data: { query: 'cgst', rewritten_query: 'CGST meaning', intent: 'taxation', filters: {} } },
      { step: 'rrf_merge', data: { candidate_count: 42, top_candidates: [] } },
    ]
    render(<TracePanel steps={steps} />)

    const headers = screen.getAllByRole('heading', { level: 3 })
    expect(headers.map((h) => h.textContent)).toEqual(['Intent', 'RRF merge'])
    expect(screen.getByText(/CGST meaning/)).toBeInTheDocument()
    expect(screen.getByText(/42/)).toBeInTheDocument()
  })

  it('renders a query_analysis step with shape summary and chunk breakdown', () => {
    const steps: TraceStep[] = [
      {
        step: 'query_analysis',
        data: {
          query: 'Dimension Data India section 92C',
          shape: 'provision',
          expanded_query: null,
          chunks: [
            { text: 'Dimension Data India', proximity: 5, type: 'text', alt_text: null },
            { text: 'section 92C', proximity: 0, type: 'section', alt_text: 'section 092C' },
          ],
          es_query: { bool: { should: [], minimum_should_match: 1 } },
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByRole('heading', { level: 3, name: 'Query analysis' })).toBeInTheDocument()
    expect(screen.getByText(/shape: provision/)).toBeInTheDocument()
    expect(screen.getByText(/2 chunks/)).toBeInTheDocument()
    expect(screen.getByText('"Dimension Data India"')).toBeInTheDocument()
    expect(screen.getByText(/text, slop 5/)).toBeInTheDocument()
    expect(screen.getByText('"section 92C"')).toBeInTheDocument()
    expect(screen.getByText(/section, slop 0/)).toBeInTheDocument()
    expect(screen.getByText('"section 092C"')).toBeInTheDocument()
  })

  it('truncates long lists to 5 with a Show more button that reveals the rest locally', async () => {
    const user = userEvent.setup()
    const topHits = Array.from({ length: 8 }, (_, i) => ({
      chunk_id: `c${i}`, doc_id: 'd1', score: 1, text_preview: `preview ${i}`,
    }))
    const steps: TraceStep[] = [
      { step: 'milvus_dense', data: { collections: [{ name: 'ruling', hit_count: 8, top_hits: topHits }] } },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText(/preview 4/)).toBeInTheDocument()
    expect(screen.queryByText(/preview 5/)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show 3 more/i }))

    expect(screen.getByText(/preview 5/)).toBeInTheDocument()
    expect(screen.getByText(/preview 7/)).toBeInTheDocument()
  })

  it('renders rrf_score (not score) for rrf_merge candidates', () => {
    const steps: TraceStep[] = [
      {
        step: 'rrf_merge',
        data: {
          candidate_count: 1,
          top_candidates: [{ chunk_id: 'c1', doc_id: 'd1', rrf_score: 0.016393, text_preview: 'hello' }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText(/\[0\.0164\]/)).toBeInTheDocument()
    expect(screen.queryByText(/^\[\]/)).not.toBeInTheDocument()
  })

  it('renders doc_id as a clickable link when onOpenDocument is provided, and calls it with the doc_id', async () => {
    const user = userEvent.setup()
    const onOpenDocument = vi.fn()
    const steps: TraceStep[] = [
      { step: 'es_search', data: { hits: [{ doc_id: 'd1', score: 4.2, heading: 'Heading', subheading: 'Sub' }] } },
    ]
    render(<TracePanel steps={steps} onOpenDocument={onOpenDocument} />)

    const link = screen.getByRole('button', { name: 'd1' })
    await user.click(link)

    expect(onOpenDocument).toHaveBeenCalledWith('d1')
  })

  it('renders rerank_score (not score) for rerank top chunks', () => {
    const steps: TraceStep[] = [
      {
        step: 'rerank',
        data: {
          considered_count: 1,
          top_chunks: [{ chunk_id: 'c1', doc_id: 'd1', rerank_score: 0.87654, text: 'hello' }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText(/\[0\.8765\]/)).toBeInTheDocument()
    expect(screen.queryByText(/^\[\]/)).not.toBeInTheDocument()
  })

  it('renders an agent_tool_call step with its name and arguments', () => {
    render(<TracePanel steps={[{ step: 'agent_tool_call', data: { name: 'search_es', arguments: { query: 'gst rate' } } }]} />)
    expect(screen.getByText('Agent tool call')).toBeInTheDocument()
    expect(screen.getByText(/search_es/)).toBeInTheDocument()
    expect(screen.getByText(/gst rate/)).toBeInTheDocument()
  })

  it('renders an agent_tool_result step showing hit count', () => {
    render(<TracePanel steps={[{
      step: 'agent_tool_result',
      data: { name: 'search_es', result: { rows: [{ doc_id: 'd1', score: 1, heading: 'H' }] } },
    }]} />)
    expect(screen.getByText('Agent tool result')).toBeInTheDocument()
    expect(screen.getByText(/1 row/)).toBeInTheDocument()
  })

  it('renders an agent_tool_result error without crashing', () => {
    render(<TracePanel steps={[{ step: 'agent_tool_result', data: { name: 'search_es', result: { error: 'ES timed out' } } }]} />)
    expect(screen.getByText(/error: ES timed out/)).toBeInTheDocument()
  })

  it('renders an agent_citation_rejected step with attempt and invalid ids', () => {
    render(<TracePanel steps={[{ step: 'agent_citation_rejected', data: { invalid_doc_ids: ['d999'], attempt: 1 } }]} />)
    expect(screen.getByText('Citation rejected — retrying')).toBeInTheDocument()
    expect(screen.getByText(/attempt 1/)).toBeInTheDocument()
    expect(screen.getByText(/d999/)).toBeInTheDocument()
  })

  it('renders an agent_answer step with cited doc count', () => {
    render(<TracePanel steps={[{ step: 'agent_answer', data: { answer: 'See [d1].', doc_ids: ['d1'] } }]} />)
    expect(screen.getByText('Agent answer')).toBeInTheDocument()
    expect(screen.getByText(/1 doc/)).toBeInTheDocument()
  })

  it('renders agent_tool_result rows as clickable links when onOpenDocument is provided', async () => {
    const user = userEvent.setup()
    const onOpenDocument = vi.fn()
    const steps: TraceStep[] = [
      {
        step: 'agent_tool_result',
        data: {
          name: 'search_es',
          result: { rows: [{ doc_id: 'd2', score: 3.5, heading: 'Tool Result' }] },
        },
      },
    ]
    render(<TracePanel steps={steps} onOpenDocument={onOpenDocument} />)

    const link = screen.getByRole('button', { name: 'd2' })
    await user.click(link)

    expect(onOpenDocument).toHaveBeenCalledWith('d2')
  })
})
