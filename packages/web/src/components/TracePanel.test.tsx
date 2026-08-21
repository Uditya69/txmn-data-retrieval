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
      {
        step: 'intent',
        data: {
          query: 'cgst',
          original_query: 'cgst',
          search_query: 'CGST meaning',
          intent: ['caselaws', 'acts'],
          filters: {},
        },
      },
      { step: 'rrf_merge', data: { candidate_count: 42, top_candidates: [] } },
    ]
    render(<TracePanel steps={steps} />)

    const headers = screen.getAllByRole('heading', { level: 3 })
    expect(headers.map((h) => h.textContent)).toEqual(['Intent', 'RRF merge'])
    expect(screen.getByText(/CGST meaning/)).toBeInTheDocument()
    expect(screen.getByText(/caselaws, acts/)).toBeInTheDocument()
    expect(screen.getByText(/42/)).toBeInTheDocument()
  })

  it('renders a query_correction step with no corrections', () => {
    const steps: TraceStep[] = [
      { step: 'query_correction', data: { original: 'q', corrected: 'q', corrections: [] } },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByRole('heading', { level: 3, name: 'Query correction' })).toBeInTheDocument()
    expect(screen.getByText(/no corrections/)).toBeInTheDocument()
  })

  it('renders a query_correction step listing the original -> corrected term', () => {
    const steps: TraceStep[] = [
      {
        step: 'query_correction',
        data: {
          original: 'case from AHMDABAD tribunal',
          corrected: 'case from AHMEDABAD tribunal',
          corrections: [{ original: 'AHMDABAD', corrected: 'AHMEDABAD', score: 94.1 }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText(/1 correction/)).toBeInTheDocument()
    expect(screen.getByText(/"AHMDABAD" -> "AHMEDABAD"/)).toBeInTheDocument()
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

})
