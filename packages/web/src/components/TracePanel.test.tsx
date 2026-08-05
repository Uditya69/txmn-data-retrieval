import { describe, expect, it } from 'vitest'
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
