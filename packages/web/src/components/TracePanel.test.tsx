import { describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
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

  it('shows the raw ES query collapsed by default, with a copy button', () => {
    const esQuery = { bool: { should: [{ match: { heading: 'Section 52' } }], minimum_should_match: 1 } }
    const steps: TraceStep[] = [
      {
        step: 'query_analysis',
        data: { query: 'section 52', shape: 'KEYWORD', chunks: [], es_query: esQuery },
      },
    ]
    const { container } = render(<TracePanel steps={steps} />)

    expect(screen.getByText('Show ES query')).toBeInTheDocument()
    const details = container.querySelector('details')
    expect(details?.open).toBe(false)
    expect(details?.textContent).toContain('"minimum_should_match": 1')
    // Wrapped as a full request body - copyable straight into Elasticvue/curl, not just
    // the bare query clause.
    expect(details?.textContent).toContain('"size": 20')
  })

  it('copies the ES query wrapped as a full request body ({query, size}), without toggling the details panel open state', async () => {
    const writeText = vi.fn()
    Object.assign(navigator, { clipboard: { writeText } })
    const esQuery = { bool: { should: [], minimum_should_match: 1 } }
    const steps: TraceStep[] = [
      { step: 'query_analysis', data: { query: 'q', shape: 'KEYWORD', chunks: [], es_query: esQuery } },
    ]
    const { container } = render(<TracePanel steps={steps} />)

    await userEvent.click(screen.getByText('Copy'))

    expect(writeText).toHaveBeenCalledWith(JSON.stringify({ query: esQuery, size: 20 }, null, 2))
    expect(container.querySelector('details')?.open).toBe(false)
  })

  it('does not render the ES query block when es_query is absent', () => {
    const steps: TraceStep[] = [
      { step: 'query_analysis', data: { query: 'q', shape: 'KEYWORD', chunks: [] } },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.queryByText('Show ES query')).not.toBeInTheDocument()
  })

  it('shows the query_analysis chunks feeding the classifier decision on the classifier card', () => {
    const steps: TraceStep[] = [
      {
        step: 'query_analysis',
        data: {
          query: 'Section 54',
          shape: 'HYBRID',
          chunks: [{ text: 'Section 54', proximity: 0, type: 'section', alt_text: null }],
        },
      },
      {
        step: 'classifier',
        data: { label: 'HYBRID', confidence: 0.6, auto_route: true, plan: { es: true, milvus: true, fuse: true } },
      },
    ]
    render(<TracePanel steps={steps} />)

    const classifierCard = screen.getByRole('heading', { level: 3, name: 'Classifier' }).closest('section')!
    expect(within(classifierCard).getByText('"Section 54"')).toBeInTheDocument()
    expect(within(classifierCard).getByText(/section, slop 0/)).toBeInTheDocument()
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

  it('renders a keyword_search step with query, candidate count, and top doc ids', () => {
    const onOpenDocument = vi.fn()
    const steps: TraceStep[] = [
      {
        step: 'keyword_search',
        data: {
          query: 'section 55', mode: 'keyword', candidate_count: 20,
          top_doc_ids: ['d1', 'd2'],
        },
      },
    ]
    render(<TracePanel steps={steps} onOpenDocument={onOpenDocument} />)

    expect(screen.getByRole('heading', { level: 3, name: /keyword search/i })).toBeInTheDocument()
    expect(screen.getByText(/section 55/)).toBeInTheDocument()
    expect(screen.getByText(/20 candidate/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'd1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'd2' })).toBeInTheDocument()
  })

  it('renders a keyword_expansion step summary and its reasoning', () => {
    const steps: TraceStep[] = [
      {
        step: 'keyword_expansion',
        data: {
          query: 'Section 55', added_keywords: ['deemed income', 'non-resident'],
          reasoning: 'Section 55 concerns deemed income of non-residents, so both terms help.',
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByRole('heading', { level: 3, name: 'Keyword expansion' })).toBeInTheDocument()
    expect(screen.getByText(/\+2: deemed income, non-resident/)).toBeInTheDocument()
    expect(screen.getByText('Show reasoning')).toBeInTheDocument()
    expect(screen.getByText(/deemed income of non-residents, so both terms help/)).toBeInTheDocument()
  })

  it('shows "no keywords added" and still surfaces reasoning when expansion added nothing', () => {
    const steps: TraceStep[] = [
      {
        step: 'keyword_expansion',
        data: {
          query: 'Section 55', added_keywords: [],
          reasoning: 'Already a precise anchor; no synonym adds real recall here.',
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText('no keywords added')).toBeInTheDocument()
    expect(screen.getByText('Show reasoning')).toBeInTheDocument()
    expect(screen.getByText(/no synonym adds real recall here/)).toBeInTheDocument()
  })

  it('does not render a reasoning block when the step data has none', () => {
    const steps: TraceStep[] = [
      { step: 'keyword_search', data: { query: 'section 55', mode: 'keyword', candidate_count: 20, top_doc_ids: [] } },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.queryByText('Show reasoning')).not.toBeInTheDocument()
  })

  it('labels ai_milvus_sparse as ES fallback only when every hit is ES-origin', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          collections: [{
            name: 'act_section', hit_count: 20,
            top_hits: [{ chunk_id: 'es:d1:0', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'es' }],
          }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByRole('heading', { level: 3, name: 'ES fallback sparse search' })).toBeInTheDocument()
    expect(screen.getByText(/ES fallback only/)).toBeInTheDocument()
  })

  it('labels ai_milvus_sparse as Milvus sparse search when every hit is Milvus-origin', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          collections: [{
            name: 'held', hit_count: 3,
            top_hits: [{ chunk_id: 'c1', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'milvus' }],
          }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByRole('heading', { level: 3, name: 'Milvus sparse search' })).toBeInTheDocument()
    expect(screen.getByText(/Milvus native only/)).toBeInTheDocument()
  })

  it('labels ai_milvus_sparse as round-robin merged when both origins are present', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          collections: [
            {
              name: 'held', hit_count: 3,
              top_hits: [{ chunk_id: 'c1', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'milvus' }],
            },
            {
              name: 'act_section', hit_count: 20,
              top_hits: [{ chunk_id: 'es:d2:0', doc_id: 'd2', score: 1.0, text_preview: 't', origin: 'es' }],
            },
          ],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(
      screen.getByRole('heading', { level: 3, name: 'Milvus + ES fallback sparse (round-robin merged)' }),
    ).toBeInTheDocument()
    expect(screen.getAllByText(/round-robin merged/).length).toBeGreaterThanOrEqual(2)
  })

  it('shows the query ai_milvus_dense actually searched with', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_dense',
        data: { query: 'rewritten by LLM', collections: [{ name: 'ruling', hit_count: 1, top_hits: [] }] },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText('rewritten by LLM')).toBeInTheDocument()
  })

  it('shows only the ES fallback query (not a Milvus line) when ai_milvus_sparse ran ES-only, plus a copyable ES query body', async () => {
    const esQueryBody = { bool: { must: [{ terms: { 'groups.group.name.keyword': ['ACT'] } }] } }
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          milvus_query: 'rewritten by LLM',
          es_query: 'section 55',
          es_query_body: esQueryBody,
          collections: [{
            name: 'act_section', hit_count: 20,
            top_hits: [{ chunk_id: 'es:d1:0', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'es' }],
          }],
        },
      },
    ]
    const { container } = render(<TracePanel steps={steps} />)

    // ES actually ran (act_section has no native sparse) - Milvus contributed nothing to
    // this step, so "Milvus searched" must not appear even though milvus_query is present.
    expect(screen.queryByText('rewritten by LLM')).not.toBeInTheDocument()
    expect(screen.getByText('section 55')).toBeInTheDocument()
    expect(screen.getByText('Show ES query')).toBeInTheDocument()

    const details = container.querySelectorAll('details')
    const esQueryDetails = Array.from(details).find((el) => el.textContent?.includes('Show ES query'))
    expect(esQueryDetails?.textContent).toContain('"size": 20')

    const writeText = vi.fn()
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, writable: true, configurable: true })
    await userEvent.click(screen.getByText('Copy'))
    expect(writeText).toHaveBeenCalledWith(JSON.stringify({ query: esQueryBody, size: 20 }, null, 2))
  })

  it('shows both queries for ai_milvus_sparse when the step genuinely mixes Milvus and ES origins', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          milvus_query: 'rewritten by LLM',
          es_query: 'section 55',
          es_query_body: { bool: { must: [] } },
          collections: [
            {
              name: 'held', hit_count: 3,
              top_hits: [{ chunk_id: 'c1', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'milvus' }],
            },
            {
              name: 'act_section', hit_count: 20,
              top_hits: [{ chunk_id: 'es:d2:0', doc_id: 'd2', score: 1.0, text_preview: 't', origin: 'es' }],
            },
          ],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText('rewritten by LLM')).toBeInTheDocument()
    expect(screen.getByText('section 55')).toBeInTheDocument()
  })

  it('does not show an ES fallback query line or ES query block when ai_milvus_sparse ran Milvus-native only', () => {
    const steps: TraceStep[] = [
      {
        step: 'ai_milvus_sparse',
        data: {
          milvus_query: 'q',
          collections: [{
            name: 'held', hit_count: 3,
            top_hits: [{ chunk_id: 'c1', doc_id: 'd1', score: 1.0, text_preview: 't', origin: 'milvus' }],
          }],
        },
      },
    ]
    render(<TracePanel steps={steps} />)

    expect(screen.getByText('q')).toBeInTheDocument()
    expect(screen.queryByText('Show ES query')).not.toBeInTheDocument()
  })

})
