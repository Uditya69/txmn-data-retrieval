import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import DocumentsFeed from './DocumentsFeed'

describe('DocumentsFeed', () => {
  it('shows a placeholder before any search has been made', () => {
    render(<DocumentsFeed instant={null} aiMode={null} devMode={false} highlightedDocId={null} />)
    expect(screen.getByText('Search to see documents.')).toBeInTheDocument()
  })

  it('shows a no-results message when both legs are empty', () => {
    render(
      <DocumentsFeed
        instant={{ es: [], es_error: null, milvus: {}, milvus_error: null }}
        aiMode={null}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('No results found.')).toBeInTheDocument()
  })

  it('renders merged cards with a result count', () => {
    render(
      <DocumentsFeed
        instant={{
          es: [{ doc_id: 'd1', score: 10, heading: 'Heading about capital gains', subheading: 'Party A vs. Party B' }],
          es_error: null,
          milvus: { facts: [{ chunk_id: 'd2::facts::0', doc_id: 'd2', text: 'Milvus snippet text', score: 5 }] },
          milvus_error: null,
        }}
        aiMode={null}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('Heading about capital gains')).toBeInTheDocument()
    expect(screen.getByText(/Milvus snippet text/)).toBeInTheDocument()
  })

  it('caps rendered cards at 20, keeping the highest-scored ones', () => {
    const es = Array.from({ length: 25 }, (_, i) => ({
      doc_id: `d${i}`,
      score: i, // d24 highest, d0 lowest
      heading: `heading ${i}`,
      subheading: `snippet ${i}`,
    }))
    render(
      <DocumentsFeed
        instant={{ es, es_error: null, milvus: null, milvus_error: null }}
        aiMode={null}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('20')).toBeInTheDocument()
    expect(screen.getByText(/snippet 24/)).toBeInTheDocument()
    expect(screen.getByText(/snippet 5/)).toBeInTheDocument()
    expect(screen.queryByText(/snippet 4$/)).not.toBeInTheDocument()
    expect(screen.queryByText(/snippet 0$/)).not.toBeInTheDocument()
  })

  it('shows a cited badge derived from the AI Mode answer text', () => {
    render(
      <DocumentsFeed
        instant={{
          es: [{ doc_id: 'd1', score: 10, heading: 'Heading', subheading: 'ES snippet' }],
          es_error: null,
          milvus: null,
          milvus_error: null,
        }}
        aiMode={{ ok: true, answer: 'Cited once [d1] and again [d1].', citations: {} }}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('Cited 2')).toBeInTheDocument()
  })
})
