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
          es: [{ doc_id: 'd1', score: 10, snippet: 'ES snippet about capital gains' }],
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
    expect(screen.getByText(/ES snippet about capital gains/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus snippet text/)).toBeInTheDocument()
  })

  it('shows a cited badge derived from the AI Mode answer text', () => {
    render(
      <DocumentsFeed
        instant={{
          es: [{ doc_id: 'd1', score: 10, snippet: 'ES snippet' }],
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
