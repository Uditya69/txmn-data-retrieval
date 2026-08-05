import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import DocumentModal from './DocumentModal'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('DocumentModal', () => {
  it('renders nothing when docId is null', () => {
    const { container } = render(
      <DocumentModal docId={null} apiBaseUrl="http://api" onClose={vi.fn()} onNavigate={vi.fn()} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('shows a loading state, then renders fetched blocks with citation links', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          doc_id: 'd1',
          blocks: [
            { type: 'paragraph', text: 'Plain text.', links: [] },
            {
              type: 'paragraph',
              text: 'See [1957] 32 ITR 592 (Raj.).',
              links: [{ text: '[1957] 32 ITR 592 (Raj.)', doc_id: 'd2' }],
            },
          ],
        }),
      }),
    )

    render(<DocumentModal docId="d1" apiBaseUrl="http://api" onClose={vi.fn()} onNavigate={vi.fn()} />)

    expect(screen.getByTestId('document-modal-loading')).toBeInTheDocument()

    await waitFor(() => expect(screen.getByText('Plain text.')).toBeInTheDocument())
    expect(fetch).toHaveBeenCalledWith('http://api/documents/d1')
    expect(screen.getByText('[1957] 32 ITR 592 (Raj.)')).toBeInTheDocument()
  })

  it('calls onNavigate with the linked doc_id when a citation link is clicked', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          doc_id: 'd1',
          blocks: [
            {
              type: 'paragraph',
              text: 'See case.',
              links: [{ text: 'case', doc_id: 'd2' }],
            },
          ],
        }),
      }),
    )
    const onNavigate = vi.fn()

    render(<DocumentModal docId="d1" apiBaseUrl="http://api" onClose={vi.fn()} onNavigate={onNavigate} />)

    await waitFor(() => expect(screen.getByText('case')).toBeInTheDocument())
    fireEvent.click(screen.getByText('case'))

    expect(onNavigate).toHaveBeenCalledWith('d2')
  })

  it('shows an error message when the fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({}) }))

    render(<DocumentModal docId="missing" apiBaseUrl="http://api" onClose={vi.fn()} onNavigate={vi.fn()} />)

    await waitFor(() => expect(screen.getByText(/could not load document/i)).toBeInTheDocument())
  })

  it('calls onClose when the close button is clicked', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ doc_id: 'd1', blocks: [] }) }),
    )
    const onClose = vi.fn()

    render(<DocumentModal docId="d1" apiBaseUrl="http://api" onClose={onClose} onNavigate={vi.fn()} />)

    await waitFor(() => expect(screen.queryByTestId('document-modal-loading')).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /close/i }))

    expect(onClose).toHaveBeenCalled()
  })
})
