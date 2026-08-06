import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import DebugPage from './DebugPage'
import { useSearch } from './api/useSearch'

vi.mock('./api/useSearch', () => ({
  useSearch: vi.fn(() => ({
    loading: false,
    instant: null,
    aiMode: null,
    traceSteps: [{ step: 'es_search', data: { hits: [{ doc_id: 'd1', score: 1, heading: 'H', subheading: 'S' }] } }],
    wsError: null,
    search: vi.fn(),
  })),
}))

describe('DebugPage', () => {
  it('renders the trace panel with retrieval steps and a link back to search', () => {
    render(<DebugPage />, { wrapper: MemoryRouter })
    expect(screen.getByText(/Retrieval Debug/)).toBeInTheDocument()
    expect(screen.getByText(/ES search/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /back to search/i })).toBeInTheDocument()
  })

  it('calls search in instant mode with trace enabled', async () => {
    const search = vi.fn()
    vi.mocked(useSearch).mockReturnValue({
      loading: false,
      instant: null,
      aiMode: null,
      traceSteps: [],
      wsError: null,
      search,
    })
    const { default: userEvent } = await import('@testing-library/user-event')
    const user = userEvent.setup()
    render(<DebugPage />, { wrapper: MemoryRouter })

    await user.type(screen.getByRole('textbox'), 'cgst{enter}')

    expect(search).toHaveBeenCalledWith('cgst', true, 'instant')
  })
})
