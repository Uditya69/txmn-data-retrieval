import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import AgentPage from './AgentPage'
import { useAgentSearch } from './api/useAgentSearch'

vi.mock('./api/useAgentSearch', () => ({
  useAgentSearch: vi.fn(() => ({
    loading: false,
    traceSteps: [],
    result: null,
    wsError: null,
    search: vi.fn(),
  })),
}))

describe('AgentPage', () => {
  it('renders a heading, a link back to search, and the trace panel placeholder', () => {
    render(<AgentPage />, { wrapper: MemoryRouter })
    expect(screen.getByText(/Agentic Search/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /back to search/i })).toBeInTheDocument()
    expect(screen.getByText(/no trace yet/i)).toBeInTheDocument()
  })

  it('calls search with the typed query', async () => {
    const search = vi.fn()
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: null, wsError: null, search,
    })
    const { default: userEvent } = await import('@testing-library/user-event')
    const user = userEvent.setup()
    render(<AgentPage />, { wrapper: MemoryRouter })

    await user.type(screen.getByRole('textbox'), 'gst rate{enter}')

    expect(search).toHaveBeenCalledWith('gst rate')
  })

  it('renders a successful cited answer with its doc_ids', () => {
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: { ok: true, answer: 'See [d1].', docIds: ['d1'] }, wsError: null, search: vi.fn(),
    })
    render(<AgentPage />, { wrapper: MemoryRouter })

    expect(screen.getByText(/See \[d1\]\./)).toBeInTheDocument()
    expect(screen.getByText(/d1/)).toBeInTheDocument()
  })

  it('renders an unverifiable/error result distinctly from a successful answer', () => {
    vi.mocked(useAgentSearch).mockReturnValue({
      loading: false, traceSteps: [], result: { ok: false, error: 'Could not produce a fully cited answer.' }, wsError: null, search: vi.fn(),
    })
    render(<AgentPage />, { wrapper: MemoryRouter })

    expect(screen.getByText(/Could not produce a fully cited answer\./)).toBeInTheDocument()
  })
})
