import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useAgentSearch', () => ({ useAgentSearch: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
}

function baseAgentState() {
  return { loading: false, traceSteps: [], result: null, wsError: null, search: vi.fn() }
}

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.mocked(useSearch).mockReturnValue(baseSearchState())
    vi.mocked(useAgentSearch).mockReturnValue(baseAgentState())
  })

  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })

  it('renders the Classic/Agent mode toggle', () => {
    render(<App />)
    expect(screen.getByText('classic')).toBeInTheDocument()
    expect(screen.getByText('agent')).toBeInTheDocument()
  })

  it('submits a question via the chat input and triggers classic search', () => {
    const search = vi.fn()
    vi.mocked(useSearch).mockReturnValue({ ...baseSearchState(), search })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'what is section 80HH' } })
    fireEvent.click(screen.getByText('Send'))

    expect(search).toHaveBeenCalledWith('what is section 80HH', true, 'both')
    expect(screen.getAllByText('what is section 80HH').length).toBeGreaterThan(0)
  })
})
