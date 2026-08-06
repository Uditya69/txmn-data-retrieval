import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { useSearch } from './api/useSearch'

describe('App', () => {
  it('renders the page title', () => {
    render(<App />, { wrapper: MemoryRouter })
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })
})

vi.mock('./api/useSearch', () => ({
  useSearch: vi.fn(() => ({
    loading: false,
    instant: null,
    aiMode: null,
    traceSteps: [{ step: 'intent', data: { query: 'q', rewritten_query: 'q', intent: 'x', filters: {} } }],
    wsError: null,
    search: () => {},
  })),
}))

describe('App with a trace', () => {
  it('shows the TracePanel in a two-column layout when dev mode is on', () => {
    window.history.pushState({}, '', '/?dev=1')
    render(<App />, { wrapper: MemoryRouter })
    expect(screen.getByText(/Intent/)).toBeInTheDocument()
  })

  it('does not show the TracePanel when dev mode is on but there are no trace steps yet', () => {
    vi.mocked(useSearch).mockReturnValue({
      loading: false,
      instant: null,
      aiMode: null,
      traceSteps: [],
      wsError: null,
      search: () => {},
    })
    window.history.pushState({}, '', '/?dev=1')
    render(<App />, { wrapper: MemoryRouter })
    expect(screen.queryByText(/Intent/)).not.toBeInTheDocument()
  })
})
