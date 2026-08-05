import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })
})

vi.mock('./api/useSearch', () => ({
  useSearch: () => ({
    loading: false,
    instant: null,
    aiMode: null,
    traceSteps: [{ step: 'intent', data: { query: 'q', rewritten_query: 'q', intent: 'x', filters: {} } }],
    wsError: null,
    search: () => {},
  }),
}))

describe('App with a trace', () => {
  it('shows the TracePanel in a two-column layout when dev mode is on', () => {
    window.history.pushState({}, '', '/?dev=1')
    render(<App />)
    expect(screen.getByText(/Intent/)).toBeInTheDocument()
  })
})
