import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import { useConversations } from './api/useConversations'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useAgentSearch', () => ({ useAgentSearch: vi.fn() }))
vi.mock('./api/useConversations', () => ({ useConversations: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
}

function baseAgentState() {
  return { loading: false, traceSteps: [], result: null, wsError: null, search: vi.fn() }
}

function baseConversationsState() {
  return { conversations: [], refresh: vi.fn(), loadConversation: vi.fn(), remove: vi.fn() }
}

describe('App', () => {
  beforeEach(() => {
    vi.mocked(useSearch).mockReturnValue(baseSearchState())
    vi.mocked(useAgentSearch).mockReturnValue(baseAgentState())
    vi.mocked(useConversations).mockReturnValue(baseConversationsState())
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

  it('defaults dev mode on with no ?dev URL param', () => {
    render(<App />)
    expect(screen.getByLabelText('Dev mode', { selector: 'input' })).toBeChecked()
  })

  it('turns dev mode off when the URL has ?dev=0', () => {
    window.history.pushState({}, '', '/?dev=0')
    render(<App />)
    expect(screen.getByLabelText('Dev mode', { selector: 'input' })).not.toBeChecked()
    window.history.pushState({}, '', '/')
  })

  it('submits a question via the chat input and triggers classic search', () => {
    const search = vi.fn()
    vi.mocked(useSearch).mockReturnValue({ ...baseSearchState(), search })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'what is section 80HH' } })
    fireEvent.click(screen.getByLabelText('Send'))

    expect(search).toHaveBeenCalledWith('what is section 80HH', true, 'both', false, undefined)
    expect(screen.getAllByText('what is section 80HH').length).toBeGreaterThan(0)
  })

  it('never touches localStorage', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')
    render(<App />)
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'test' } })
    fireEvent.click(screen.getByLabelText('Send'))
    expect(setItemSpy).not.toHaveBeenCalled()
    setItemSpy.mockRestore()
  })
})
