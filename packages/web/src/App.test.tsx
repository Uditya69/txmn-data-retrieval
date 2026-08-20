import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import { useConversations } from './api/useConversations'
import { useAuth } from './api/useAuth'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useAgentSearch', () => ({ useAgentSearch: vi.fn() }))
vi.mock('./api/useConversations', () => ({ useConversations: vi.fn() }))
vi.mock('./api/useAuth', () => ({ useAuth: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
}

function baseAgentState() {
  return { loading: false, traceSteps: [], result: null, wsError: null, search: vi.fn() }
}

function baseConversationsState() {
  return { conversations: [], refresh: vi.fn(), loadConversation: vi.fn(), remove: vi.fn(), clear: vi.fn() }
}

function baseAuthState() {
  return {
    token: null as string | null,
    email: null as string | null,
    loading: false,
    error: null,
    signup: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
    refresh: vi.fn(),
  }
}

describe('App', () => {
  beforeEach(() => {
    vi.mocked(useSearch).mockReturnValue(baseSearchState())
    vi.mocked(useAgentSearch).mockReturnValue(baseAgentState())
    vi.mocked(useConversations).mockReturnValue(baseConversationsState())
    vi.mocked(useAuth).mockReturnValue(baseAuthState())
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

    expect(search).toHaveBeenCalledWith('what is section 80HH', true, 'both', false, false, undefined)
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

  it('clears the remote conversation list synchronously on every auth token change, so a stale user\'s chats never leak into the next session', () => {
    const clear = vi.fn()
    vi.mocked(useConversations).mockReturnValue({ ...baseConversationsState(), clear })

    vi.mocked(useAuth).mockReturnValue({ ...baseAuthState(), token: 'token-a', email: 'a@example.com' })
    const { rerender } = render(<App />)
    expect(clear).toHaveBeenCalledTimes(1)

    // Switching to a different logged-in user (new truthy token) must clear
    // the previous user's list before the new one's refresh() resolves.
    vi.mocked(useAuth).mockReturnValue({ ...baseAuthState(), token: 'token-b', email: 'b@example.com' })
    rerender(<App />)
    expect(clear).toHaveBeenCalledTimes(2)

    // Logging out must also clear it.
    vi.mocked(useAuth).mockReturnValue({ ...baseAuthState(), token: null, email: null })
    rerender(<App />)
    expect(clear).toHaveBeenCalledTimes(3)
  })

  it('keeps a not-yet-persisted agent-mode conversation visible in the sidebar for a logged-in user', () => {
    // /ws/agent doesn't wire conversation_id through (out of scope for this
    // fix wave), so an agent-mode conversation never appears in the remote
    // list. The sidebar must still source it from local state so it isn't
    // unreachable once "New chat" is clicked.
    vi.mocked(useAuth).mockReturnValue({ ...baseAuthState(), token: 'token-a', email: 'a@example.com' })
    vi.mocked(useConversations).mockReturnValue(baseConversationsState())
    render(<App />)

    fireEvent.click(screen.getByText('agent'))
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'agent mode question' } })
    fireEvent.click(screen.getByLabelText('Send'))

    expect(screen.getByRole('button', { name: 'agent mode question' })).toBeInTheDocument()
  })
})
