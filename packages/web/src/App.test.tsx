import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { useSearch } from './api/useSearch'
import { useConversations } from './api/useConversations'
import { useAuth } from './api/useAuth'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useConversations', () => ({ useConversations: vi.fn() }))
vi.mock('./api/useAuth', () => ({ useAuth: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
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
    vi.mocked(useConversations).mockReturnValue(baseConversationsState())
    vi.mocked(useAuth).mockReturnValue(baseAuthState())
  })

  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
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

    expect(search).toHaveBeenCalledWith(
      'what is section 80HH', true, 'both', true, true, undefined, true, 'repotaxmannapi',
    )
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

  // Regression test for C2 (2026-09-02 final review): fetchInstantPage (the paged
  // re-fetch triggered by Next/Prev when VITE_ENABLE_PAGINATION is on) must not null out
  // an already-rendered aiMode/'done' status on the message it targets - it should patch
  // only `instant` into the existing ResultState, leaving aiMode/status untouched.
  it('a paged instant-only re-fetch does not clobber an already-done aiMode answer', async () => {
    vi.resetModules()
    vi.stubEnv('VITE_ENABLE_PAGINATION', 'true')
    try {
      const search = vi.fn()
      const manyEsHits = Array.from({ length: 25 }, (_, i) => ({ doc_id: `d${i}`, score: 1, heading: `h${i}`, subheading: '' }))

      // Phase 1: mid-flight, matches the state right after handleSubmit fires - loading,
      // nothing back from the server yet.
      vi.mocked(useSearch).mockReturnValue({ ...baseSearchState(), search })

      const { default: FreshApp } = await import('./App')
      const { rerender } = render(<FreshApp />)

      fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'what is section 80HH' } })
      fireEvent.click(screen.getByLabelText('Send'))

      // Phase 2: the full ('both'-mode) turn completes - AI Mode's answer is done and
      // Instant has its first page of results. Simulates classicSearch's own state update
      // (a real hook re-renders its consumer with a new object on every state change).
      vi.mocked(useSearch).mockReturnValue({
        loading: false,
        instant: { es: manyEsHits, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null },
        aiMode: { ok: true, answer: 'The final answer.', citations: {} },
        traceSteps: [],
        wsError: null,
        search,
      })
      rerender(<FreshApp />)

      expect(screen.getByText(/The final answer\./)).toBeInTheDocument()

      // Next click triggers fetchInstantPage - an instant-only ('instant' mode) re-fetch,
      // via the real (un-mocked) ChatMessageView/InstantPane Next button, wired through
      // paginationEnabled=true now that the env flag is on.
      fireEvent.click(screen.getByText('Next'))
      expect(search).toHaveBeenLastCalledWith(
        'what is section 80HH', true, 'instant', true, true, undefined, true, 'repotaxmannapi', 2, 20,
      )

      // Phase 3: the instant-only fetch's response lands. Before the C2 fix, the
      // reflect-effect would recompute status as classicSearch.aiMode ? 'done' : 'loading'
      // off of THIS instant-only classicSearch state - but mode:'instant' never populates
      // aiMode, so it evaluates to permanently 'loading' and overwrites the message's
      // already-rendered aiMode with null.
      const pagedEsHits = Array.from({ length: 25 }, (_, i) => ({ doc_id: `p${i}`, score: 1, heading: `p${i}`, subheading: '' }))
      vi.mocked(useSearch).mockReturnValue({
        loading: false,
        instant: { es: pagedEsHits, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null },
        aiMode: null,
        traceSteps: [],
        wsError: null,
        search,
      })
      rerender(<FreshApp />)

      // The already-rendered answer must still be there - not nulled out/stuck loading.
      expect(screen.getAllByText(/The final answer\./).length).toBeGreaterThan(0)
    } finally {
      vi.unstubAllEnvs()
      vi.resetModules()
    }
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
})
