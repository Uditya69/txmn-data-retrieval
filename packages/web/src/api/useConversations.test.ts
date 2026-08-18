import { describe, expect, it, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { useConversations } from './useConversations'

describe('useConversations', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('does not fetch and returns an empty list when there is no token', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const { result } = renderHook(() => useConversations('http://api', null))

    await act(async () => {
      await result.current.refresh()
    })

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(result.current.conversations).toEqual([])
  })

  it('fetches and stores the conversation list when a token is present', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => [{ id: 'conv-1', title: 'q1', updated_at: '2026-08-18T00:00:00Z' }],
    } as Response)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))

    await act(async () => {
      await result.current.refresh()
    })

    await waitFor(() => {
      expect(result.current.conversations).toEqual([{ id: 'conv-1', title: 'q1', updated_at: '2026-08-18T00:00:00Z' }])
    })
  })

  it('loadConversation hydrates the server\'s flat {role,text} records into ChatMessages', async () => {
    // Realistic server response: chat/repository.py persists flat
    // {role, text} dicts, not the frontend's rich ChatMessage shape (which
    // only ever exists in-memory - results, activeMode, trace steps, etc).
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'conv-1', title: 'q1', created_at: 'x', updated_at: 'x',
        messages: [
          { role: 'user', text: 'what is section 80HH' },
          { role: 'assistant', text: 'Section 80HH provides a deduction...' },
        ],
      }),
    } as Response)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))
    const messages = await result.current.loadConversation('conv-1')

    expect(messages).toEqual([
      { id: 'conv-1-0', role: 'user', text: 'what is section 80HH' },
      {
        id: 'conv-1-1',
        role: 'assistant',
        question: 'what is section 80HH',
        activeMode: 'classic',
        results: {
          classic: {
            status: 'done',
            aiMode: { ok: true, answer: 'Section 80HH provides a deduction...', citations: {} },
            traceSteps: [],
          },
        },
      },
    ])

    // Every hydrated assistant message must carry a `results` field keyed by
    // its activeMode - ChatMessageView does `message.results[message.activeMode]`
    // unconditionally, so a missing `results` field crashes the reopen.
    for (const m of messages) {
      if (m.role === 'assistant') {
        expect(m.results).toBeDefined()
        expect(m.results[m.activeMode]).toBeDefined()
      }
    }
  })

  it('remove calls DELETE and drops the conversation from local state', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => [] } as Response)
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))
    await act(async () => {
      await result.current.remove('conv-1')
    })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://api/conversations/conv-1',
      expect.objectContaining({ method: 'DELETE' }),
    )
  })
})
