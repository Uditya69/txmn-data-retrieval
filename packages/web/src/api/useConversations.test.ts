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

  it('loadConversation returns the messages from the detail endpoint', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'conv-1', title: 'q1', created_at: 'x', updated_at: 'x',
        messages: [{ id: 'm1', role: 'user', text: 'hi' }],
      }),
    } as Response)

    const { result } = renderHook(() => useConversations('http://api', 'token-123'))
    const messages = await result.current.loadConversation('conv-1')

    expect(messages).toEqual([{ id: 'm1', role: 'user', text: 'hi' }])
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
