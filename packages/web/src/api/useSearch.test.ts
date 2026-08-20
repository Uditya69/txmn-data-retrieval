import { describe, expect, it, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSearch } from './useSearch'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  listeners: Record<string, Array<(event: unknown) => void>> = {}
  sent: string[] = []
  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }
  addEventListener(type: string, listener: (event: unknown) => void) {
    ;(this.listeners[type] ??= []).push(listener)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
  emit(type: string, event: unknown = {}) {
    for (const listener of this.listeners[type] ?? []) listener(event)
  }
}

beforeEach(() => {
  MockWebSocket.instances = []
  // @ts-expect-error - test double, not a full WebSocket implementation
  global.WebSocket = MockWebSocket
})

describe('useSearch', () => {
  it('omits access_token from the payload when none is provided (guest)', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).not.toHaveProperty('access_token')
  })

  it('includes access_token in the payload when a token is provided', () => {
    const { result } = renderHook(() => useSearch('ws://test', 'tok-abc'))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).toMatchObject({ access_token: 'tok-abc' })
  })

  it('includes conversation_id in the payload when provided', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true, 'both', false, false, 'conv-42')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).toMatchObject({ conversation_id: 'conv-42' })
  })

  it('omits conversation_id from the payload when not provided', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(JSON.parse(socket.sent[0])).not.toHaveProperty('conversation_id')
  })

  it('sends the query with mode "both" and the trace flag once the socket opens, and stores the instant result', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(socket.sent).toEqual([JSON.stringify({ query: 'cgst', mode: 'both', trace: true, rerank: false, rrf: false })])

    act(() => {
      socket.emit('message', {
        data: JSON.stringify({
          type: 'instant_result',
          es: [{ doc_id: 'd1', score: 1, snippet: 's' }],
          es_error: null,
          milvus: null,
          milvus_error: null,
        }),
      })
    })

    expect(result.current.instant).toEqual({
      es: [{ doc_id: 'd1', score: 1, snippet: 's' }],
      es_error: null,
      milvus: null,
      milvus_sparse: null,
      milvus_error: null,
      reranked: null,
      reranked_error: null,
    })
    expect(result.current.loading).toBe(true)
  })

  it('marks loading false and stores the answer on ai_mode_done', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst', false)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_done', answer: 'answer text', citations: {} }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: true, answer: 'answer text', citations: {}, reasoning: null })
  })

  it('stores the reasoning trace when the backend provides one', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst', false)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_done', answer: 'answer text', citations: {}, reasoning: 'step by step...' }),
      })
    })

    expect(result.current.aiMode).toEqual({ ok: true, answer: 'answer text', citations: {}, reasoning: 'step by step...' })
  })

  it('marks loading false and stores the error on ai_mode_error', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst', false)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_error', error: 'boom' }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: false, error: 'boom' })
  })

  it('accumulates ai_mode_trace messages into traceSteps, in arrival order', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'intent', data: { search_query: 'r', intent: ['caselaws'] } }),
      })
    })
    expect(result.current.traceSteps).toEqual([{ step: 'intent', data: { search_query: 'r', intent: ['caselaws'] } }])

    act(() => {
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'filters_resolved', data: { doc_id_count: 0 } }),
      })
    })
    expect(result.current.traceSteps).toEqual([
      { step: 'intent', data: { search_query: 'r', intent: ['caselaws'] } },
      { step: 'filters_resolved', data: { doc_id_count: 0 } },
    ])
  })

  it('resets traceSteps to empty when a new search starts', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('first query', true)
    })
    let socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', {
        data: JSON.stringify({ type: 'ai_mode_trace', step: 'intent', data: {} }),
      })
    })
    expect(result.current.traceSteps).toHaveLength(1)

    act(() => {
      result.current.search('second query', true)
    })
    expect(result.current.traceSteps).toEqual([])
  })

  it('sets a wsError and calls onSessionExpired when the server reports session_expired', () => {
    const onSessionExpired = vi.fn()
    const { result } = renderHook(() => useSearch('ws://test', 'stale-token', onSessionExpired))

    act(() => {
      result.current.search('cgst', true)
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'session_expired' }) })
    })

    expect(result.current.wsError).toBe('Reconnecting your session…')
    expect(onSessionExpired).toHaveBeenCalledOnce()
  })
})
