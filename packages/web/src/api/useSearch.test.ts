import { describe, expect, it, beforeEach } from 'vitest'
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
  it('sends the query with mode "both" once the socket opens, and stores the instant result', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(socket.sent).toEqual([JSON.stringify({ query: 'cgst', mode: 'both' })])

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
      milvus_error: null,
    })
    expect(result.current.loading).toBe(true)
  })

  it('marks loading false and stores the answer on ai_mode_done', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_done', answer: 'answer text', citations: {} }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: true, answer: 'answer text', citations: {} })
  })

  it('marks loading false and stores the error on ai_mode_error', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_error', error: 'boom' }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: false, error: 'boom' })
  })
})
