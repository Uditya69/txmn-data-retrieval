import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useAgentSearch } from './useAgentSearch'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  sent: string[] = []
  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }
  addEventListener(type: string, handler: any) {
    if (type === 'open') {
      this.onopen = handler
      // Trigger onopen immediately to simulate connection
      handler()
    }
    if (type === 'message') this.onmessage = handler
    if (type === 'error') this.onerror = handler
    if (type === 'close') this.onclose = handler
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
}

beforeEach(() => {
  FakeWebSocket.instances = []
  // @ts-expect-error test stub
  global.WebSocket = FakeWebSocket
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useAgentSearch', () => {
  it('sends the query and reports the final answer', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))

    act(() => result.current.search('gst rate'))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0])).toEqual({ query: 'gst rate' })

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'ai_mode_trace', step: 'agent_tool_call', data: { name: 'search_es', arguments: {} } }) })
    })
    expect(result.current.traceSteps).toEqual([{ step: 'agent_tool_call', data: { name: 'search_es', arguments: {} } }])

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_done', answer: 'See [d1].', doc_ids: ['d1'] }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result).toEqual({ ok: true, answer: 'See [d1].', docIds: ['d1'] })
  })

  it('reports an unverifiable answer as a non-ok result', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))
    act(() => result.current.search('q'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_unverifiable', invalid_doc_ids: ['d999'] }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result?.ok).toBe(false)
    expect((result.current.result as { ok: false; error: string }).error).toContain('d999')
  })

  it('reports a pipeline error as a non-ok result', async () => {
    const { result } = renderHook(() => useAgentSearch('ws://x/ws/agent'))
    act(() => result.current.search('q'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'agent_error', error: 'RuntimeError: gateway down' }) })
    })

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.result).toEqual({ ok: false, error: 'RuntimeError: gateway down' })
  })
})
