import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { useAdminEvalRun } from './useAdminEvalRun'

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

describe('useAdminEvalRun', () => {
  it('sends suite/token/limit and accumulates case/progress events', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))

    act(() => result.current.run('slm_intent', 'tok', 10))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0])).toEqual({ suite: 'slm_intent', token: 'tok', limit: 10 })
    expect(result.current.running).toBe(true)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'case', id: 'S01', query: 'q', status: 'pass', detail: {} }) })
    })
    expect(result.current.cases).toEqual([{ type: 'case', id: 'S01', query: 'q', status: 'pass', detail: {} }])
    expect(result.current.passed).toBe(1)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'progress', done: 1, total: 2, percent: 50 }) })
    })
    expect(result.current.percent).toBe(50)
    expect(result.current.total).toBe(2)

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'done', summary: { total: 2, passed: 1 } }) })
    })
    await waitFor(() => expect(result.current.running).toBe(false))
  })

  it('surfaces a server error event and stops running', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    act(() => result.current.run('slm_intent', 'bad-token'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })
    })
    await waitFor(() => expect(result.current.running).toBe(false))
    expect(result.current.error).toBe('unauthorized')
  })

  it('marks the run interrupted if the socket closes while still running', async () => {
    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    act(() => result.current.run('slm_intent', 'tok'))
    const socket = FakeWebSocket.instances[0]

    act(() => {
      socket.onclose?.()
    })
    await waitFor(() => expect(result.current.running).toBe(false))
    expect(result.current.error).toBe('Run interrupted.')
  })

  it('loadCached hydrates state from a prior run without setting running', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        summary: { total: 2, passed: 1 },
        cases: [
          { type: 'case', id: 'S01', query: 'q1', status: 'pass', detail: {} },
          { type: 'case', id: 'S02', query: 'q2', status: 'fail', detail: {} },
        ],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    await act(async () => {
      await result.current.loadCached('http://x', 'slm_intent', 'tok')
    })

    expect(fetchMock).toHaveBeenCalledWith('http://x/admin/api/eval-runs/slm_intent', {
      headers: { 'X-Admin-Token': 'tok' },
    })
    expect(result.current.running).toBe(false)
    expect(result.current.cases).toHaveLength(2)
    expect(result.current.total).toBe(2)
    expect(result.current.passed).toBe(1)
    expect(result.current.percent).toBe(100)
  })

  it('loadCached leaves state untouched when no run has completed yet (null response)', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => null })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useAdminEvalRun('ws://x/ws/admin-eval'))
    await act(async () => {
      await result.current.loadCached('http://x', 'slm_intent', 'tok')
    })

    expect(result.current.cases).toEqual([])
    expect(result.current.total).toBe(0)
  })
})
