import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SuiteRunner from './SuiteRunner'

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
  // Mount always calls loadCached() first (see spec's "survive a page refresh"
  // goal) - stub a no-prior-run response by default so every test's mount step
  // doesn't need its own fetch mock; the cache-hydration test below overrides this.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => null }))
})

describe('SuiteRunner', () => {
  it('runs the selected suite and renders a case row as it streams in', async () => {
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    expect(JSON.parse(socket.sent[0]).suite).toBe('slm_intent')

    socket.onmessage?.({
      data: JSON.stringify({ type: 'case', id: 'S01', query: 'case law for X', status: 'pass', detail: { rewrite: 'X' } }),
    })

    expect(await screen.findByText('S01')).toBeInTheDocument()
    expect(screen.getByText('case law for X')).toBeInTheDocument()
    expect(screen.getByText('pass')).toBeInTheDocument()
  })

  it('calls onUnauthorized when the server reports an unauthorized error', async () => {
    const onUnauthorized = vi.fn()
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="bad" onUnauthorized={onUnauthorized} />)

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })

    // socket.onmessage is invoked directly here (not via userEvent/act), so the
    // resulting state update is scheduled rather than flushed synchronously -
    // wait for it the same way useAdminEvalRun.test.ts does for the same event.
    await waitFor(() => expect(onUnauthorized).toHaveBeenCalled())
  })

  it('hydrates from a cached run on mount, without opening a WS connection', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          summary: { total: 1, passed: 1 },
          cases: [{ type: 'case', id: 'S01', query: 'cached query', status: 'pass', detail: {} }],
        }),
      }),
    )

    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    expect(await screen.findByText('cached query')).toBeInTheDocument()
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it('disables suite-picker buttons while a run is in progress, preventing a mid-run switch', async () => {
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    await userEvent.click(screen.getByRole('button', { name: /run/i }))

    const otherSuiteButton = screen.getByRole('button', { name: /intent \+ filters/i })
    expect(otherSuiteButton).toBeDisabled()

    await userEvent.click(otherSuiteButton)

    // Click had no effect: still only one WS connection, and it was opened
    // for the originally-selected suite, not the one just (attemptedly) clicked.
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(JSON.parse(FakeWebSocket.instances[0].sent[0]).suite).toBe('slm_intent')
  })

  it('treats a limit of "0" as no limit rather than sending limit: 0', async () => {
    render(<SuiteRunner wsUrl="ws://x/ws/admin-eval" apiBaseUrl="http://x" token="tok" onUnauthorized={vi.fn()} />)

    const limitInput = screen.getByPlaceholderText(/limit/i)
    await userEvent.type(limitInput, '0')
    await userEvent.click(screen.getByRole('button', { name: /run/i }))

    const socket = FakeWebSocket.instances[0]
    const sentMessage = JSON.parse(socket.sent[0])
    expect(sentMessage.limit).toBeUndefined()
  })
})
