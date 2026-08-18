// packages/web/src/admin/AdminApp.test.tsx
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AdminApp from './AdminApp'
import { getStoredAdminToken } from '../lib/adminAuth'

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
  sessionStorage.clear()
  FakeWebSocket.instances = []
  // @ts-expect-error test stub
  global.WebSocket = FakeWebSocket
  // SuiteRunner (rendered once logged in) always calls loadCached() on mount -
  // stub a no-prior-run response so that doesn't need its own setup per test.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => null }))
})

describe('AdminApp', () => {
  it('shows the login form when no token is stored', () => {
    render(<AdminApp />)
    expect(screen.getByPlaceholderText('Admin token')).toBeInTheDocument()
  })

  it('stores the token and shows the runner after login', async () => {
    render(<AdminApp />)
    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'my-secret')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    expect(getStoredAdminToken()).toBe('my-secret')
    expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
  })

  it('returns to the login form and clears the token when the server reports unauthorized', async () => {
    render(<AdminApp />)
    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'stale-token')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    await userEvent.click(screen.getByRole('button', { name: /run/i }))
    const socket = FakeWebSocket.instances[0]
    socket.onmessage?.({ data: JSON.stringify({ type: 'error', reason: 'unauthorized' }) })

    expect(await screen.findByPlaceholderText('Admin token')).toBeInTheDocument()
    expect(getStoredAdminToken()).toBeNull()
  })
})
