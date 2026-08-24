import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useAuth } from './useAuth'
import { setStoredSession } from '../lib/auth'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function mockFetchOnce(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    }),
  )
}

// exp far in the future so the auto-refresh scheduling effect (real timers,
// not faked in most of these tests) never actually fires mid-test.
function farFutureJwt(): string {
  const base64url = (obj: object) => btoa(JSON.stringify(obj)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  return `${base64url({ alg: 'HS256' })}.${base64url({ user_id: 'user-123', exp: Math.floor(Date.now() / 1000) + 3600 })}.sig`
}

describe('useAuth', () => {
  it('starts with no session when localStorage is empty', () => {
    const { result } = renderHook(() => useAuth('http://api.test'))
    expect(result.current.token).toBeNull()
    expect(result.current.email).toBeNull()
  })

  it('signup stores the session on success and posts to /auth/signup', async () => {
    const token = farFutureJwt()
    mockFetchOnce(200, { access_token: token, refresh_token: 'refresh-abc', token_type: 'bearer' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    let ok = false
    await act(async () => {
      ok = await result.current.signup('alice@example.com', 'hunter2')
    })

    expect(ok).toBe(true)
    expect(result.current.token).toBe(token)
    expect(result.current.email).toBe('alice@example.com')
    expect(fetch).toHaveBeenCalledWith(
      'http://api.test/auth/signup',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ email: 'alice@example.com', password: 'hunter2' }),
      }),
    )
  })

  it('login sets a friendly error message on 401 and does not store a session', async () => {
    mockFetchOnce(401, { detail: 'invalid email or password' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    let ok = true
    await act(async () => {
      ok = await result.current.login('alice@example.com', 'wrong')
    })

    expect(ok).toBe(false)
    expect(result.current.token).toBeNull()
    await waitFor(() => expect(result.current.error).toMatch(/incorrect email or password/i))
  })

  it('signup sets a friendly error message on 409 (duplicate email)', async () => {
    mockFetchOnce(409, { detail: 'email already registered' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.signup('alice@example.com', 'hunter2')
    })

    expect(result.current.error).toMatch(/already exists/i)
  })

  it('logout clears the stored session and posts the refresh token to /auth/logout', async () => {
    const token = farFutureJwt()
    mockFetchOnce(200, { access_token: token, refresh_token: 'refresh-abc', token_type: 'bearer' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.login('alice@example.com', 'hunter2')
    })
    expect(result.current.token).toBe(token)

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 204, json: async () => ({}) }))

    act(() => {
      result.current.logout()
    })

    expect(result.current.token).toBeNull()
    expect(result.current.email).toBeNull()
    expect(fetch).toHaveBeenCalledWith(
      'http://api.test/auth/logout',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ refresh_token: 'refresh-abc' }) }),
    )
  })

  it('logout does not call /auth/logout when there is no stored refresh token', () => {
    vi.stubGlobal('fetch', vi.fn())
    const { result } = renderHook(() => useAuth('http://api.test'))

    act(() => {
      result.current.logout()
    })

    expect(fetch).not.toHaveBeenCalled()
  })

  it('refresh exchanges the stored refresh token for a new pair', async () => {
    setStoredSession(farFutureJwt(), 'refresh-old', 'alice@example.com')
    const newToken = farFutureJwt()
    mockFetchOnce(200, { access_token: newToken, refresh_token: 'refresh-new', token_type: 'bearer' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.refresh()
    })

    expect(result.current.token).toBe(newToken)
    expect(fetch).toHaveBeenCalledWith(
      'http://api.test/auth/refresh',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ refresh_token: 'refresh-old' }) }),
    )
  })

  it('refresh logs out when the refresh token itself is rejected', async () => {
    setStoredSession(farFutureJwt(), 'refresh-old', 'alice@example.com')
    mockFetchOnce(401, { detail: 'refresh token invalid or expired' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.refresh()
    })

    expect(result.current.token).toBeNull()
    expect(result.current.email).toBeNull()
  })

  it('refresh is a no-op that logs out when there is no stored session at all', async () => {
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.refresh()
    })

    expect(result.current.token).toBeNull()
  })
})
