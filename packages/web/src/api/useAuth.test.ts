import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useAuth } from './useAuth'

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
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

describe('useAuth', () => {
  it('starts with no session when localStorage is empty', () => {
    const { result } = renderHook(() => useAuth('http://api.test'))
    expect(result.current.token).toBeNull()
    expect(result.current.email).toBeNull()
  })

  it('signup stores the session on success and posts to /auth/signup', async () => {
    mockFetchOnce(200, { access_token: 'tok-abc', token_type: 'bearer' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    let ok = false
    await act(async () => {
      ok = await result.current.signup('alice@example.com', 'hunter2')
    })

    expect(ok).toBe(true)
    expect(result.current.token).toBe('tok-abc')
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

  it('logout clears the stored session', async () => {
    mockFetchOnce(200, { access_token: 'tok-abc', token_type: 'bearer' })
    const { result } = renderHook(() => useAuth('http://api.test'))

    await act(async () => {
      await result.current.login('alice@example.com', 'hunter2')
    })
    expect(result.current.token).toBe('tok-abc')

    act(() => {
      result.current.logout()
    })

    expect(result.current.token).toBeNull()
    expect(result.current.email).toBeNull()
  })
})
