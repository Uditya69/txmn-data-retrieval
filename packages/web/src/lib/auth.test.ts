import { describe, expect, it, beforeEach } from 'vitest'
import { getStoredToken, getStoredRefreshToken, getStoredEmail, setStoredSession, clearStoredSession, decodeTokenExpiry } from './auth'

beforeEach(() => {
  localStorage.clear()
})

describe('auth storage', () => {
  it('returns null for token, refresh token, and email when nothing is stored', () => {
    expect(getStoredToken()).toBeNull()
    expect(getStoredRefreshToken()).toBeNull()
    expect(getStoredEmail()).toBeNull()
  })

  it('stores and retrieves a session', () => {
    setStoredSession('tok-123', 'refresh-456', 'alice@example.com')
    expect(getStoredToken()).toBe('tok-123')
    expect(getStoredRefreshToken()).toBe('refresh-456')
    expect(getStoredEmail()).toBe('alice@example.com')
  })

  it('clears a stored session', () => {
    setStoredSession('tok-123', 'refresh-456', 'alice@example.com')
    clearStoredSession()
    expect(getStoredToken()).toBeNull()
    expect(getStoredRefreshToken()).toBeNull()
    expect(getStoredEmail()).toBeNull()
  })
})

describe('decodeTokenExpiry', () => {
  function makeJwt(payload: object): string {
    const base64url = (obj: object) => btoa(JSON.stringify(obj)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
    return `${base64url({ alg: 'HS256', typ: 'JWT' })}.${base64url(payload)}.fake-signature`
  }

  it('reads the exp claim from a well-formed JWT', () => {
    const token = makeJwt({ user_id: 'user-123', exp: 1234567890 })
    expect(decodeTokenExpiry(token)).toBe(1234567890)
  })

  it('returns null for a token with no exp claim', () => {
    const token = makeJwt({ user_id: 'user-123' })
    expect(decodeTokenExpiry(token)).toBeNull()
  })

  it('returns null for garbage input instead of throwing', () => {
    expect(decodeTokenExpiry('not-a-jwt-at-all')).toBeNull()
    expect(decodeTokenExpiry('')).toBeNull()
  })
})
