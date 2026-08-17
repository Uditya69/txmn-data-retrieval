import { describe, expect, it, beforeEach } from 'vitest'
import { getStoredToken, getStoredEmail, setStoredSession, clearStoredSession } from './auth'

beforeEach(() => {
  localStorage.clear()
})

describe('auth storage', () => {
  it('returns null for token and email when nothing is stored', () => {
    expect(getStoredToken()).toBeNull()
    expect(getStoredEmail()).toBeNull()
  })

  it('stores and retrieves a session', () => {
    setStoredSession('tok-123', 'alice@example.com')
    expect(getStoredToken()).toBe('tok-123')
    expect(getStoredEmail()).toBe('alice@example.com')
  })

  it('clears a stored session', () => {
    setStoredSession('tok-123', 'alice@example.com')
    clearStoredSession()
    expect(getStoredToken()).toBeNull()
    expect(getStoredEmail()).toBeNull()
  })
})
