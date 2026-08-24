import { describe, expect, it, beforeEach } from 'vitest'
import { getStoredAdminToken, setStoredAdminToken, clearStoredAdminToken } from './adminAuth'

describe('adminAuth', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('returns null when nothing stored', () => {
    expect(getStoredAdminToken()).toBeNull()
  })

  it('stores and retrieves a token', () => {
    setStoredAdminToken('secret-123')
    expect(getStoredAdminToken()).toBe('secret-123')
  })

  it('clears a stored token', () => {
    setStoredAdminToken('secret-123')
    clearStoredAdminToken()
    expect(getStoredAdminToken()).toBeNull()
  })
})
