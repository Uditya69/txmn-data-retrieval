// src/lib/auth.ts
const TOKEN_KEY = 'taxmann-retrieval-access-token'
const REFRESH_TOKEN_KEY = 'taxmann-retrieval-refresh-token'
const EMAIL_KEY = 'taxmann-retrieval-account-email'

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY)
}

export function getStoredEmail(): string | null {
  return localStorage.getItem(EMAIL_KEY)
}

export function setStoredSession(token: string, refreshToken: string, email: string) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
  localStorage.setItem(EMAIL_KEY, email)
}

export function clearStoredSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
  localStorage.removeItem(EMAIL_KEY)
}

/**
 * Reads a JWT's `exp` claim (seconds since epoch) without verifying its
 * signature - the payload segment of a JWT is just base64url-encoded JSON,
 * readable by anyone, verification or not. This is only ever used to schedule
 * a proactive silent refresh a little before the access token actually
 * expires; it is never trusted for anything security-relevant - the server
 * independently verifies the token's signature on every real request
 * regardless of what this function returns. Returns null if the token isn't
 * shaped like a JWT or has no exp claim.
 */
export function decodeTokenExpiry(token: string): number | null {
  try {
    const payloadSegment = token.split('.')[1]
    const base64 = payloadSegment.replace(/-/g, '+').replace(/_/g, '/')
    const payload = JSON.parse(atob(base64))
    return typeof payload.exp === 'number' ? payload.exp : null
  } catch {
    return null
  }
}
