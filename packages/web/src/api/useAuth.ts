// src/api/useAuth.ts
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getStoredEmail,
  getStoredRefreshToken,
  getStoredToken,
  setStoredSession,
  clearStoredSession,
  decodeTokenExpiry,
} from '../lib/auth'

export interface AuthState {
  token: string | null
  email: string | null
  loading: boolean
  error: string | null
}

export interface AuthActions {
  signup: (email: string, password: string) => Promise<boolean>
  login: (email: string, password: string) => Promise<boolean>
  logout: () => void
  /** Silently exchanges the stored refresh token for a new access/refresh pair.
   * Used both proactively (scheduled shortly before the access token's own
   * expiry) and reactively (ws.py's `session_expired` signal - see useSearch's
   * onSessionExpired). Falls back to a full logout() if the refresh token
   * itself is also dead (expired, already rotated away, or never existed) -
   * that's the only case where the user actually gets signed out. */
  refresh: () => Promise<void>
}

// Refresh this many seconds before the access token's own expiry - not at the
// exact expiry instant, so a request that starts just before expiry doesn't
// race a token that goes stale mid-flight.
const REFRESH_SKEW_SECONDS = 60

function errorMessageForStatus(status: number): string {
  if (status === 409) return 'An account with that email already exists.'
  if (status === 401) return 'Incorrect email or password.'
  return 'Something went wrong. Please try again.'
}

export function useAuth(apiBaseUrl: string): AuthState & AuthActions {
  const [token, setToken] = useState<string | null>(getStoredToken)
  const [email, setEmail] = useState<string | null>(getStoredEmail)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearScheduledRefresh = useCallback(() => {
    if (refreshTimerRef.current !== null) {
      clearTimeout(refreshTimerRef.current)
      refreshTimerRef.current = null
    }
  }, [])

  const logout = useCallback(() => {
    clearScheduledRefresh()
    // Best-effort server-side revocation - fire and forget, never blocks the
    // client-side logout on network availability. Reads the token directly
    // from storage rather than the `refreshToken` state, since logout() can
    // be called from a stale closure (e.g. refresh()'s own failure path).
    const storedRefreshToken = getStoredRefreshToken()
    if (storedRefreshToken) {
      fetch(`${apiBaseUrl}/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: storedRefreshToken }),
      }).catch(() => {})
    }
    clearStoredSession()
    setToken(null)
    setEmail(null)
  }, [apiBaseUrl, clearScheduledRefresh])

  const refresh = useCallback(async () => {
    const storedRefreshToken = getStoredRefreshToken()
    const storedEmail = getStoredEmail()
    if (!storedRefreshToken || !storedEmail) {
      logout()
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: storedRefreshToken }),
      })
      if (!response.ok) {
        logout()
        return
      }
      const data = (await response.json()) as { access_token: string; refresh_token: string }
      setStoredSession(data.access_token, data.refresh_token, storedEmail)
      setToken(data.access_token)
    } catch {
      // Network failure mid-refresh: leave the current (soon-to-expire) session
      // in place rather than logging out - a transient network blip shouldn't
      // sign the user out. The next scheduled refresh, or the reactive
      // session_expired path, gets another chance.
    }
  }, [apiBaseUrl, logout])

  // (Re)schedule a proactive refresh whenever the access token changes -
  // covers the initial page load (token restored from localStorage) and every
  // subsequent login/refresh. Cleared on unmount and before each reschedule.
  useEffect(() => {
    clearScheduledRefresh()
    if (!token) return
    const exp = decodeTokenExpiry(token)
    if (exp === null) return
    const msUntilRefresh = (exp - REFRESH_SKEW_SECONDS) * 1000 - Date.now()
    // Refresh immediately (still async, still off the render path) rather than
    // scheduling a negative/zero delay if the stored token is already past its
    // skew window (e.g. the page was left closed for a while).
    refreshTimerRef.current = setTimeout(() => {
      refresh()
    }, Math.max(msUntilRefresh, 0))
    return clearScheduledRefresh
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  const submit = useCallback(
    async (path: 'signup' | 'login', submittedEmail: string, password: string): Promise<boolean> => {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch(`${apiBaseUrl}/auth/${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: submittedEmail, password }),
        })
        if (!response.ok) {
          setError(errorMessageForStatus(response.status))
          return false
        }
        const data = (await response.json()) as { access_token: string; refresh_token: string }
        setStoredSession(data.access_token, data.refresh_token, submittedEmail)
        setToken(data.access_token)
        setEmail(submittedEmail)
        return true
      } catch {
        setError('Could not reach the server. Please try again.')
        return false
      } finally {
        setLoading(false)
      }
    },
    [apiBaseUrl],
  )

  const signup = useCallback((submittedEmail: string, password: string) => submit('signup', submittedEmail, password), [submit])
  const login = useCallback((submittedEmail: string, password: string) => submit('login', submittedEmail, password), [submit])

  return { token, email, loading, error, signup, login, logout, refresh }
}
