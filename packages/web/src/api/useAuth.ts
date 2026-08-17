// src/api/useAuth.ts
import { useCallback, useState } from 'react'
import { getStoredEmail, getStoredToken, setStoredSession, clearStoredSession } from '../lib/auth'

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
}

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
        const data = (await response.json()) as { access_token: string }
        setStoredSession(data.access_token, submittedEmail)
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

  const logout = useCallback(() => {
    clearStoredSession()
    setToken(null)
    setEmail(null)
  }, [])

  return { token, email, loading, error, signup, login, logout }
}
