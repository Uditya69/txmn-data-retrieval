// src/lib/auth.ts
const TOKEN_KEY = 'taxmann-retrieval-access-token'
const EMAIL_KEY = 'taxmann-retrieval-account-email'

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function getStoredEmail(): string | null {
  return localStorage.getItem(EMAIL_KEY)
}

export function setStoredSession(token: string, email: string) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(EMAIL_KEY, email)
}

export function clearStoredSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(EMAIL_KEY)
}
