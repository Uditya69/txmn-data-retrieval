const ADMIN_TOKEN_KEY = 'taxmann-admin-token'

// sessionStorage (not localStorage) deliberately - the admin token shouldn't
// outlive the browser tab; each new admin session re-enters it.
export function getStoredAdminToken(): string | null {
  return sessionStorage.getItem(ADMIN_TOKEN_KEY)
}

export function setStoredAdminToken(token: string): void {
  sessionStorage.setItem(ADMIN_TOKEN_KEY, token)
}

export function clearStoredAdminToken(): void {
  sessionStorage.removeItem(ADMIN_TOKEN_KEY)
}
