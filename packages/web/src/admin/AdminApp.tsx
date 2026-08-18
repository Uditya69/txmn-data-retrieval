// packages/web/src/admin/AdminApp.tsx
import { useState } from 'react'
import AdminLogin from './AdminLogin'
import SuiteRunner from './SuiteRunner'
import { getStoredAdminToken, setStoredAdminToken, clearStoredAdminToken } from '../lib/adminAuth'
import { resolveWsUrl, resolveApiBaseUrl, resolveAdminWsUrl } from '../lib/config'

export default function AdminApp() {
  const [token, setToken] = useState<string | null>(getStoredAdminToken)
  const [loginError, setLoginError] = useState<string | null>(null)

  const apiBaseUrl = resolveApiBaseUrl(resolveWsUrl())
  const adminWsUrl = resolveAdminWsUrl(apiBaseUrl)

  function handleLogin(candidate: string) {
    setStoredAdminToken(candidate)
    setToken(candidate)
    setLoginError(null)
  }

  function handleUnauthorized() {
    clearStoredAdminToken()
    setToken(null)
    setLoginError('Invalid token.')
  }

  if (!token) {
    return <AdminLogin onSubmit={handleLogin} error={loginError} />
  }

  return <SuiteRunner wsUrl={adminWsUrl} apiBaseUrl={apiBaseUrl} token={token} onUnauthorized={handleUnauthorized} />
}
