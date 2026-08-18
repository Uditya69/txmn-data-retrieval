import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import AdminApp from './admin/AdminApp'
import { ErrorBoundary } from './components/ErrorBoundary'
import './index.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('Root container #root not found')
}

// No router dependency in this app (see App.tsx's own URLSearchParams-based
// dev-mode flag for precedent) - a plain pathname branch is enough for one
// extra page.
const isAdminRoute = window.location.pathname.startsWith('/admin')

createRoot(container).render(
  <StrictMode>
    <ErrorBoundary>
      {isAdminRoute ? <AdminApp /> : <App />}
    </ErrorBoundary>
  </StrictMode>,
)
