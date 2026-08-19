import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import AdminApp from './admin/AdminApp'
import HowItWorksApp from './how-it-works/HowItWorksApp'
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
const isHowItWorksRoute = window.location.pathname.startsWith('/how-it-works')

function Root() {
  if (isAdminRoute) return <AdminApp />
  if (isHowItWorksRoute) return <HowItWorksApp />
  return <App />
}

createRoot(container).render(
  <StrictMode>
    <ErrorBoundary>
      <Root />
    </ErrorBoundary>
  </StrictMode>,
)
