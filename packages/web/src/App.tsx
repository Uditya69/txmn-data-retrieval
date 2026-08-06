// src/App.tsx
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import OverviewCard from './components/OverviewCard'
import DocumentsFeed from './components/DocumentsFeed'
import DevModeToggle from './components/DevModeToggle'
import DocumentModal from './components/DocumentModal'
import TracePanel from './components/TracePanel'
import { useSearch } from './api/useSearch'
import { resolveWsUrl, resolveApiBaseUrl } from './lib/config'
import styles from './App.module.css'

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function App() {
  const wsUrl = resolveWsUrl()
  const { instant, aiMode, traceSteps, loading, wsError, search } = useSearch(wsUrl)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [highlightedDocId, setHighlightedDocId] = useState<string | null>(null)
  const [openDocId, setOpenDocId] = useState<string | null>(null)

  useEffect(() => {
    if (highlightedDocId === null) return
    const timeout = window.setTimeout(() => setHighlightedDocId(null), 2000)
    return () => window.clearTimeout(timeout)
  }, [highlightedDocId])

  function handleCitationClick(docId: string) {
    setHighlightedDocId(docId)
    document.getElementById(`document-${docId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setOpenDocId(docId)
  }

  const showTrace = devMode && traceSteps.length > 0

  const mainContent = (
    <div>
      <OverviewCard aiMode={aiMode} loading={loading} onCitationClick={handleCitationClick} />
      <DocumentsFeed
        instant={instant}
        aiMode={aiMode}
        devMode={devMode}
        highlightedDocId={highlightedDocId}
        onOpenDocument={setOpenDocId}
      />
    </div>
  )

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Taxmann Retrieval</h1>
        <div className={styles.headerActions}>
          <Link to="/debug">Retrieval debug</Link>
          <Link to="/agent">Agentic search</Link>
          <DevModeToggle devMode={devMode} onToggle={setDevMode} />
        </div>
      </header>
      <SearchBar onSearch={(query) => search(query, devMode)} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      {showTrace ? (
        <div className={styles.splitLayout}>
          {mainContent}
          <aside className={styles.tracePane}>
            <h2>AI Mode trace</h2>
            <TracePanel steps={traceSteps} onOpenDocument={setOpenDocId} />
          </aside>
        </div>
      ) : (
        mainContent
      )}
      <DocumentModal
        docId={openDocId}
        apiBaseUrl={resolveApiBaseUrl(wsUrl)}
        onClose={() => setOpenDocId(null)}
        onNavigate={setOpenDocId}
      />
    </div>
  )
}
