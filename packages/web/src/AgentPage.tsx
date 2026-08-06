// src/AgentPage.tsx
import { useState } from 'react'
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import TracePanel from './components/TracePanel'
import DevModeToggle from './components/DevModeToggle'
import { useAgentSearch } from './api/useAgentSearch'
import { resolveAgentWsUrl } from './lib/config'
import styles from './App.module.css'

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function AgentPage() {
  const wsUrl = resolveAgentWsUrl()
  const { traceSteps, loading, result, wsError, search } = useAgentSearch(wsUrl)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)

  const showTrace = devMode && traceSteps.length > 0

  const mainContent = (
    <div>
      {loading && <p>Searching…</p>}
      {result && result.ok && (
        <section>
          <p>{result.answer}</p>
          <p data-testid="cited-doc-ids">Cited: {result.docIds.join(', ')}</p>
        </section>
      )}
      {result && !result.ok && <p className={styles.wsError}>{result.error}</p>}
    </div>
  )

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Agentic Search — tool-calling agent, cited answers only</h1>
        <div className={styles.headerActions}>
          <Link to="/">Back to search</Link>
          <DevModeToggle devMode={devMode} onToggle={setDevMode} />
        </div>
      </header>
      <SearchBar onSearch={(query) => search(query)} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      {showTrace ? (
        <div className={styles.splitLayout}>
          {mainContent}
          <aside className={styles.tracePane}>
            <h2>Agent trace</h2>
            <TracePanel steps={traceSteps} />
          </aside>
        </div>
      ) : (
        mainContent
      )}
    </div>
  )
}
