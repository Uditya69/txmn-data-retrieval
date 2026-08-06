// src/AgentPage.tsx
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import TracePanel from './components/TracePanel'
import { useAgentSearch } from './api/useAgentSearch'
import { resolveAgentWsUrl } from './lib/config'
import styles from './App.module.css'

export default function AgentPage() {
  const wsUrl = resolveAgentWsUrl()
  const { traceSteps, loading, result, wsError, search } = useAgentSearch(wsUrl)

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Agentic Search — tool-calling agent, cited answers only</h1>
        <Link to="/">Back to search</Link>
      </header>
      <SearchBar onSearch={(query) => search(query)} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      {result && result.ok && (
        <section>
          <p>{result.answer}</p>
        </section>
      )}
      {result && !result.ok && <p className={styles.wsError}>{result.error}</p>}
      <TracePanel steps={traceSteps} />
    </div>
  )
}
