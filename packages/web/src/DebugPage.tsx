// src/DebugPage.tsx
import { useState } from 'react'
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import TracePanel from './components/TracePanel'
import DocumentModal from './components/DocumentModal'
import { useSearch } from './api/useSearch'
import { resolveWsUrl, resolveApiBaseUrl } from './lib/config'
import styles from './App.module.css'

export default function DebugPage() {
  const wsUrl = resolveWsUrl()
  const { traceSteps, loading, wsError, search } = useSearch(wsUrl)
  const [openDocId, setOpenDocId] = useState<string | null>(null)

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Retrieval Debug — raw ES / Milvus dense / Milvus sparse</h1>
        <Link to="/">Back to search</Link>
      </header>
      <SearchBar onSearch={(query) => search(query, true, 'instant')} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      <TracePanel steps={traceSteps} onOpenDocument={setOpenDocId} />
      <DocumentModal
        docId={openDocId}
        apiBaseUrl={resolveApiBaseUrl(wsUrl)}
        onClose={() => setOpenDocId(null)}
        onNavigate={setOpenDocId}
      />
    </div>
  )
}
