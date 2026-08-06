// src/AgentPage.tsx
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import SearchBar from './components/SearchBar'
import TracePanel from './components/TracePanel'
import DevModeToggle from './components/DevModeToggle'
import DocumentModal from './components/DocumentModal'
import { useAgentSearch } from './api/useAgentSearch'
import { resolveAgentWsUrl, resolveApiBaseUrl } from './lib/config'
import { parseCitations } from './lib/citations'
import { groupIntoParagraphs, renderInlineText } from './lib/richText'
import styles from './App.module.css'
import overviewStyles from './components/OverviewCard.module.css'

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function AgentPage() {
  const wsUrl = resolveAgentWsUrl()
  const { traceSteps, loading, result, wsError, search } = useAgentSearch(wsUrl)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [openDocId, setOpenDocId] = useState<string | null>(null)

  const showTrace = devMode && traceSteps.length > 0

  const parsed = useMemo(() => {
    if (!result || !result.ok) return null
    return parseCitations(result.answer)
  }, [result])
  const paragraphs = useMemo(() => (parsed ? groupIntoParagraphs(parsed.segments) : []), [parsed])

  const mainContent = (
    <div>
      {loading && <p>Searching…</p>}
      {result && result.ok && (
        <section className={overviewStyles.card}>
          {paragraphs.map((paragraph, pIndex) => (
            <p key={pIndex} className={overviewStyles.answer}>
              {paragraph.map((segment, index) =>
                segment.type === 'text' ? (
                  renderInlineText(segment.text, `${pIndex}-${index}`)
                ) : (
                  <span key={index} className={overviewStyles.citationGroup}>
                    {segment.numbers.map((n, i) => (
                      <sup key={i} className={overviewStyles.pill}>
                        {n}
                      </sup>
                    ))}
                  </span>
                ),
              )}
            </p>
          ))}
          {parsed && parsed.citations.length > 0 && (
            <div className={overviewStyles.chipRow}>
              {parsed.citations.map((citation) => (
                <button
                  key={citation.doc_id}
                  type="button"
                  className={overviewStyles.chip}
                  onClick={() => setOpenDocId(citation.doc_id)}
                >
                  {citation.number}. {citation.doc_id} ({citation.count})
                </button>
              ))}
            </div>
          )}
          <p data-testid="cited-doc-ids">
            Verified citations:{' '}
            {result.docIds.map((docId, index) => (
              <span key={docId}>
                {index > 0 && ', '}
                <button type="button" className={overviewStyles.chip} onClick={() => setOpenDocId(docId)}>
                  {docId}
                </button>
              </span>
            ))}
          </p>
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
