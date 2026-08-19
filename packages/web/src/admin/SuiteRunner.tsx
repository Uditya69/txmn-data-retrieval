import { useEffect, useState } from 'react'
import { useAdminEvalRun } from './useAdminEvalRun'

const SUITES: { id: string; name: string }[] = [
  { id: 'slm_intent', name: 'SLM Intent, Filters & Rewrite' },
  { id: 'collection_routing', name: 'Collection Routing' },
  { id: 'retrieval', name: 'Retrieval Pipeline' },
]

interface SuiteRunnerProps {
  wsUrl: string
  apiBaseUrl: string
  token: string
  onUnauthorized: () => void
}

export default function SuiteRunner({ wsUrl, apiBaseUrl, token, onUnauthorized }: SuiteRunnerProps) {
  const evalRun = useAdminEvalRun(wsUrl)
  const [selected, setSelected] = useState(SUITES[0].id)
  const [limit, setLimit] = useState('')

  useEffect(() => {
    if (evalRun.error === 'unauthorized') onUnauthorized()
  }, [evalRun.error, onUnauthorized])

  // Shows the last completed run for the newly-selected suite immediately,
  // before/without starting a new WS run - survives a page refresh and makes
  // switching suites not look like a blank slate if one already ran.
  useEffect(() => {
    evalRun.loadCached(apiBaseUrl, selected, token)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, apiBaseUrl, token])

  return (
    <div className="max-w-4xl mx-auto py-8 flex flex-col gap-4">
      <div className="flex gap-2 flex-wrap">
        {SUITES.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setSelected(s.id)}
            disabled={evalRun.running}
            className="px-3 py-1.5 rounded border text-sm disabled:opacity-50"
            style={{ fontWeight: selected === s.id ? 600 : 400 }}
          >
            {s.name}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <input
          type="number"
          min={1}
          value={limit}
          onChange={(e) => setLimit(e.target.value)}
          placeholder="limit (optional)"
          className="border rounded px-2 py-1 w-40 text-sm"
        />
        <button
          type="button"
          onClick={() => evalRun.run(selected, token, limit && Number(limit) > 0 ? Number(limit) : undefined)}
          disabled={evalRun.running}
          className="border rounded px-3 py-1.5 text-sm font-medium"
        >
          {evalRun.running ? 'Running…' : 'Run'}
        </button>
        {evalRun.error && evalRun.error !== 'unauthorized' && (
          <span className="text-sm" style={{ color: 'crimson' }}>{evalRun.error}</span>
        )}
      </div>

      {evalRun.total > 0 && (
        <div className="flex flex-col gap-1">
          <div className="h-2 rounded bg-gray-200 overflow-hidden">
            <div className="h-full bg-green-600" style={{ width: `${evalRun.percent}%` }} />
          </div>
          <p className="text-xs text-gray-600">
            {evalRun.percent}% · {evalRun.passed}/{evalRun.cases.length} passed of {evalRun.total}
          </p>
        </div>
      )}

      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left border-b">
            <th className="py-1 pr-2">ID</th>
            <th className="py-1 pr-2">Query</th>
            <th className="py-1 pr-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {evalRun.cases.map((c) => (
            <tr key={c.id} className="border-b align-top">
              <td className="py-1 pr-2 font-mono">{c.id}</td>
              <td className="py-1 pr-2">{c.query}</td>
              <td className="py-1 pr-2">
                <span style={{ color: c.status === 'pass' ? 'green' : 'crimson' }}>{c.status}</span>
                <details className="mt-1">
                  <summary className="cursor-pointer text-xs text-gray-500">detail</summary>
                  <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(c.detail, null, 2)}</pre>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
