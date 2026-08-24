import { useCallback, useRef, useState } from 'react'

export interface CaseEvent {
  type: 'case'
  id: string
  query: string
  status: 'pass' | 'fail' | 'error'
  detail: Record<string, unknown>
}

export interface AdminEvalState {
  running: boolean
  percent: number
  total: number
  passed: number
  cases: CaseEvent[]
  error: string | null
}

const INITIAL_STATE: AdminEvalState = { running: false, percent: 0, total: 0, passed: 0, cases: [], error: null }

export function useAdminEvalRun(
  wsUrl: string,
): AdminEvalState & {
  run: (suite: string, token: string, limit?: number) => void
  loadCached: (apiBaseUrl: string, suite: string, token: string) => Promise<void>
} {
  const [state, setState] = useState<AdminEvalState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  // Hydrates from GET /admin/api/eval-runs/{suite} (the spec's cache-read
  // endpoint) - lets a completed run survive a page refresh or a suite switch
  // without starting a new WS run. Deliberately does not touch `running`.
  const loadCached = useCallback(async (apiBaseUrl: string, suite: string, token: string) => {
    const response = await fetch(`${apiBaseUrl}/admin/api/eval-runs/${suite}`, {
      headers: { 'X-Admin-Token': token },
    })
    if (!response.ok) return
    const cached = await response.json()
    if (!cached) return
    setState((prev) => ({
      ...prev,
      cases: cached.cases,
      total: cached.summary.total,
      passed: cached.summary.passed,
      percent: 100,
    }))
  }, [])

  const run = useCallback(
    (suite: string, token: string, limit?: number) => {
      socketRef.current?.close()
      setState({ ...INITIAL_STATE, running: true })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, running: false, error: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ suite, token, limit }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'case') {
          setState((prev) => ({
            ...prev,
            cases: [...prev.cases, message as CaseEvent],
            passed: prev.passed + (message.status === 'pass' ? 1 : 0),
          }))
        } else if (message.type === 'progress') {
          setState((prev) => ({ ...prev, percent: message.percent, total: message.total }))
        } else if (message.type === 'done') {
          setState((prev) => ({ ...prev, running: false }))
        } else if (message.type === 'error') {
          setState((prev) => ({ ...prev, running: false, error: message.reason }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, running: false, error: 'Connection failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.running ? { ...prev, running: false, error: 'Run interrupted.' } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, run, loadCached }
}
