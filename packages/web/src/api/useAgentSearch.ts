import { useCallback, useRef, useState } from 'react'
import type { TraceStep } from './useSearch'

export type AgentResult =
  | { ok: true; answer: string; docIds: string[] }
  | { ok: false; error: string }

export interface AgentSearchState {
  loading: boolean
  traceSteps: TraceStep[]
  result: AgentResult | null
  wsError: string | null
}

const INITIAL_STATE: AgentSearchState = { loading: false, traceSteps: [], result: null, wsError: null }

export function useAgentSearch(wsUrl: string): AgentSearchState & { search: (query: string) => void } {
  const [state, setState] = useState<AgentSearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (query: string) => {
      socketRef.current?.close()
      setState({ loading: true, traceSteps: [], result: null, wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ query }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'ai_mode_trace') {
          setState((prev) => ({ ...prev, traceSteps: [...prev.traceSteps, { step: message.step, data: message.data }] }))
        } else if (message.type === 'agent_done') {
          setState((prev) => ({ ...prev, loading: false, result: { ok: true, answer: message.answer, docIds: message.doc_ids ?? [] } }))
        } else if (message.type === 'agent_unverifiable') {
          setState((prev) => ({
            ...prev,
            loading: false,
            result: {
              ok: false,
              error: `Could not produce a fully cited answer. Invalid doc_id(s): ${(message.invalid_doc_ids ?? []).join(', ')}`,
            },
          }))
        } else if (message.type === 'agent_error') {
          setState((prev) => ({ ...prev, loading: false, result: { ok: false, error: message.error } }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the agent service failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.loading ? { ...prev, loading: false } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, search }
}
