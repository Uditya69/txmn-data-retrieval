import { useCallback, useRef, useState } from 'react'
import type { EsHit, MilvusByCollection, RerankedHit } from '../lib/mergeResults'

export type AiModeCitation = Record<string, unknown>

export interface InstantResult {
  es: EsHit[] | null
  es_error: string | null
  milvus: MilvusByCollection | null
  milvus_sparse: MilvusByCollection | null
  milvus_error: string | null
  reranked?: RerankedHit[] | null
  reranked_error?: string | null
}

export type AiModeResult =
  | { ok: true; answer: string; citations: Record<string, AiModeCitation>; reasoning?: string | null }
  | { ok: false; error: string }

export interface TraceStep {
  step: string
  data: Record<string, unknown>
}

export interface SearchState {
  /** true from search() until ai_mode_done/ai_mode_error/an error/close arrives - tracks AI Mode specifically, not the Documents feed (that only depends on `instant`). */
  loading: boolean
  instant: InstantResult | null
  aiMode: AiModeResult | null
  traceSteps: TraceStep[]
  wsError: string | null
}

const INITIAL_STATE: SearchState = { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null }

export type SearchMode = 'instant' | 'ai_mode' | 'both'

export function useSearch(
  wsUrl: string,
  accessToken?: string | null,
  onSessionExpired?: () => void,
): SearchState & { search: (query: string, trace: boolean, mode?: SearchMode, rerank?: boolean) => void } {
  const [state, setState] = useState<SearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (query: string, trace: boolean, mode: SearchMode = 'both', rerank: boolean = false) => {
      socketRef.current?.close()
      setState({ loading: true, instant: null, aiMode: null, traceSteps: [], wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        // access_token is only included when a user is signed in - the backend
        // treats it as fully optional (see ws.py's _resolve_user_id) and this
        // keeps guest requests byte-identical to before persona existed.
        const payload: Record<string, unknown> = { query, mode, trace, rerank }
        if (accessToken) payload.access_token = accessToken
        socket.send(JSON.stringify(payload))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'instant_result') {
          setState((prev) => ({
            ...prev,
            instant: {
              es: message.es ?? null,
              es_error: message.es_error ?? null,
              milvus: message.milvus ?? null,
              milvus_sparse: message.milvus_sparse ?? null,
              milvus_error: message.milvus_error ?? null,
              reranked: message.reranked ?? null,
              reranked_error: message.reranked_error ?? null,
            },
          }))
        } else if (message.type === 'ai_mode_trace') {
          setState((prev) => ({
            ...prev,
            traceSteps: [...prev.traceSteps, { step: message.step, data: message.data }],
          }))
        } else if (message.type === 'ai_mode_done') {
          setState((prev) => ({
            ...prev,
            loading: false,
            aiMode: { ok: true, answer: message.answer, citations: message.citations ?? {}, reasoning: message.reasoning ?? null },
          }))
        } else if (message.type === 'ai_mode_error') {
          setState((prev) => ({ ...prev, loading: false, aiMode: { ok: false, error: message.error } }))
        } else if (message.type === 'session_expired') {
          // The access_token we sent didn't decode server-side (most commonly expired -
          // see ws.py's _resolve_user_id) - the request still completed as a guest, but
          // silently staying "logged in" with a dead token means persona/history features
          // quietly stop working with no visible sign. Surface it and clear the stale
          // session instead of letting that drift unnoticed.
          setState((prev) => ({ ...prev, wsError: 'Your session expired — please sign in again.' }))
          onSessionExpired?.()
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the search service failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.loading ? { ...prev, loading: false } : prev))
      })
    },
    [wsUrl, accessToken, onSessionExpired],
  )

  return { ...state, search }
}
