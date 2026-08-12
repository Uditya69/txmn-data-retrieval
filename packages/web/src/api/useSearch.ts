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
        socket.send(JSON.stringify({ query, mode, trace, rerank }))
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
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the search service failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.loading ? { ...prev, loading: false } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, search }
}
