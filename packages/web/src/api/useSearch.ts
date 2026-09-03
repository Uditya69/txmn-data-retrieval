import { useCallback, useRef, useState } from 'react'
import type { EsHit, MilvusByCollection, RerankedHit } from '../lib/mergeResults'

export type AiModeCitation = Record<string, unknown>

export interface DocMeta {
  category: string | null
  group: string | null
  judge?: string[]
  party?: string[]
  date?: string
  viewcount?: number
  documenttypeboost?: number
  court_boost?: number
  fullcitation?: string
  referenced_act?: string[]
  referenced_section?: string[]
  cases_referred?: string[]
  // The specific Act/Rule instrument this doc belongs to (e.g. "Companies Act, 2013") -
  // distinct from `referenced_act` (a cross-reference to OTHER acts this section relates
  // to, never the doc's own). See common.es_client.fetch_doc_categories's docstring.
  act_name?: string
}

export interface InstantResult {
  es: EsHit[] | null
  es_error: string | null
  milvus: MilvusByCollection | null
  milvus_sparse: MilvusByCollection | null
  milvus_error: string | null
  reranked?: RerankedHit[] | null
  reranked_error?: string | null
  // "category | group" badge data, keyed by doc_id - populated for every card
  // regardless of which engine (ES/Milvus/reranked) surfaced it. See
  // common.es_client.fetch_doc_categories.
  doc_meta?: Record<string, DocMeta> | null
  // Sectioned result view, boost_source="repotaxmannapi" only - keyed by raw ES
  // groups.group.name value (e.g. "ACT", "Experts Opinion"), already ordered by fixed
  // priority (see common.es_client._GROUPED_SECTION_PRIORITY). null/absent under
  // boost_source="sum" - grouping has no sum-mode equivalent.
  grouped_es?: Record<string, EsHit[]> | null
  grouped_es_error?: string | null
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
): SearchState & {
  search: (
    query: string, trace: boolean, mode?: SearchMode, rrf?: boolean, autoRoute?: boolean,
    conversationId?: string, boost?: boolean, boostSource?: 'sum' | 'repotaxmannapi',
    page?: number, pageSize?: number,
  ) => void
} {
  const [state, setState] = useState<SearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (
      query: string, trace: boolean, mode: SearchMode = 'both', rrf: boolean = false,
      autoRoute: boolean = false, conversationId?: string, boost: boolean = false,
      boostSource: 'sum' | 'repotaxmannapi' = 'sum', page?: number, pageSize?: number,
    ) => {
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
        const payload: Record<string, unknown> = {
          query, mode, trace, rrf, auto_route: autoRoute, boost, boost_source: boostSource,
        }
        if (page !== undefined) payload.page = page
        if (pageSize !== undefined) payload.page_size = pageSize
        if (accessToken) payload.access_token = accessToken
        if (conversationId) payload.conversation_id = conversationId
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
              doc_meta: message.doc_meta ?? null,
              grouped_es: message.grouped_es ?? null,
              grouped_es_error: message.grouped_es_error ?? null,
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
          // see ws.py's _resolve_user_id) - this specific request already completed as a
          // guest and can't be retroactively fixed, but onSessionExpired (wired to
          // useAuth's silent refresh, not an immediate logout) gives the *next* request
          // a fresh token before the user notices anything. Only actually signs the user
          // out if the refresh token is also dead - the common case resolves silently.
          setState((prev) => ({ ...prev, wsError: 'Reconnecting your session…' }))
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
