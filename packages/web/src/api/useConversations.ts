import { useCallback, useState } from 'react'
import type { ChatMessage } from '../types'
import type { AiModeCitation, InstantResult, TraceStep } from './useSearch'
import type { RerankedHit } from '../lib/mergeResults'

export interface ConversationSummary {
  id: string
  title: string
  updated_at: string
}

// The shape actually persisted server-side (chat/repository.py's `messages`
// list) - flat {role, text} records, NOT the frontend's rich ChatMessage
// shape (which carries per-mode results, trace steps, etc. that only ever
// exist in-memory). loadConversation hydrates these into real ChatMessages
// below before handing them back to callers.
interface StoredMessage {
  role: 'user' | 'assistant'
  text: string
  // Only present on assistant messages, and only for turns saved after
  // retrieval-trace persistence landed - older stored conversations won't
  // have it, so this stays optional and hydration falls back to `{}`.
  citations?: Record<string, AiModeCitation>
}

interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[]
  created_at: string
}

// GET /conversations/{id}/traces - one document per Instant/AI Mode run this
// conversation ever made, oldest first (see chat/repository.py's
// list_retrieval_traces). `steps` mirrors the live `ai_mode_trace` websocket
// events (`{step, data}`) verbatim - the same shape TracePanel/TraceSection
// already render for a live search.
interface StoredTrace {
  mode: 'instant' | 'ai_mode'
  instant?: { doc_ids: string[]; steps: TraceStep[] } | null
  ai_mode?: {
    steps: TraceStep[]
    citations: Record<string, AiModeCitation>
    intent: string[]
    reasoning?: string | null
  } | null
}

// Turns the server's flat {role, text} records into the ChatMessage shape
// the rest of the app (ChatMessageView in particular) expects. An assistant
// message is hydrated into a "done" classic-mode result carrying its text as
// the AI Mode answer - it's the only mode we have a flat answer string for,
// and 'classic' is also this app's default mode.
//
// `traces` pairs up positionally: the Nth Instant-mode trace doc and the Nth
// AI-Mode trace doc line up with the Nth assistant message, because every
// stored turn today comes from a `mode: "both"` websocket request (the web
// client always sends "both" once signed in - see App.tsx). This breaks if a
// conversation ever mixes single-mode turns in, which this app's own UI
// cannot currently do.
export function hydrateStoredMessages(
  conversationId: string, stored: StoredMessage[], traces: StoredTrace[] = [],
): ChatMessage[] {
  const instantTraces = traces.filter((t) => t.mode === 'instant')
  const aiModeTraces = traces.filter((t) => t.mode === 'ai_mode')
  let lastQuestion = ''
  let turnIndex = 0
  return stored.map((m, index) => {
    const id = `${conversationId}-${index}`
    if (m.role === 'user') {
      lastQuestion = m.text
      return { id, role: 'user', text: m.text }
    }
    const instantTrace = instantTraces[turnIndex]
    const aiModeTrace = aiModeTraces[turnIndex]
    turnIndex += 1

    const instantSteps = instantTrace?.instant?.steps ?? []
    // The final fused/reranked hit list is just another captured step
    // (instant_reranked's `data.hits`) - the exact same objects Instant mode
    // sends live in `instant_result.reranked`, so InstantPane renders it
    // unmodified with no adaptation needed.
    const rerankedStep = instantSteps.find((s) => s.step === 'instant_reranked')
    const instant: InstantResult | undefined = rerankedStep
      ? {
          es: null, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null,
          reranked: (rerankedStep.data as { hits: RerankedHit[] }).hits, reranked_error: null,
        }
      : undefined

    return {
      id,
      role: 'assistant',
      question: lastQuestion,
      activeMode: 'classic',
      results: {
        classic: {
          status: 'done',
          aiMode: {
            ok: true, answer: m.text, citations: m.citations ?? {},
            reasoning: aiModeTrace?.ai_mode?.reasoning ?? null,
          },
          instant,
          traceSteps: [...instantSteps, ...(aiModeTrace?.ai_mode?.steps ?? [])],
        },
      },
    }
  })
}

export function useConversations(apiBaseUrl: string, token: string | null) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])

  const refresh = useCallback(async () => {
    if (!token) {
      setConversations([])
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) return
      const data = (await response.json()) as ConversationSummary[]
      setConversations(data)
    } catch {
      // Network failure: leave whatever list is already in state rather than
      // clearing the sidebar on a transient blip.
    }
  }, [apiBaseUrl, token])

  const loadConversation = useCallback(
    async (id: string): Promise<ChatMessage[]> => {
      if (!token) return []
      const [messagesResponse, tracesResponse] = await Promise.all([
        fetch(`${apiBaseUrl}/conversations/${id}`, { headers: { Authorization: `Bearer ${token}` } }),
        // Best-effort: a failed/unreachable traces fetch must not block loading the
        // conversation itself - it just falls back to no dev-trace/Instant-doc data.
        fetch(`${apiBaseUrl}/conversations/${id}/traces`, { headers: { Authorization: `Bearer ${token}` } }).catch(
          () => null,
        ),
      ])
      if (!messagesResponse.ok) return []
      const data = (await messagesResponse.json()) as ConversationDetail
      const traces = tracesResponse?.ok ? ((await tracesResponse.json()) as StoredTrace[]) : []
      return hydrateStoredMessages(id, data.messages, traces)
    },
    [apiBaseUrl, token],
  )

  const remove = useCallback(
    async (id: string) => {
      if (!token) return
      await fetch(`${apiBaseUrl}/conversations/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {})
      setConversations((prev) => prev.filter((c) => c.id !== id))
    },
    [apiBaseUrl, token],
  )

  // Synchronously resets the list to empty. Used on logout and on switching
  // to a different logged-in user, so a stale user-A conversation list can
  // never remain visible in the sidebar while user B's `refresh()` fetch is
  // still in flight.
  const clear = useCallback(() => {
    setConversations([])
  }, [])

  return { conversations, refresh, loadConversation, remove, clear }
}
