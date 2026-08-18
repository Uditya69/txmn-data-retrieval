import { useCallback, useState } from 'react'
import type { ChatMessage } from '../types'

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
}

interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[]
  created_at: string
}

// Turns the server's flat {role, text} records into the ChatMessage shape
// the rest of the app (ChatMessageView in particular) expects. An assistant
// message is hydrated into a "done" classic-mode result carrying its text as
// the AI Mode answer - it's the only mode we have a flat answer string for,
// and 'classic' is also this app's default mode.
export function hydrateStoredMessages(conversationId: string, stored: StoredMessage[]): ChatMessage[] {
  let lastQuestion = ''
  return stored.map((m, index) => {
    const id = `${conversationId}-${index}`
    if (m.role === 'user') {
      lastQuestion = m.text
      return { id, role: 'user', text: m.text }
    }
    return {
      id,
      role: 'assistant',
      question: lastQuestion,
      activeMode: 'classic',
      results: {
        classic: {
          status: 'done',
          aiMode: { ok: true, answer: m.text, citations: {} },
          traceSteps: [],
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
      const response = await fetch(`${apiBaseUrl}/conversations/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) return []
      const data = (await response.json()) as ConversationDetail
      return hydrateStoredMessages(id, data.messages)
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
