import { useCallback, useState } from 'react'
import type { ChatMessage } from '../types'

export interface ConversationSummary {
  id: string
  title: string
  updated_at: string
}

interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[]
  created_at: string
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
      return data.messages
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

  return { conversations, refresh, loadConversation, remove }
}
