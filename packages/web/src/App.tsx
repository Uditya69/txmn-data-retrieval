import { useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatInput from './components/ChatInput'
import { ChatMessageView } from './components/ChatMessageView'
import DocumentReader from './components/DocumentReader'
import DevModeToggle from './components/DevModeToggle'
import RerankToggle from './components/RerankToggle'
import AuthMenu from './components/AuthMenu'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import { useAuth } from './api/useAuth'
import { useConversations } from './api/useConversations'
import { resolveWsUrl, resolveAgentWsUrl, resolveApiBaseUrl } from './lib/config'
import type { ChatMessage, ChatMode, Conversation, ResultState } from './types'

let nextId = 0
function genId(prefix: string) {
  nextId += 1
  return `${prefix}-${nextId}`
}

function titleFromQuestion(question: string) {
  return question.length > 48 ? `${question.slice(0, 48)}…` : question
}

function loadingResult(): ResultState {
  return { status: 'loading', traceSteps: [] }
}

function readDevModeFromUrl(): boolean {
  // Defaults on - ?dev=0 is the explicit opt-out, not ?dev=1 the opt-in.
  return new URLSearchParams(window.location.search).get('dev') !== '0'
}

export default function App() {
  const wsUrl = resolveWsUrl()
  const agentWsUrl = resolveAgentWsUrl()
  const apiBaseUrl = resolveApiBaseUrl(wsUrl)
  const auth = useAuth(apiBaseUrl)
  const classicSearch = useSearch(wsUrl, auth.token, auth.refresh)
  const agentSearch = useAgentSearch(agentWsUrl)

  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const remoteConversations = useConversations(apiBaseUrl, auth.token)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [mode, setMode] = useState<ChatMode>('classic')
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [rrf, setRrf] = useState(false)
  const [autoRoute, setAutoRoute] = useState(false)
  const [showReasoning, setShowReasoning] = useState(false)
  const [openDocId, setOpenDocId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const pendingClassicRef = useRef<{ conversationId: string; assistantId: string } | null>(null)
  const pendingAgentRef = useRef<{ conversationId: string; assistantId: string } | null>(null)

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null
  const messages = activeConversation?.messages ?? []

  // Login/logout changes where conversations come from - fetch the remote
  // list for logged-in users, or clear in-memory state on logout so the
  // previous user's chats don't leak to the next guest session.
  useEffect(() => {
    // Clear the remote list synchronously first, on every token change
    // (login, logout, or switching to a different logged-in user) - so a
    // stale previous-user conversation list is never visibly shown in the
    // sidebar while the new user's `refresh()` fetch is still in flight.
    remoteConversations.clear()
    if (auth.token) {
      remoteConversations.refresh()
    } else {
      setConversations([])
      setActiveId(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.token])

  // Auto-scroll only when a message is actually added (new question asked, or
  // switching conversations) - not on every streaming patch. patchResult
  // replaces `results` inside the last message on every instant/trace/answer
  // chunk that arrives, which gives `messages` a new array reference each
  // time; scrolling on every one of those fights the user's own scroll-up
  // while results are still streaming in.
  const scrollTrackRef = useRef<{ activeId: string | null; count: number }>({ activeId: null, count: 0 })
  useEffect(() => {
    const prev = scrollTrackRef.current
    const shouldScroll = activeId !== prev.activeId || messages.length > prev.count
    scrollTrackRef.current = { activeId, count: messages.length }
    if (shouldScroll) {
      bottomRef.current?.scrollIntoView?.({ behavior: 'smooth' })
    }
  }, [messages, activeId])

  function updateConversationMessages(id: string, updater: (msgs: ChatMessage[]) => ChatMessage[]) {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, messages: updater(c.messages) } : c)))
  }

  function patchResult(conversationId: string, assistantId: string, targetMode: ChatMode, updater: (r: ResultState) => ResultState) {
    updateConversationMessages(conversationId, (msgs) =>
      msgs.map((m) =>
        m.id === assistantId && m.role === 'assistant'
          ? { ...m, results: { ...m.results, [targetMode]: updater(m.results[targetMode] ?? loadingResult()) } }
          : m,
      ),
    )
  }

  // Reflect the classic-mode hook's live state into whichever turn is currently pending.
  useEffect(() => {
    const pending = pendingClassicRef.current
    if (!pending) return
    patchResult(pending.conversationId, pending.assistantId, 'classic', () => ({
      status: classicSearch.loading ? 'loading' : classicSearch.aiMode ? 'done' : 'loading',
      instant: classicSearch.instant,
      aiMode: classicSearch.aiMode,
      traceSteps: classicSearch.traceSteps,
    }))
    if (!classicSearch.loading && classicSearch.aiMode && auth.token) {
      remoteConversations.refresh()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classicSearch.instant, classicSearch.aiMode, classicSearch.traceSteps, classicSearch.loading])

  useEffect(() => {
    const pending = pendingAgentRef.current
    if (!pending) return
    patchResult(pending.conversationId, pending.assistantId, 'agent', () => ({
      status: agentSearch.loading ? 'loading' : agentSearch.result ? 'done' : 'loading',
      agent: agentSearch.result,
      traceSteps: agentSearch.traceSteps,
    }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentSearch.result, agentSearch.traceSteps, agentSearch.loading])

  function runQuery(conversationId: string, assistantId: string, question: string, targetMode: ChatMode) {
    if (targetMode === 'classic') {
      pendingClassicRef.current = { conversationId, assistantId }
      classicSearch.search(question, true, 'both', rrf, autoRoute, auth.token ? conversationId : undefined)
    } else {
      pendingAgentRef.current = { conversationId, assistantId }
      agentSearch.search(question)
    }
  }

  function handleNewChat() {
    setActiveId(null)
  }

  async function handleSelectConversation(id: string) {
    if (auth.token) {
      const existing = conversations.find((c) => c.id === id)
      if (!existing) {
        const messages = await remoteConversations.loadConversation(id)
        const summary = remoteConversations.conversations.find((c) => c.id === id)
        setConversations((prev) => [...prev, { id, title: summary?.title ?? id, messages }])
      }
    }
    setActiveId(id)
  }

  function handleSubmit(question: string) {
    const userMsg: ChatMessage = { id: genId('msg'), role: 'user', text: question }
    const assistantId = genId('msg')
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      question,
      activeMode: mode,
      results: { [mode]: loadingResult() },
    }

    let conversationId = activeId
    if (!conversationId) {
      // Sent to the server as the conversation's Mongo _id (see App's runQuery
      // -> useSearch's conversation_id payload field) - must be globally
      // unique across users, not just unique within this page load. genId's
      // module-scoped counter resets to "conv-1" on every fresh page load, so
      // two different users' first conversations would collide and the
      // second write would silently clobber the first (repository.py's
      // create_conversation upserts by _id). crypto.randomUUID() avoids that.
      conversationId = crypto.randomUUID()
      const newConversation: Conversation = { id: conversationId, title: titleFromQuestion(question), messages: [userMsg, assistantMsg] }
      setConversations((prev) => [newConversation, ...prev])
      setActiveId(conversationId)
    } else {
      updateConversationMessages(conversationId, (msgs) => [...msgs, userMsg, assistantMsg])
    }

    runQuery(conversationId, assistantId, question, mode)
  }

  function switchMode(m: ChatMode) {
    setMode(m)
    if (!activeId) return

    const conv = conversations.find((c) => c.id === activeId)
    const last = conv?.messages[conv.messages.length - 1]
    if (!last || last.role !== 'assistant') return

    const alreadyFetched = Boolean(last.results[m])
    updateConversationMessages(activeId, (msgs) =>
      msgs.map((msg, i) =>
        i === msgs.length - 1 && msg.role === 'assistant'
          ? { ...msg, activeMode: m, results: alreadyFetched ? msg.results : { ...msg.results, [m]: loadingResult() } }
          : msg,
      ),
    )

    if (!alreadyFetched) {
      runQuery(activeId, last.id, last.question, m)
    }
  }

  const pending = classicSearch.loading || agentSearch.loading
  const wsError = classicSearch.wsError || agentSearch.wsError

  // Agent-mode conversations never get a conversation_id wired through /ws/agent
  // (out of scope for this fix wave - see design spec), so they're never
  // persisted server-side and never show up in remoteConversations. Without
  // this merge, a logged-in user's agent-mode chat would vanish from the
  // sidebar (and become unreachable) the moment they start a new chat, since
  // the sidebar for logged-in users otherwise sources ONLY the remote list.
  // Merge in any local conversation not already represented remotely (by id).
  const sidebarConversations = auth.token
    ? [
        ...conversations.filter((c) => !remoteConversations.conversations.some((rc) => rc.id === c.id)),
        ...remoteConversations.conversations.map((c) => ({ id: c.id, title: c.title, messages: [] })),
      ]
    : conversations

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--ink)' }}>
      <Sidebar
        conversations={sidebarConversations}
        activeId={activeId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-10 backdrop-blur-sm" style={{ background: 'oklch(0.99 0.002 90 / 0.9)', borderBottom: '1px solid var(--border-soft)' }}>
          <div className="relative w-full mx-auto px-6 py-4 flex items-center">
            <button
              onClick={handleNewChat}
              className="text-lg font-semibold tracking-tight cursor-pointer"
              style={{ color: 'var(--text)' }}
              title="Start a new chat"
            >
              Taxmann Retrieval
            </button>

            <div
              className="absolute left-1/2 -translate-x-1/2 inline-flex gap-1 p-1 rounded-full"
              style={{ background: 'var(--surface-raised)', border: '1px solid var(--border-soft)' }}
            >
              {(['classic', 'agent'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => switchMode(m)}
                  className="text-xs px-4 py-1.5 rounded-full font-medium cursor-pointer capitalize"
                  style={{
                    background: mode === m ? 'var(--surface)' : 'transparent',
                    color: mode === m ? 'var(--text)' : 'var(--text-faint)',
                    boxShadow: mode === m ? '0 1px 2px oklch(0 0 0 / 0.06)' : 'none',
                  }}
                >
                  {m}
                </button>
              ))}
            </div>

            <div className="ml-auto flex items-center gap-3">
              <RerankToggle label="RRF" checked={rrf} onToggle={setRrf} />
              <RerankToggle label="Auto-Route" checked={autoRoute} onToggle={setAutoRoute} />
              <RerankToggle label="Reasoning" checked={showReasoning} onToggle={setShowReasoning} />
              <DevModeToggle devMode={devMode} onToggle={setDevMode} />
              <AuthMenu
                email={auth.email}
                loading={auth.loading}
                error={auth.error}
                onSignup={auth.signup}
                onLogin={auth.login}
                onLogout={auth.logout}
              />
            </div>
          </div>
        </header>

        <main className="flex-1 w-full mx-auto px-6 flex flex-col min-w-0">
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col justify-center max-w-3xl mx-auto w-full">
              <p className="text-center" style={{ color: 'var(--text-faint)' }}>
                Ask a question about Indian tax case law to get started.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col gap-4 py-6">
              {messages.map((m) => (
                <ChatMessageView
                  key={m.id}
                  message={m}
                  devMode={devMode}
                  showReasoning={showReasoning}
                  onOpenDocument={setOpenDocId}
                />
              ))}
              <div ref={bottomRef} />
            </div>
          )}

          <div className="sticky bottom-0">
            {wsError && (
              <p className="text-sm rounded-lg p-3 mb-3 max-w-3xl mx-auto" style={{ color: 'var(--danger)', background: 'var(--danger-soft)' }}>
                {wsError}
              </p>
            )}

            <div className="max-w-3xl mx-auto w-full">
              <ChatInput onSubmit={handleSubmit} disabled={pending} />
            </div>
          </div>
        </main>
      </div>

      <DocumentReader docId={openDocId} apiBaseUrl={apiBaseUrl} onClose={() => setOpenDocId(null)} onOpenDocument={setOpenDocId} />
    </div>
  )
}
