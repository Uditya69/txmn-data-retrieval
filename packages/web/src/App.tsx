import { useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatInput from './components/ChatInput'
import { ChatMessageView } from './components/ChatMessageView'
import DocumentReader from './components/DocumentReader'
import DevModeToggle from './components/DevModeToggle'
import RerankToggle from './components/RerankToggle'
import AuthMenu from './components/AuthMenu'
import { useSearch } from './api/useSearch'
import { useAuth } from './api/useAuth'
import { useConversations } from './api/useConversations'
import { resolveWsUrl, resolveApiBaseUrl } from './lib/config'
import type { ChatMessage, ChatMode, Conversation, ResultState } from './types'

const MODE: ChatMode = 'classic'

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

// Hidden-by-default real pagination (2026-09-02) - off unless explicitly built with this
// flag set; when off, InstantPane's existing client-side 10-per-page slice over a flat
// 20-result fetch is completely unchanged. No UI checkbox for this, deliberately -
// purely a build-time flag.
const PAGINATION_ENABLED = import.meta.env.VITE_ENABLE_PAGINATION === 'true'

export default function App() {
  const wsUrl = resolveWsUrl()
  const apiBaseUrl = resolveApiBaseUrl(wsUrl)
  const auth = useAuth(apiBaseUrl)
  const classicSearch = useSearch(wsUrl, auth.token, auth.refresh)

  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const remoteConversations = useConversations(apiBaseUrl, auth.token)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [rrf, setRrf] = useState(true)
  const [boost, setBoost] = useState(true)
  const [autoRoute, setAutoRoute] = useState(true)
  const [showReasoning, setShowReasoning] = useState(true)
  const [openDocId, setOpenDocId] = useState<string | null>(null)
  // The query that led to the currently-open document, so DocumentReader can apply the
  // same yellow-highlight treatment the result list already gives matched terms. Sticky
  // across in-document cross-reference navigation (openDocument called with no query) -
  // jumping to a linked section keeps the highlight context of the search that got you
  // there, rather than losing it.
  const [openDocQuery, setOpenDocQuery] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  function openDocument(docId: string, query?: string) {
    if (query !== undefined) setOpenDocQuery(query)
    setOpenDocId(docId)
  }

  // `kind` distinguishes a full ('both'-mode) turn from an instant-only paged re-fetch, so
  // the reflect-effect below knows whether to overwrite the whole ResultState (full turn)
  // or patch only `instant` into whatever's already there (paged re-fetch - must not clobber
  // an already-rendered aiMode/status, see C2 in the 2026-09-02 review fix).
  const pendingClassicRef = useRef<{ conversationId: string; assistantId: string; kind: 'full' | 'instant_page' } | null>(null)

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
    if (pending.kind === 'instant_page') {
      // A paged re-fetch only ever runs mode:'instant' - classicSearch.aiMode/loading here
      // reflect that instant-only request, not the (already-finished) AI Mode answer this
      // message may already be showing. Patch only `instant` into whatever ResultState the
      // message already has, so an existing aiMode/'done' status is never overwritten.
      patchResult(pending.conversationId, pending.assistantId, 'classic', (prev) => ({
        ...prev,
        instant: classicSearch.instant,
      }))
      return
    }
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

  const [instantPage, setInstantPage] = useState(1)

  function fetchInstantPage(conversationId: string, assistantId: string, question: string, page: number) {
    if (!PAGINATION_ENABLED) return
    setInstantPage(page)
    pendingClassicRef.current = { conversationId, assistantId, kind: 'instant_page' }
    classicSearch.search(question, true, 'instant', rrf, autoRoute, auth.token ? conversationId : undefined, boost, page, 20)
  }

  function runQuery(conversationId: string, assistantId: string, question: string) {
    pendingClassicRef.current = { conversationId, assistantId, kind: 'full' }
    setInstantPage(1)
    classicSearch.search(question, true, 'both', rrf, autoRoute, auth.token ? conversationId : undefined, boost)
  }

  function handleNewChat() {
    setActiveId(null)
  }

  async function handleDeleteConversation(id: string) {
    // Sidebar already confirmed via its own dialog before calling this.
    // remove() itself no-ops for a guest (no token) - always safe to call, and
    // this still needs to drop the conversation from local-only state either way.
    await remoteConversations.remove(id)
    setConversations((prev) => prev.filter((c) => c.id !== id))
    if (activeId === id) setActiveId(null)
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
      activeMode: MODE,
      results: { [MODE]: loadingResult() },
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

    runQuery(conversationId, assistantId, question)
  }

  const pending = classicSearch.loading
  const wsError = classicSearch.wsError

  // Merge in any local conversation not already represented remotely (by id) -
  // a brand-new chat isn't in remoteConversations until the next refresh().
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
        onDelete={handleDeleteConversation}
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

            <div className="ml-auto flex items-center gap-3">
              <RerankToggle label="RRF" checked={rrf} onToggle={setRrf} />
              <RerankToggle label="Boost" checked={boost} onToggle={setBoost} />
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
                  onOpenDocument={(docId) => openDocument(docId, m.role === 'assistant' ? m.question : undefined)}
                  paginationEnabled={PAGINATION_ENABLED}
                  currentPage={instantPage}
                  onFetchPage={(page) => fetchInstantPage(activeId ?? '', m.id, m.role === 'assistant' ? m.question : '', page)}
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
              <ChatInput onSubmit={handleSubmit} disabled={pending} focusKey={activeId} />
            </div>
          </div>
        </main>
      </div>

      <DocumentReader
        docId={openDocId}
        apiBaseUrl={apiBaseUrl}
        query={openDocQuery}
        onClose={() => {
          setOpenDocId(null)
          setOpenDocQuery(null)
        }}
        onOpenDocument={(docId) => openDocument(docId)}
      />
    </div>
  )
}
