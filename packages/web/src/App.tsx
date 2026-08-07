import { useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatInput from './components/ChatInput'
import { ChatMessageView } from './components/ChatMessageView'
import DocumentReader from './components/DocumentReader'
import DevModeToggle from './components/DevModeToggle'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import { resolveWsUrl, resolveAgentWsUrl, resolveApiBaseUrl } from './lib/config'
import type { ChatMessage, ChatMode, Conversation, ResultState } from './types'

const CONVERSATIONS_KEY = 'taxmann-retrieval-conversations'
const SIDEBAR_KEY = 'taxmann-retrieval-sidebar-collapsed'

let nextId = 0
function genId(prefix: string) {
  nextId += 1
  return `${prefix}-${nextId}`
}

function isValidMessage(m: unknown): m is ChatMessage {
  if (!m || typeof m !== 'object') return false
  const msg = m as Record<string, unknown>
  if (msg.role === 'user') return typeof msg.text === 'string'
  if (msg.role === 'assistant') return typeof msg.question === 'string' && typeof msg.results === 'object' && msg.results !== null
  return false
}

function loadConversations(): Conversation[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(CONVERSATIONS_KEY) ?? '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (c): c is Conversation => c && typeof c === 'object' && Array.isArray(c.messages) && c.messages.every(isValidMessage),
    )
  } catch {
    return []
  }
}

// Trace steps carry full rerank chunk text and the synthesis prompt (see
// ai_mode/citations.py, ai_mode/synthesize.py) - they're only ever shown
// live behind devMode (ChatMessageView's TraceSection), so persisting them
// serves no purpose and is what blows past localStorage's ~5-10MB quota
// after a handful of chats.
export function toPersistable(conversations: Conversation[]): Conversation[] {
  return conversations.map((c) => ({
    ...c,
    messages: c.messages.map((m) =>
      m.role === 'assistant'
        ? {
            ...m,
            results: Object.fromEntries(
              Object.entries(m.results).map(([mode, r]) => [mode, r ? { ...r, traceSteps: [] } : r]),
            ) as Partial<Record<ChatMode, ResultState>>,
          }
        : m,
    ),
  }))
}

function isQuotaExceeded(err: unknown): boolean {
  return err instanceof DOMException && (err.name === 'QuotaExceededError' || err.code === 22)
}

// Drops the oldest conversations (they're appended at the end - see
// handleSubmit's `[newConversation, ...prev]`) until the write fits, instead
// of letting an uncaught QuotaExceededError crash the whole app.
export function persistConversations(conversations: Conversation[]) {
  const payload = toPersistable(conversations)
  for (let keep = payload.length; keep >= 0; keep -= 1) {
    try {
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(payload.slice(0, keep)))
      return
    } catch (err) {
      if (!isQuotaExceeded(err)) {
        console.error('Failed to persist conversations', err)
        return
      }
    }
  }
}

function titleFromQuestion(question: string) {
  return question.length > 48 ? `${question.slice(0, 48)}…` : question
}

function loadingResult(): ResultState {
  return { status: 'loading', traceSteps: [] }
}

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function App() {
  const wsUrl = resolveWsUrl()
  const agentWsUrl = resolveAgentWsUrl()
  const classicSearch = useSearch(wsUrl)
  const agentSearch = useAgentSearch(agentWsUrl)

  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem(SIDEBAR_KEY) === '1')
  const [mode, setMode] = useState<ChatMode>('classic')
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [openDocId, setOpenDocId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const pendingClassicRef = useRef<{ conversationId: string; assistantId: string } | null>(null)
  const pendingAgentRef = useRef<{ conversationId: string; assistantId: string } | null>(null)

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null
  const messages = activeConversation?.messages ?? []

  useEffect(() => {
    persistConversations(conversations)
  }, [conversations])

  useEffect(() => {
    localStorage.setItem(SIDEBAR_KEY, sidebarCollapsed ? '1' : '0')
  }, [sidebarCollapsed])

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [messages])

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
      classicSearch.search(question, true, 'both')
    } else {
      pendingAgentRef.current = { conversationId, assistantId }
      agentSearch.search(question)
    }
  }

  function handleNewChat() {
    setActiveId(null)
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
      conversationId = genId('conv')
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

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--ink)' }}>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
        onSelect={setActiveId}
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

            <div className="ml-auto">
              <DevModeToggle devMode={devMode} onToggle={setDevMode} />
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
                <ChatMessageView key={m.id} message={m} devMode={devMode} onOpenDocument={setOpenDocId} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}

          {wsError && (
            <p className="text-sm rounded-lg p-3 mb-3" style={{ color: 'var(--danger)', background: 'var(--danger-soft)' }}>
              {wsError}
            </p>
          )}

          <div className="max-w-3xl mx-auto w-full">
            <ChatInput onSubmit={handleSubmit} disabled={pending} />
          </div>
        </main>
      </div>

      <DocumentReader docId={openDocId} apiBaseUrl={resolveApiBaseUrl(wsUrl)} onClose={() => setOpenDocId(null)} onOpenDocument={setOpenDocId} />
    </div>
  )
}
