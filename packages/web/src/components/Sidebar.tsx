import type { Conversation } from '../types'

type Props = {
  conversations: Conversation[]
  activeId: string | null
  collapsed: boolean
  onToggleCollapsed: () => void
  onSelect: (id: string) => void
  onNewChat: () => void
}

function ChevronIcon({ direction }: { direction: 'left' | 'right' }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {direction === 'left' ? <polyline points="15 18 9 12 15 6" /> : <polyline points="9 18 15 12 9 6" />}
    </svg>
  )
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  )
}

export default function Sidebar({ conversations, activeId, collapsed, onToggleCollapsed, onSelect, onNewChat }: Props) {
  if (collapsed) {
    return (
      <div
        className="shrink-0 h-screen sticky top-0 flex flex-col items-center gap-3 py-4 px-2"
        style={{ width: 56, borderRight: '1px solid var(--border-soft)', background: 'var(--surface)' }}
      >
        <button
          onClick={onToggleCollapsed}
          title="Expand sidebar"
          className="h-8 w-8 rounded-lg flex items-center justify-center cursor-pointer"
          style={{ color: 'var(--text-faint)' }}
        >
          <ChevronIcon direction="right" />
        </button>
        <button onClick={onNewChat} title="New chat" className="h-8 w-8 rounded-lg flex items-center justify-center cursor-pointer" style={{ color: 'var(--text-faint)' }}>
          <PlusIcon />
        </button>
      </div>
    )
  }

  return (
    <div
      className="shrink-0 h-screen sticky top-0 flex flex-col py-4 px-3"
      style={{ width: 260, borderRight: '1px solid var(--border-soft)', background: 'var(--surface)' }}
    >
      <div className="flex items-center justify-between px-1 mb-3">
        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
          Chats
        </span>
        <button onClick={onToggleCollapsed} title="Collapse sidebar" className="h-7 w-7 rounded-lg flex items-center justify-center cursor-pointer" style={{ color: 'var(--text-faint)' }}>
          <ChevronIcon direction="left" />
        </button>
      </div>

      <button
        onClick={onNewChat}
        className="flex items-center gap-2 text-sm rounded-lg px-2.5 py-2 mb-3 cursor-pointer"
        style={{ background: 'var(--surface-raised)', color: 'var(--text)', border: '1px solid var(--border-soft)' }}
      >
        <PlusIcon />
        New chat
      </button>

      <ul className="flex-1 overflow-y-auto space-y-0.5 -mx-1">
        {conversations.map((c) => (
          <li key={c.id}>
            <button
              onClick={() => onSelect(c.id)}
              className="text-sm text-left truncate block w-full rounded-lg px-2.5 py-2 cursor-pointer"
              style={{
                background: c.id === activeId ? 'var(--surface-raised)' : 'transparent',
                color: c.id === activeId ? 'var(--text)' : 'var(--text-muted)',
              }}
            >
              {c.title}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
