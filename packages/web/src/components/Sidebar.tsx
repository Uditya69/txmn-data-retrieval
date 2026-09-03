import { useState } from 'react'
import type { Conversation } from '../types'
import ConfirmDialog from './ConfirmDialog'

type Props = {
  conversations: Conversation[]
  activeId: string | null
  collapsed: boolean
  onToggleCollapsed: () => void
  onSelect: (id: string) => void
  onNewChat: () => void
  onDelete: (id: string) => void
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

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  )
}

export default function Sidebar({ conversations, activeId, collapsed, onToggleCollapsed, onSelect, onNewChat, onDelete }: Props) {
  const [pendingDelete, setPendingDelete] = useState<{ id: string; title: string } | null>(null)

  // Both branches always render - which one is visible is a pure CSS decision (Tailwind's
  // `lg` breakpoint, 1024px), not a JS width check. A JS width check (window.innerWidth
  // read once at mount, or even behind a resize listener) only ever reacts to *state
  // changes App itself causes* - it doesn't re-render on the browser window being resized
  // by the user unless something wires that up, which is exactly the bug this replaced:
  // shrinking the tab after load never collapsed the sidebar because nothing was listening
  // for it. CSS media queries are inherently resize-reactive with zero JS, so they can't
  // have that bug.
  //
  // `lg` (1024px), not `md` (768px): a 260px sidebar plus real content needs more than a
  // phone-vs-desktop split to stay comfortable - collapsing already at a modestly narrowed
  // window (not just full mobile width) is the intended feel here.
  //
  // Below lg (max-lg:): the rail always shows and the full panel always hides, regardless
  // of `collapsed` - a screen this narrow has no room for a 260px sidebar no matter what the
  // user last manually toggled.
  // At/above lg: `collapsed` alone decides, exactly as before.
  const railClassName = collapsed ? 'flex' : 'hidden max-lg:flex'
  const fullClassName = collapsed ? 'hidden' : 'flex max-lg:hidden'

  return (
    <>
      <div
        className={`shrink-0 h-screen sticky top-0 flex-col items-center gap-3 py-4 px-2 ${railClassName}`}
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

    <div
      className={`shrink-0 h-screen sticky top-0 flex-col py-4 px-3 ${fullClassName}`}
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
          <li key={c.id} className="group flex items-center gap-1">
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
            <button
              // stopPropagation isn't actually load-bearing here - this button
              // is a sibling of the select button, not nested inside it, so a
              // click on it was never going to bubble into onSelect. Kept
              // anyway as a guard against that ever changing (e.g. if this
              // row is later wrapped in a single clickable container).
              onClick={(e) => {
                e.stopPropagation()
                setPendingDelete({ id: c.id, title: c.title })
              }}
              aria-label={`Delete "${c.title}"`}
              title="Delete conversation"
              // Hidden until the row is hovered/focused - a permanently-visible
              // trash icon next to every single chat title is noisy for an
              // action almost never taken. focus-visible keeps it reachable
              // via keyboard nav despite the hover gate.
              className="shrink-0 h-7 w-7 rounded-lg flex items-center justify-center cursor-pointer opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity duration-150"
              style={{ color: 'var(--text-faint)' }}
            >
              <TrashIcon />
            </button>
          </li>
        ))}
      </ul>

      {pendingDelete && (
        <ConfirmDialog
          title={`Delete "${pendingDelete.title}"?`}
          message="This cannot be undone."
          confirmLabel="Delete"
          onConfirm={() => {
            onDelete(pendingDelete.id)
            setPendingDelete(null)
          }}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
    </>
  )
}
