import { useEffect, useRef, useState, type FormEvent } from 'react'

type Props = {
  onSubmit: (question: string) => void
  disabled: boolean
  /** Changing this (e.g. the active conversation id, including to null on "New chat")
   * refocuses the input - same as the initial-mount autofocus below, but re-triggered
   * without a remount so switching/starting a conversation never leaves focus stranded
   * on whatever was last clicked (a sidebar item, the "New chat" button). */
  focusKey?: string | null
}

export default function ChatInput({ onSubmit, disabled, focusKey }: Props) {
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  // Focused on first render (page load/refresh) so the user can start typing
  // immediately - no click into the box required.
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    inputRef.current?.focus()
  }, [focusKey])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!draft.trim() || disabled) return
    onSubmit(draft.trim())
    setDraft('')
  }

  const canSend = !disabled && draft.trim().length > 0

  return (
    // No page-colored panel behind this - the pill floats directly on the app
    // background (ChatGPT-style), instead of sitting inside a boxed bar.
    <div className="sticky bottom-0 pb-4 pt-2">
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 rounded-full pl-5 pr-2 py-2"
        style={{ background: 'var(--surface-raised)', border: '1.5px solid var(--border-strong)' }}
      >
        <input
          ref={inputRef}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a legal/tax question..."
          aria-label="Search query"
          className="flex-1 bg-transparent text-[15px] outline-none"
          style={{ color: 'var(--text)' }}
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send"
          className="shrink-0 h-9 w-9 rounded-full flex items-center justify-center cursor-pointer disabled:cursor-not-allowed transition-colors duration-150"
          style={{
            background: canSend ? 'var(--accent-strong)' : 'var(--surface-hover)',
            color: canSend ? 'var(--accent-ink)' : 'var(--text-faint)',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5" />
            <path d="M5 12l7-7 7 7" />
          </svg>
        </button>
      </form>
    </div>
  )
}
