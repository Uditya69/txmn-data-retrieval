import { useState, type FormEvent } from 'react'

type Props = {
  onSubmit: (question: string) => void
  disabled: boolean
}

export default function ChatInput({ onSubmit, disabled }: Props) {
  const [draft, setDraft] = useState('')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!draft.trim() || disabled) return
    onSubmit(draft.trim())
    setDraft('')
  }

  return (
    <div className="sticky bottom-0 pt-3 pb-5" style={{ background: 'linear-gradient(to top, var(--ink) 60%, transparent)' }}>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a legal/tax question..."
          aria-label="Search query"
          className="w-full rounded-xl px-4 py-3.5 text-[15px] outline-none"
          style={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', color: 'var(--text)' }}
        />
        <button
          type="submit"
          disabled={disabled || !draft.trim()}
          className="rounded-xl px-6 py-3.5 text-[15px] font-medium cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          style={{ background: 'var(--accent-strong)', color: 'var(--accent-ink)' }}
        >
          Send
        </button>
      </form>
    </div>
  )
}
