import { useState } from 'react'
import { createPortal } from 'react-dom'

export interface AuthModalProps {
  onSignup: (email: string, password: string) => Promise<boolean>
  onLogin: (email: string, password: string) => Promise<boolean>
  loading: boolean
  error: string | null
  onClose: () => void
}

export default function AuthModal({ onSignup, onLogin, loading, error, onClose }: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'signup'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const ok = mode === 'login' ? await onLogin(email, password) : await onSignup(email, password)
    if (ok) onClose()
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'oklch(0 0 0 / 0.4)' }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-xl p-6"
        style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text)' }}>
          {mode === 'login' ? 'Sign in' : 'Create an account'}
        </h2>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="rounded-lg px-3 py-2 text-sm"
            style={{ border: '1px solid var(--border-soft)', background: 'var(--surface-raised)', color: 'var(--text)' }}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            maxLength={72}
            className="rounded-lg px-3 py-2 text-sm"
            style={{ border: '1px solid var(--border-soft)', background: 'var(--surface-raised)', color: 'var(--text)' }}
          />

          {error && (
            <p className="text-sm" style={{ color: 'var(--danger)' }}>
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="rounded-lg px-3 py-2 text-sm font-medium cursor-pointer"
            style={{ background: 'var(--text)', color: 'var(--surface)' }}
          >
            {loading ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Sign up'}
          </button>
        </form>

        <button
          type="button"
          onClick={() => setMode(mode === 'login' ? 'signup' : 'login')}
          className="mt-3 text-sm cursor-pointer"
          style={{ color: 'var(--text-faint)' }}
        >
          {mode === 'login' ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
        </button>
      </div>
    </div>,
    document.body,
  )
}
