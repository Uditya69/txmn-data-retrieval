import { useState } from 'react'
import AuthModal from './AuthModal'

export interface AuthMenuProps {
  email: string | null
  loading: boolean
  error: string | null
  onSignup: (email: string, password: string) => Promise<boolean>
  onLogin: (email: string, password: string) => Promise<boolean>
  onLogout: () => void
}

export default function AuthMenu({ email, loading, error, onSignup, onLogin, onLogout }: AuthMenuProps) {
  const [modalOpen, setModalOpen] = useState(false)

  if (email) {
    return (
      <div className="inline-flex items-center gap-2 text-sm" style={{ color: 'var(--text-faint)' }}>
        <span>{email}</span>
        <button type="button" onClick={onLogout} className="cursor-pointer underline">
          Sign out
        </button>
      </div>
    )
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="text-sm px-3 py-1.5 rounded-full font-medium cursor-pointer"
        style={{ background: 'var(--surface-raised)', color: 'var(--text)', border: '1px solid var(--border-soft)' }}
      >
        Sign in
      </button>
      {modalOpen && (
        <AuthModal onSignup={onSignup} onLogin={onLogin} loading={loading} error={error} onClose={() => setModalOpen(false)} />
      )}
    </>
  )
}
