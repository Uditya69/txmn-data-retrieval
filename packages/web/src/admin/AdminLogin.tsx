import { useState } from 'react'

interface AdminLoginProps {
  onSubmit: (token: string) => void
  error: string | null
}

export default function AdminLogin({ onSubmit, error }: AdminLoginProps) {
  const [token, setToken] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onSubmit(token)
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-sm mx-auto mt-24 flex flex-col gap-3">
      <h1 className="text-lg font-semibold">Admin</h1>
      <input
        type="password"
        value={token}
        onChange={(e) => setToken(e.target.value)}
        placeholder="Admin token"
        className="border rounded px-3 py-2"
        autoFocus
      />
      {error && <p className="text-sm" style={{ color: 'crimson' }}>{error}</p>}
      <button type="submit" className="border rounded px-3 py-2 font-medium">
        Enter
      </button>
    </form>
  )
}
