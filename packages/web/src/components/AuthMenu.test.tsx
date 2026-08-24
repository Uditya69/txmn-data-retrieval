import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AuthMenu from './AuthMenu'

describe('AuthMenu', () => {
  it('shows a Sign in button when logged out', () => {
    render(<AuthMenu email={null} loading={false} error={null} onSignup={vi.fn()} onLogin={vi.fn()} onLogout={vi.fn()} />)
    expect(screen.getByText('Sign in')).toBeInTheDocument()
  })

  it('opens the auth modal when Sign in is clicked', () => {
    render(<AuthMenu email={null} loading={false} error={null} onSignup={vi.fn()} onLogin={vi.fn()} onLogout={vi.fn()} />)
    fireEvent.click(screen.getByText('Sign in'))
    expect(screen.getByPlaceholderText('Email')).toBeInTheDocument()
  })

  it('shows the account email and a sign out control when logged in', () => {
    const onLogout = vi.fn()
    render(
      <AuthMenu email="alice@example.com" loading={false} error={null} onSignup={vi.fn()} onLogin={vi.fn()} onLogout={onLogout} />,
    )
    expect(screen.getByText('alice@example.com')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Sign out'))
    expect(onLogout).toHaveBeenCalled()
  })
})
