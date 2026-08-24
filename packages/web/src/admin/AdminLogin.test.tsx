import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AdminLogin from './AdminLogin'

describe('AdminLogin', () => {
  it('submits the entered token', async () => {
    const onSubmit = vi.fn()
    render(<AdminLogin onSubmit={onSubmit} error={null} />)

    await userEvent.type(screen.getByPlaceholderText('Admin token'), 'my-secret')
    await userEvent.click(screen.getByRole('button', { name: /enter/i }))

    expect(onSubmit).toHaveBeenCalledWith('my-secret')
  })

  it('shows an error message when provided', () => {
    render(<AdminLogin onSubmit={vi.fn()} error="Invalid token." />)
    expect(screen.getByText('Invalid token.')).toBeInTheDocument()
  })
})
