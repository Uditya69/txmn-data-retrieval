import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DevModeToggle from './DevModeToggle'

describe('DevModeToggle', () => {
  it('calls onToggle with the new checked value', () => {
    const onToggle = vi.fn()
    render(<DevModeToggle devMode={false} onToggle={onToggle} />)

    fireEvent.click(screen.getByLabelText('Dev mode'))

    expect(onToggle).toHaveBeenCalledWith(true)
  })
})
