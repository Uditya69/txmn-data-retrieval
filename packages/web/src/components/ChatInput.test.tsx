import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChatInput from './ChatInput'

describe('ChatInput', () => {
  it('autofocuses the search input on mount, so no click is needed before typing', () => {
    render(<ChatInput onSubmit={vi.fn()} disabled={false} />)

    expect(screen.getByLabelText('Search query')).toHaveFocus()
  })

  it('refocuses the input when focusKey changes (e.g. "New chat" resets activeId)', () => {
    const { rerender } = render(<ChatInput onSubmit={vi.fn()} disabled={false} focusKey="conv-1" />)
    const input = screen.getByLabelText('Search query')

    input.blur()
    expect(input).not.toHaveFocus()

    rerender(<ChatInput onSubmit={vi.fn()} disabled={false} focusKey={null} />)

    expect(input).toHaveFocus()
  })
})
