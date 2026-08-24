import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ChatInput from './ChatInput'

describe('ChatInput', () => {
  it('autofocuses the search input on mount, so no click is needed before typing', () => {
    render(<ChatInput onSubmit={vi.fn()} disabled={false} />)

    expect(screen.getByLabelText('Search query')).toHaveFocus()
  })
})
