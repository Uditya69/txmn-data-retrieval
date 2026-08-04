import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SearchBar from './SearchBar'

describe('SearchBar', () => {
  it('calls onSearch with the trimmed query on submit', () => {
    const onSearch = vi.fn()
    render(<SearchBar onSearch={onSearch} />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: '  what is cgst  ' } })
    fireEvent.click(screen.getByText('Search'))

    expect(onSearch).toHaveBeenCalledWith('what is cgst')
  })

  it('does not call onSearch for an empty query', () => {
    const onSearch = vi.fn()
    render(<SearchBar onSearch={onSearch} />)

    fireEvent.click(screen.getByText('Search'))

    expect(onSearch).not.toHaveBeenCalled()
  })
})
