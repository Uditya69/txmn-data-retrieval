import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Sidebar from './Sidebar'

const conversations = [
  { id: 'conv-1', title: 'first question', messages: [] },
  { id: 'conv-2', title: 'second question', messages: [] },
]

function renderSidebar(overrides: Partial<Parameters<typeof Sidebar>[0]> = {}) {
  return render(
    <Sidebar
      conversations={conversations}
      activeId={null}
      collapsed={false}
      onToggleCollapsed={vi.fn()}
      onSelect={vi.fn()}
      onNewChat={vi.fn()}
      onDelete={vi.fn()}
      {...overrides}
    />,
  )
}

describe('Sidebar', () => {
  it('reveals a conversation\'s delete button only on hover, not always visible', () => {
    renderSidebar()

    // opacity-0 is the "hidden until hovered" state - group-hover:opacity-100
    // is what reveals it. Asserting both here pins the hover-reveal contract
    // itself (not just "an element with this label exists").
    const deleteButton = screen.getByLabelText('Delete "first question"')
    expect(deleteButton.className).toContain('opacity-0')
    expect(deleteButton.className).toContain('group-hover:opacity-100')
  })

  it('clicking delete opens a confirm dialog instead of deleting immediately', () => {
    const onDelete = vi.fn()
    renderSidebar({ onDelete })

    fireEvent.click(screen.getByLabelText('Delete "first question"'))

    expect(onDelete).not.toHaveBeenCalled()
    expect(screen.getByText('Delete "first question"?')).toBeInTheDocument()
  })

  it('calls onDelete only after confirming in the dialog', () => {
    const onDelete = vi.fn()
    renderSidebar({ onDelete })

    fireEvent.click(screen.getByLabelText('Delete "first question"'))
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

    expect(onDelete).toHaveBeenCalledWith('conv-1')
  })

  it('canceling the confirm dialog does not call onDelete', () => {
    const onDelete = vi.fn()
    renderSidebar({ onDelete })

    fireEvent.click(screen.getByLabelText('Delete "first question"'))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(onDelete).not.toHaveBeenCalled()
    expect(screen.queryByText('Delete "first question"?')).not.toBeInTheDocument()
  })

  it('clicking delete does not also select the conversation', () => {
    const onSelect = vi.fn()
    renderSidebar({ onSelect })

    fireEvent.click(screen.getByLabelText('Delete "first question"'))

    expect(onSelect).not.toHaveBeenCalled()
  })

  it('clicking a conversation title still selects it', () => {
    const onSelect = vi.fn()
    renderSidebar({ onSelect })

    fireEvent.click(screen.getByText('first question'))

    expect(onSelect).toHaveBeenCalledWith('conv-1')
  })
})
