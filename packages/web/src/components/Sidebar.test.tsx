import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Sidebar from './Sidebar'

const conversations = [
  { id: 'conv-1', title: 'first question', messages: [] },
  { id: 'conv-2', title: 'second question', messages: [] },
]

describe('Sidebar', () => {
  it('calls onDelete with the conversation id when its delete button is clicked', () => {
    const onDelete = vi.fn()
    render(
      <Sidebar
        conversations={conversations}
        activeId={null}
        collapsed={false}
        onToggleCollapsed={vi.fn()}
        onSelect={vi.fn()}
        onNewChat={vi.fn()}
        onDelete={onDelete}
      />,
    )

    fireEvent.click(screen.getByLabelText('Delete "first question"'))

    expect(onDelete).toHaveBeenCalledWith('conv-1')
  })

  it('clicking delete does not also select the conversation', () => {
    const onSelect = vi.fn()
    render(
      <Sidebar
        conversations={conversations}
        activeId={null}
        collapsed={false}
        onToggleCollapsed={vi.fn()}
        onSelect={onSelect}
        onNewChat={vi.fn()}
        onDelete={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByLabelText('Delete "first question"'))

    expect(onSelect).not.toHaveBeenCalled()
  })

  it('clicking a conversation title still selects it', () => {
    const onSelect = vi.fn()
    render(
      <Sidebar
        conversations={conversations}
        activeId={null}
        collapsed={false}
        onToggleCollapsed={vi.fn()}
        onSelect={onSelect}
        onNewChat={vi.fn()}
        onDelete={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText('first question'))

    expect(onSelect).toHaveBeenCalledWith('conv-1')
  })
})
