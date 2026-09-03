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

  // Regression test: narrow-screen auto-collapse used to be a JS window.innerWidth check
  // read once at mount, so shrinking the browser after load never collapsed the sidebar -
  // nothing was listening for the resize. It's pure CSS now (Tailwind's `max-md:`
  // breakpoint), which reacts to a live resize automatically; these tests pin the
  // classes that make that true rather than simulating an actual resize event (jsdom has
  // no real viewport to resize, and a media-query CSS rule can't be evaluated by jsdom
  // anyway - the browser applies it, not JS).
  describe('narrow-screen auto-collapse (CSS-driven, not JS width tracking)', () => {
    it('collapsed=false: full panel is CSS-visible at lg+ and CSS-hidden below lg, rail is the reverse', () => {
      const { container } = renderSidebar({ collapsed: false })
      const [rail, full] = container.querySelectorAll(':scope > div')

      expect(rail.className).toContain('hidden')
      expect(rail.className).toContain('max-lg:flex')
      expect(full.className).toContain('flex')
      expect(full.className).toContain('max-lg:hidden')
    })

    it('collapsed=true: rail is always visible and full is always hidden, regardless of screen size', () => {
      const { container } = renderSidebar({ collapsed: true })
      const [rail, full] = container.querySelectorAll(':scope > div')

      expect(rail.className).toContain('flex')
      expect(rail.className).not.toContain('max-lg:')
      expect(full.className).toContain('hidden')
      expect(full.className).not.toContain('max-lg:')
    })
  })
})
