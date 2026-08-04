// src/components/OverviewCard.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import OverviewCard from './OverviewCard'

describe('OverviewCard', () => {
  it('renders nothing before any AI Mode response has arrived and loading is false', () => {
    const { container } = render(<OverviewCard aiMode={null} loading={false} onCitationClick={vi.fn()} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows a loading skeleton while AI Mode is pending', () => {
    render(<OverviewCard aiMode={null} loading={true} onCitationClick={vi.fn()} />)
    expect(screen.getByTestId('overview-loading')).toBeInTheDocument()
  })

  it('renders numbered citation pills and chips, and calls onCitationClick', () => {
    const onCitationClick = vi.fn()
    render(
      <OverviewCard
        aiMode={{ ok: true, answer: 'Yes. [d1] Also see [d1, d2].', citations: {} }}
        loading={false}
        onCitationClick={onCitationClick}
      />,
    )

    expect(screen.getAllByText('1', { selector: 'sup' }).length).toBeGreaterThan(0)
    expect(screen.getAllByText('2', { selector: 'sup' }).length).toBeGreaterThan(0)

    fireEvent.click(screen.getByText('1. d1 (2)'))
    expect(onCitationClick).toHaveBeenCalledWith('d1')
  })

  it('shows an inline error message when AI Mode failed', () => {
    render(<OverviewCard aiMode={{ ok: false, error: 'boom' }} loading={false} onCitationClick={vi.fn()} />)
    expect(screen.getByText(/AI Mode is currently unavailable: boom/)).toBeInTheDocument()
  })
})
