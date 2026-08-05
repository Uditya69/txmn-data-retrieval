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

  it('shows the party name label from citation metadata when available', () => {
    const onCitationClick = vi.fn()
    render(
      <OverviewCard
        aiMode={{
          ok: true,
          answer: 'Yes. [d1] Also see [d1, d2].',
          citations: { d1: { otherinfo: { partyname: 'ACME v. Widget Co' } } },
        }}
        loading={false}
        onCitationClick={onCitationClick}
      />,
    )

    expect(screen.getByText('1. ACME v. Widget Co (2)')).toBeInTheDocument()

    fireEvent.click(screen.getByText('1. ACME v. Widget Co (2)'))
    expect(onCitationClick).toHaveBeenCalledWith('d1')
  })

  it('shows an inline error message when AI Mode failed', () => {
    render(<OverviewCard aiMode={{ ok: false, error: 'boom' }} loading={false} onCitationClick={vi.fn()} />)
    expect(screen.getByText(/AI Mode is currently unavailable: boom/)).toBeInTheDocument()
  })

  it('hides the reasoning toggle when no reasoning trace is available', () => {
    render(<OverviewCard aiMode={{ ok: true, answer: 'Yes.', citations: {} }} loading={false} onCitationClick={vi.fn()} />)
    expect(screen.queryByText(/detailed reasoning/)).not.toBeInTheDocument()
  })

  it('reveals the reasoning trace on toggle when the backend provides one', () => {
    render(
      <OverviewCard
        aiMode={{ ok: true, answer: 'Yes.', citations: {}, reasoning: 'First step.\n\nSecond step.' }}
        loading={false}
        onCitationClick={vi.fn()}
      />,
    )

    expect(screen.queryByText('First step.')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Show detailed reasoning'))
    expect(screen.getByText('First step.')).toBeInTheDocument()
    expect(screen.getByText('Second step.')).toBeInTheDocument()
  })

  it('renders paragraph breaks and bold case names in the answer', () => {
    const { container } = render(
      <OverviewCard
        aiMode={{ ok: true, answer: '**Case A** [d1] held X.\n\n**Case B** [d2] held Y.', citations: {} }}
        loading={false}
        onCitationClick={vi.fn()}
      />,
    )

    expect(screen.getByText('Case A').tagName).toBe('STRONG')
    expect(screen.getByText('Case B').tagName).toBe('STRONG')
    expect(container.querySelectorAll('p.answer, p[class*="answer"]').length).toBe(2)
  })
})
