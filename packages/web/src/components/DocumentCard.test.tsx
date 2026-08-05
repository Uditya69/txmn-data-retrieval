// src/components/DocumentCard.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DocumentCard from './DocumentCard'
import type { MergedCard } from '../lib/mergeResults'

const baseCard: MergedCard = {
  doc_id: 'd1',
  source: 'es',
  score: 10,
  snippet:
    'A very relevant snippet about capital gains and business losses that is definitely longer than eighty characters in total length.',
}

describe('DocumentCard', () => {
  it('falls back to a truncated snippet title when no citation metadata exists', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={false} />)
    expect(screen.getByRole('heading').textContent?.endsWith('…')).toBe(true)
    expect(screen.queryByText(/Cited/)).not.toBeInTheDocument()
  })

  it('shows the cited badge and a party-name title when citation metadata exists', () => {
    render(
      <DocumentCard
        card={baseCard}
        citedCount={2}
        citation={{ otherinfo: { partyname: [{ name: 'A Ltd' }, { name: 'B Ltd' }] } }}
        relevance={80}
        devMode={false}
      />,
    )
    expect(screen.getByRole('heading')).toHaveTextContent('A Ltd vs. B Ltd')
    expect(screen.getByText('Cited 2')).toBeInTheDocument()
  })

  it('shows the source badge only in dev mode', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={true} />)
    expect(screen.getByText(/ES · score/)).toBeInTheDocument()
  })

  it('hides the source badge outside dev mode', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={false} />)
    expect(screen.queryByText(/ES · score/)).not.toBeInTheDocument()
  })

  it('calls onOpenDocument with the doc_id when the card is clicked', () => {
    const onOpenDocument = vi.fn()
    render(
      <DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={false} onOpenDocument={onOpenDocument} />,
    )
    fireEvent.click(screen.getByRole('heading'))
    expect(onOpenDocument).toHaveBeenCalledWith('d1')
  })
})
