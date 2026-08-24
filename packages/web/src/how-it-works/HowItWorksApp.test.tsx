// packages/web/src/how-it-works/HowItWorksApp.test.tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import HowItWorksApp from './HowItWorksApp'

describe('HowItWorksApp', () => {
  it('shows both modes side by side by default', () => {
    render(<HowItWorksApp />)
    expect(screen.getByRole('heading', { name: 'How search works' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Instant mode' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'AI Mode' })).toBeInTheDocument()
  })

  it('lists the AI Mode pipeline stages in order', () => {
    render(<HowItWorksApp />)
    expect(screen.getByRole('heading', { name: 'Intent & rewrite' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Fusion (RRF)' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Answer synthesis' })).toBeInTheDocument()
  })

  it('shows the reranker as currently disabled rather than claiming it always runs', () => {
    render(<HowItWorksApp />)
    expect(screen.getByText(/Currently switched OFF in this environment/)).toBeInTheDocument()
  })

  it('surfaces the concurrent retrieval branches for each mode', () => {
    render(<HowItWorksApp />)
    expect(screen.getByText('Elasticsearch')).toBeInTheDocument()
    expect(screen.getByText('Milvus — dense')).toBeInTheDocument()
    expect(screen.getByText('Milvus — sparse')).toBeInTheDocument()
    expect(screen.getByText('Dense search')).toBeInTheDocument()
    expect(screen.getByText('ES sparse fallback')).toBeInTheDocument()
  })

  it('switches to the worked example on tab click', async () => {
    render(<HowItWorksApp />)
    await userEvent.click(screen.getByRole('tab', { name: 'Walk through an example' }))
    expect(screen.getByRole('heading', { name: 'One query, start to finish' })).toBeInTheDocument()
    expect(screen.getAllByText(/Shah Mohanlal Chhotalal/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Every step below is live data/)).toBeInTheDocument()
  })

  it('shows a real ES-fallback-dominant example alongside the main walkthrough', async () => {
    render(<HowItWorksApp />)
    await userEvent.click(screen.getByRole('tab', { name: 'Walk through an example' }))
    expect(
      screen.getByRole('heading', { name: 'A closer look: when Elasticsearch fills in for Milvus' }),
    ).toBeInTheDocument()
    expect(screen.getByText(/100% Elasticsearch/)).toBeInTheDocument()
  })

  it('switches to the persona tab and explains the trust gate', async () => {
    render(<HowItWorksApp />)
    await userEvent.click(screen.getByRole('tab', { name: 'Persona & personalization' }))
    expect(screen.getByRole('heading', { name: 'How personalization builds up' })).toBeInTheDocument()
    expect(screen.getByText(/20\+ recorded queries/)).toBeInTheDocument()
  })
})
