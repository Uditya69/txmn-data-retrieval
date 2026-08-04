import { describe, expect, it } from 'vitest'
import { parseCitations } from './citations'

describe('parseCitations', () => {
  it('returns the whole answer as one text segment when there are no citations', () => {
    const result = parseCitations('No citations here.')
    expect(result.segments).toEqual([{ type: 'text', text: 'No citations here.' }])
    expect(result.citations).toEqual([])
  })

  it('assigns the same number to repeated citations of the same doc_id and counts them', () => {
    const result = parseCitations('First [d1]. Second [d1].')
    expect(result.citations).toEqual([{ doc_id: 'd1', number: 1, count: 2 }])
  })

  it('numbers doc_ids in first-appearance order even when a later bracket repeats an earlier one out of order', () => {
    const result = parseCitations('See [d2] and also [d1], then again [d2].')
    expect(result.citations).toEqual([
      { doc_id: 'd2', number: 1, count: 2 },
      { doc_id: 'd1', number: 2, count: 1 },
    ])
  })

  it('handles multiple doc_ids grouped in a single bracket', () => {
    const result = parseCitations('Supported by [d1, d2].')
    expect(result.segments).toEqual([
      { type: 'text', text: 'Supported by ' },
      { type: 'citation', numbers: [1, 2] },
      { type: 'text', text: '.' },
    ])
    expect(result.citations).toEqual([
      { doc_id: 'd1', number: 1, count: 1 },
      { doc_id: 'd2', number: 2, count: 1 },
    ])
  })
})
