import { describe, expect, it } from 'vitest'
import { mergeResults } from './mergeResults'

describe('mergeResults', () => {
  it('returns an empty list when both sources are empty', () => {
    expect(mergeResults([], {})).toEqual([])
    expect(mergeResults(null, null)).toEqual([])
    expect(mergeResults(undefined, undefined)).toEqual([])
  })

  it('returns ES-only cards in their own order when Milvus has nothing', () => {
    const es = [
      { doc_id: 'd1', score: 10, snippet: 'first' },
      { doc_id: 'd2', score: 8, snippet: 'second' },
    ]
    expect(mergeResults(es, {})).toEqual([
      { doc_id: 'd1', source: 'es', score: 10, snippet: 'first' },
      { doc_id: 'd2', source: 'es', score: 8, snippet: 'second' },
    ])
  })

  it('returns Milvus-only cards, deduped across collections by best score', () => {
    const milvus = {
      facts: [{ chunk_id: 'd1::facts::0', doc_id: 'd1', text: 'facts chunk', score: 5 }],
      held: [{ chunk_id: 'd1::held::0', doc_id: 'd1', text: 'held chunk', score: 9 }],
      ruling: [{ chunk_id: 'd2::ruling::0', doc_id: 'd2', text: 'ruling chunk', score: 3 }],
    }
    expect(mergeResults([], milvus)).toEqual([
      { doc_id: 'd1', source: 'milvus', collection: 'held', score: 9, snippet: 'held chunk' },
      { doc_id: 'd2', source: 'milvus', collection: 'ruling', score: 3, snippet: 'ruling chunk' },
    ])
  })

  it('puts ES cards first, then Milvus-only cards, without re-ranking either group', () => {
    const es = [{ doc_id: 'd1', score: 1, snippet: 'es hit' }]
    const milvus = {
      facts: [
        { chunk_id: 'd1::facts::0', doc_id: 'd1', text: 'already in es, ignored', score: 999 },
        { chunk_id: 'd2::facts::0', doc_id: 'd2', text: 'milvus only', score: 2 },
      ],
    }
    expect(mergeResults(es, milvus)).toEqual([
      { doc_id: 'd1', source: 'es', score: 1, snippet: 'es hit' },
      { doc_id: 'd2', source: 'milvus', collection: 'facts', score: 2, snippet: 'milvus only' },
    ])
  })
})
