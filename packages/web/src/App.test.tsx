import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import App, { persistConversations, toPersistable } from './App'
import { useSearch } from './api/useSearch'
import { useAgentSearch } from './api/useAgentSearch'
import type { Conversation } from './types'

vi.mock('./api/useSearch', () => ({ useSearch: vi.fn() }))
vi.mock('./api/useAgentSearch', () => ({ useAgentSearch: vi.fn() }))

function baseSearchState() {
  return { loading: false, instant: null, aiMode: null, traceSteps: [], wsError: null, search: vi.fn() }
}

function baseAgentState() {
  return { loading: false, traceSteps: [], result: null, wsError: null, search: vi.fn() }
}

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.mocked(useSearch).mockReturnValue(baseSearchState())
    vi.mocked(useAgentSearch).mockReturnValue(baseAgentState())
  })

  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })

  it('renders the Classic/Agent mode toggle', () => {
    render(<App />)
    expect(screen.getByText('classic')).toBeInTheDocument()
    expect(screen.getByText('agent')).toBeInTheDocument()
  })

  it('submits a question via the chat input and triggers classic search', () => {
    const search = vi.fn()
    vi.mocked(useSearch).mockReturnValue({ ...baseSearchState(), search })
    render(<App />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'what is section 80HH' } })
    fireEvent.click(screen.getByText('Send'))

    expect(search).toHaveBeenCalledWith('what is section 80HH', true, 'both')
    expect(screen.getAllByText('what is section 80HH').length).toBeGreaterThan(0)
  })
})

describe('toPersistable', () => {
  it('strips traceSteps from every result but keeps everything else', () => {
    const conversations: Conversation[] = [
      {
        id: 'conv-1',
        title: 'q',
        messages: [
          { id: 'm1', role: 'user', text: 'q' },
          {
            id: 'm2',
            role: 'assistant',
            question: 'q',
            activeMode: 'classic',
            results: {
              classic: {
                status: 'done',
                instant: { es: [], es_error: null, milvus: null, milvus_sparse: null, milvus_error: null },
                traceSteps: [{ step: 'synthesis_prompt', data: { prompt: 'x'.repeat(10_000) } }],
              },
            },
          },
        ],
      },
    ]

    const result = toPersistable(conversations)

    const persistedResult = result[0].messages[1]
    expect(persistedResult.role).toBe('assistant')
    if (persistedResult.role !== 'assistant') throw new Error('unreachable')
    expect(persistedResult.results.classic?.traceSteps).toEqual([])
    expect(persistedResult.results.classic?.status).toBe('done')
    expect(persistedResult.results.classic?.instant).toEqual({
      es: [], es_error: null, milvus: null, milvus_sparse: null, milvus_error: null,
    })
  })
})

describe('persistConversations', () => {
  const conversation = (id: string): Conversation => ({
    id,
    title: id,
    messages: [{ id: `${id}-m`, role: 'user', text: id }],
  })

  beforeEach(() => {
    localStorage.clear()
  })

  it('writes the stripped payload to localStorage under normal conditions', () => {
    persistConversations([conversation('a'), conversation('b')])
    const stored = JSON.parse(localStorage.getItem('taxmann-retrieval-conversations') ?? '[]')
    expect(stored).toHaveLength(2)
  })

  it('drops the oldest conversations and keeps writing instead of throwing when storage is full', () => {
    const originalSetItem = Storage.prototype.setItem.bind(localStorage)
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')
    let calls = 0
    setItemSpy.mockImplementation((key, value) => {
      calls += 1
      // Fail every call except the very last (smallest) attempt.
      if (calls < 3) throw new DOMException('quota exceeded', 'QuotaExceededError')
      originalSetItem(key, value)
    })

    expect(() => persistConversations([conversation('a'), conversation('b'), conversation('c')])).not.toThrow()
    expect(calls).toBe(3)

    setItemSpy.mockRestore()
  })
})
