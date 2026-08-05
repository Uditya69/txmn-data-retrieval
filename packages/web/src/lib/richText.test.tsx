import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import { groupIntoParagraphs, renderInlineText, splitPlainTextIntoParagraphs } from './richText'
import { parseCitations } from './citations'

describe('groupIntoParagraphs', () => {
  it('keeps a single paragraph when there are no blank lines', () => {
    const { segments } = parseCitations('One [d1] sentence.')
    expect(groupIntoParagraphs(segments)).toEqual([segments])
  })

  it('splits into separate paragraphs on blank lines, keeping citations attached', () => {
    const { segments } = parseCitations('First para [d1].\n\nSecond para [d2].')

    const paragraphs = groupIntoParagraphs(segments)

    expect(paragraphs).toHaveLength(2)
    expect(paragraphs[0]).toEqual([{ type: 'text', text: 'First para ' }, { type: 'citation', numbers: [1] }, { type: 'text', text: '.' }])
    expect(paragraphs[1]).toEqual([{ type: 'text', text: 'Second para ' }, { type: 'citation', numbers: [2] }, { type: 'text', text: '.' }])
  })
})

describe('splitPlainTextIntoParagraphs', () => {
  it('splits on blank lines and trims each paragraph', () => {
    expect(splitPlainTextIntoParagraphs('  First.  \n\nSecond.\n\n\nThird.')).toEqual(['First.', 'Second.', 'Third.'])
  })
})

describe('renderInlineText', () => {
  it('renders **bold** markers as <strong> and leaves the rest as plain text', () => {
    const { container } = render(<>{renderInlineText('Plain **bold** plain.', 'k')}</>)
    expect(container.querySelector('strong')?.textContent).toBe('bold')
    expect(container.textContent).toBe('Plain bold plain.')
  })
})
