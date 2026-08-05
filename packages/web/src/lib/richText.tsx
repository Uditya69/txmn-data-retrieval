// src/lib/richText.tsx
import type { ReactNode } from 'react'
import type { AnswerSegment } from './citations'

/** Groups a flat citations.ts segment list into paragraphs, splitting text
 * segments on blank lines. Citation segments never start a new paragraph -
 * they stay attached to whichever paragraph their surrounding text is in. */
export function groupIntoParagraphs(segments: AnswerSegment[]): AnswerSegment[][] {
  const paragraphs: AnswerSegment[][] = [[]]
  for (const segment of segments) {
    if (segment.type === 'citation') {
      paragraphs[paragraphs.length - 1].push(segment)
      continue
    }
    const pieces = segment.text.split(/\n{2,}/)
    pieces.forEach((piece, i) => {
      if (i > 0) paragraphs.push([])
      if (piece.length > 0) paragraphs[paragraphs.length - 1].push({ type: 'text', text: piece })
    })
  }
  return paragraphs.filter((paragraph) => paragraph.length > 0)
}

export function splitPlainTextIntoParagraphs(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
}

/** Renders **bold** markers within a plain text run; everything else is
 * passed through as-is (the model is instructed not to use other markdown). */
export function renderInlineText(text: string, keyPrefix: string): ReactNode[] {
  return text.split(/(\*\*.+?\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={`${keyPrefix}-${i}`}>{part}</span>
    ),
  )
}
