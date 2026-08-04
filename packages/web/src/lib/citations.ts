export type AnswerSegment = { type: 'text'; text: string } | { type: 'citation'; numbers: number[] }

export interface CitationRef {
  doc_id: string
  number: number
  count: number
}

export interface ParsedAnswer {
  segments: AnswerSegment[]
  citations: CitationRef[]
}

// Matches [doc_id] or [doc_id, doc_id, ...] - synthesize.py's prompt asks the
// LLM to cite doc_ids this way. \w covers both real numeric doc_ids and
// short test fixtures like "d1".
const CITATION_PATTERN = /\[(\w+(?:\s*,\s*\w+)*)\]/g

export function parseCitations(answer: string): ParsedAnswer {
  const numberByDocId = new Map<string, number>()
  const countByDocId = new Map<string, number>()
  const segments: AnswerSegment[] = []
  let lastIndex = 0

  CITATION_PATTERN.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = CITATION_PATTERN.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', text: answer.slice(lastIndex, match.index) })
    }

    const docIds = match[1].split(',').map((s) => s.trim()).filter(Boolean)
    const numbers = docIds.map((docId) => {
      let number = numberByDocId.get(docId)
      if (number === undefined) {
        number = numberByDocId.size + 1
        numberByDocId.set(docId, number)
      }
      countByDocId.set(docId, (countByDocId.get(docId) ?? 0) + 1)
      return number
    })
    segments.push({ type: 'citation', numbers })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < answer.length) {
    segments.push({ type: 'text', text: answer.slice(lastIndex) })
  }

  const citations = Array.from(numberByDocId.entries())
    .map(([doc_id, number]) => ({ doc_id, number, count: countByDocId.get(doc_id) ?? 0 }))
    .sort((a, b) => a.number - b.number)

  return { segments, citations }
}
