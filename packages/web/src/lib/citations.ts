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

// knownDocIds is the DB-sourced allowlist (ES/Milvus doc_ids the backend actually
// retrieved for this answer - see AiModeResult.citations / AgentResult.docIds).
// A bracket is only linkified when every token inside it is in that set. This is
// what stops legal citation years - e.g. "[1957] 32 ITR 466 (SC)", standard Indian
// case-report notation - from being mistaken for a [doc_id] reference: "1957" is
// never a real doc_id the backend fetched, so the bracket is left as plain text
// instead of turning into a dead /documents/1957 link. When knownDocIds is omitted,
// no filtering happens (used by callers/tests that don't have a DB set to check against).
export function parseCitations(answer: string, knownDocIds?: Set<string>): ParsedAnswer {
  const numberByDocId = new Map<string, number>()
  const countByDocId = new Map<string, number>()
  const segments: AnswerSegment[] = []
  let lastIndex = 0

  CITATION_PATTERN.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = CITATION_PATTERN.exec(answer)) !== null) {
    const docIds = match[1].split(',').map((s) => s.trim()).filter(Boolean)
    const allKnown = !knownDocIds || docIds.every((docId) => knownDocIds.has(docId))

    if (!allKnown) {
      // Not a real doc_id citation (e.g. a legal citation year) - leave the
      // bracket as literal text rather than linkifying it.
      continue
    }

    if (match.index > lastIndex) {
      segments.push({ type: 'text', text: answer.slice(lastIndex, match.index) })
    }

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
