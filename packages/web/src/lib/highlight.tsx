import { Fragment, type ReactNode } from 'react'

// Words too common/short to be useful as a highlight signal - mirrors the kind of
// stopword trimming query_tokenizer.py does server-side, kept minimal here since this
// is presentation-only (which query words lit up why a card ranked where it did).
const STOPWORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'if', 'that',
  'this', 'it', 'its', 'their', 'they', 'does', 'do', 'did', 'can', 'could',
  'from', 'by', 'as', 'with', 'not', 'so',
])

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function queryTerms(query: string): string[] {
  const words = query.match(/[\p{L}\p{N}]+/gu) ?? []
  const unique = [...new Set(words.map((w) => w.toLowerCase()).filter((w) => w.length >= 3 && !STOPWORDS.has(w)))]
  // Longest first, so e.g. "assessee" isn't shadowed by a shorter overlapping term.
  return unique.sort((a, b) => b.length - a.length)
}

// Highlights every occurrence of a query term inside `text` - same idea as Taxmann's
// own search product (image copy.png): yellow-marking the words that made a result
// match, so it's visible at a glance why a card ranked where it did.
export function highlightMatches(text: string, query: string): ReactNode {
  const terms = queryTerms(query)
  if (terms.length === 0 || !text) return text

  const pattern = new RegExp(`\\b(${terms.map(escapeRegExp).join('|')})\\b`, 'gi')
  const parts = text.split(pattern)
  if (parts.length === 1) return text

  const lowerTerms = new Set(terms)
  return parts.map((part, i) =>
    lowerTerms.has(part.toLowerCase()) ? (
      <mark key={i} style={{ background: '#fde047', color: 'inherit', borderRadius: '2px', padding: '0 1px' }}>
        {part}
      </mark>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  )
}
