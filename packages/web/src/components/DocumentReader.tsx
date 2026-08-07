import { useEffect, useState, type ReactNode } from 'react'

const DOC_FONT = 'Georgia, "Times New Roman", Times, serif'

type Span = { type: 'text'; text: string; bold: boolean; italic: boolean } | { type: 'link'; text: string; doc_id: string }

type DocumentBlock = { type: 'paragraph' | 'fact_label' | 'headnote' | 'counsel'; spans: Span[] }

type FullDocument = {
  doc_id: string
  heading: string | null
  subheading: string | null
  year: string | null
  blocks: DocumentBlock[]
}

export interface DocumentReaderProps {
  docId: string | null
  apiBaseUrl: string
  onClose: () => void
  onOpenDocument: (docId: string) => void
}

function renderSpans(spans: Span[], onOpenDocument: (docId: string) => void): ReactNode[] {
  return spans.map((span, i) => {
    if (span.type === 'link') {
      return (
        <button
          key={i}
          type="button"
          className="underline cursor-pointer"
          style={{ color: 'var(--accent)', textDecorationStyle: 'dotted' }}
          onClick={() => onOpenDocument(span.doc_id)}
        >
          {span.text}
        </button>
      )
    }
    let node: ReactNode = span.text
    if (span.bold) node = <strong key={i}>{node}</strong>
    if (span.italic) node = <em key={i}>{node}</em>
    return <span key={i}>{node}</span>
  })
}

function Block({ block, onOpenDocument }: { block: DocumentBlock; onOpenDocument: (docId: string) => void }) {
  const content = renderSpans(block.spans, onOpenDocument)

  switch (block.type) {
    case 'fact_label':
      return (
        <h3 className="text-sm font-bold uppercase tracking-wide mt-5 mb-2" style={{ color: 'var(--text)', fontFamily: DOC_FONT }}>
          {content}
        </h3>
      )
    case 'headnote':
      return <div className="text-[15px] leading-relaxed whitespace-pre-wrap italic mb-2">{content}</div>
    case 'counsel':
      return (
        <p className="text-sm mb-2" style={{ color: 'var(--text-faint)' }}>
          {content}
        </p>
      )
    default:
      return (
        <p className="mb-4 text-justify text-[15px] leading-relaxed" style={{ textIndent: '1.5em' }}>
          {content}
        </p>
      )
  }
}

export default function DocumentReader({ docId, apiBaseUrl, onClose, onOpenDocument }: DocumentReaderProps) {
  const [doc, setDoc] = useState<FullDocument | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (docId === null) return
    setDoc(null)
    setError(null)
    fetch(`${apiBaseUrl}/documents/${docId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`status ${response.status}`)
        return response.json()
      })
      .then((data) => setDoc(data))
      .catch(() => setError('Could not load document.'))
  }, [docId, apiBaseUrl])

  useEffect(() => {
    if (!docId) return
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [docId, onClose])

  if (docId === null) return null

  return (
    <div className="fixed inset-0 flex justify-end" style={{ background: 'oklch(0.1 0.01 55 / 0.6)', zIndex: 30 }} onClick={onClose}>
      <div
        className="w-full max-w-3xl h-full overflow-y-auto shadow-2xl"
        style={{ background: 'var(--surface-raised)', zIndex: 40 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="max-w-2xl mx-auto px-6 py-5">
          <button
            onClick={onClose}
            className="text-sm mb-4 inline-flex items-center gap-1.5 cursor-pointer"
            style={{ color: 'var(--text-faint)' }}
          >
            ← Close
          </button>

          {error && (
            <p className="text-sm rounded-lg p-3" style={{ color: 'var(--danger)', background: 'var(--danger-soft)' }}>
              {error}
            </p>
          )}

          {!error && !doc && (
            <div className="space-y-3 animate-pulse" data-testid="document-reader-loading">
              <div className="h-5 w-3/4 rounded" style={{ background: 'var(--surface-hover)' }} />
              <div className="h-3 w-1/3 rounded" style={{ background: 'var(--surface-hover)' }} />
              <div className="h-3 w-full rounded mt-6" style={{ background: 'var(--surface-hover)' }} />
              <div className="h-3 w-5/6 rounded" style={{ background: 'var(--surface-hover)' }} />
            </div>
          )}

          {!error && doc && (
            <div style={{ color: 'var(--text)', fontFamily: DOC_FONT }}>
              <h2 className="text-xl font-bold text-center leading-snug">{doc.subheading || doc.heading || doc.doc_id}</h2>
              <div className="flex items-center justify-center gap-2 mt-2 mb-1 text-sm" style={{ color: 'var(--text-faint)' }}>
                {doc.heading && <span>{doc.heading}</span>}
                {doc.year && <span>· {doc.year}</span>}
              </div>
              <span className="block text-right text-xs mb-4" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
                {doc.doc_id}
              </span>

              {doc.blocks.map((block, index) => (
                <Block key={index} block={block} onOpenDocument={onOpenDocument} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
