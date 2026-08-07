import { useEffect, useState, type ReactNode } from 'react'

interface DocumentLink {
  text: string
  doc_id: string
}

interface DocumentBlock {
  type: 'paragraph'
  text: string
  links: DocumentLink[]
}

export interface DocumentReaderProps {
  docId: string | null
  apiBaseUrl: string
  onClose: () => void
  onOpenDocument: (docId: string) => void
}

function renderBlockText(block: DocumentBlock, onOpenDocument: (docId: string) => void): ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0
  block.links.forEach((link, index) => {
    const foundAt = block.text.indexOf(link.text, cursor)
    if (foundAt === -1) return
    if (foundAt > cursor) nodes.push(block.text.slice(cursor, foundAt))
    nodes.push(
      <button
        key={index}
        type="button"
        className="underline cursor-pointer"
        style={{ color: 'var(--accent)', textDecorationStyle: 'dotted' }}
        onClick={() => onOpenDocument(link.doc_id)}
      >
        {link.text}
      </button>,
    )
    cursor = foundAt + link.text.length
  })
  if (cursor < block.text.length) nodes.push(block.text.slice(cursor))
  return nodes
}

export default function DocumentReader({ docId, apiBaseUrl, onClose, onOpenDocument }: DocumentReaderProps) {
  const [blocks, setBlocks] = useState<DocumentBlock[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (docId === null) return
    setBlocks(null)
    setError(null)
    fetch(`${apiBaseUrl}/documents/${docId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`status ${response.status}`)
        return response.json()
      })
      .then((data) => setBlocks(data.blocks))
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

          {!error && !blocks && (
            <div className="space-y-3 animate-pulse" data-testid="document-reader-loading">
              <div className="h-5 w-3/4 rounded" style={{ background: 'var(--surface-hover)' }} />
              <div className="h-3 w-full rounded mt-6" style={{ background: 'var(--surface-hover)' }} />
              <div className="h-3 w-5/6 rounded" style={{ background: 'var(--surface-hover)' }} />
            </div>
          )}

          {!error &&
            blocks?.map((block, index) => (
              <p key={index} className="mb-4 text-[15px] leading-relaxed" style={{ color: 'var(--text)' }}>
                {renderBlockText(block, onOpenDocument)}
              </p>
            ))}
        </div>
      </div>
    </div>
  )
}
