import { useEffect, useState } from 'react'
import styles from './DocumentModal.module.css'

interface DocumentLink {
  text: string
  doc_id: string
}

interface DocumentBlock {
  type: 'paragraph'
  text: string
  links: DocumentLink[]
}

export interface DocumentModalProps {
  docId: string | null
  apiBaseUrl: string
  onClose: () => void
  onNavigate: (docId: string) => void
}

function renderBlockText(block: DocumentBlock, onNavigate: (docId: string) => void) {
  const nodes: React.ReactNode[] = []
  let cursor = 0
  block.links.forEach((link, index) => {
    const foundAt = block.text.indexOf(link.text, cursor)
    if (foundAt === -1) return
    if (foundAt > cursor) nodes.push(block.text.slice(cursor, foundAt))
    nodes.push(
      <button
        key={index}
        type="button"
        className={styles.link}
        onClick={() => onNavigate(link.doc_id)}
      >
        {link.text}
      </button>,
    )
    cursor = foundAt + link.text.length
  })
  if (cursor < block.text.length) nodes.push(block.text.slice(cursor))
  return nodes
}

export default function DocumentModal({ docId, apiBaseUrl, onClose, onNavigate }: DocumentModalProps) {
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

  if (docId === null) return null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={(event) => event.stopPropagation()}>
        <button type="button" className={styles.closeButton} aria-label="Close" onClick={onClose}>
          ×
        </button>
        {error && <p className={styles.error}>{error}</p>}
        {!error && !blocks && <p data-testid="document-modal-loading">Loading…</p>}
        {!error &&
          blocks?.map((block, index) => (
            <p key={index} className={styles.paragraph}>
              {renderBlockText(block, onNavigate)}
            </p>
          ))}
      </div>
    </div>
  )
}
