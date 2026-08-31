import { createPortal } from 'react-dom'

export interface ConfirmDialogProps {
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
}

// Same createPortal-overlay pattern as AuthModal.tsx - this frontend's only
// other modal - generalized into a reusable yes/no dialog instead of a raw
// window.confirm() (which can't be styled/themed and blocks the whole tab).
export default function ConfirmDialog({
  title, message, confirmLabel = 'Delete', cancelLabel = 'Cancel', onConfirm, onCancel,
}: ConfirmDialogProps) {
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'oklch(0 0 0 / 0.4)' }}
      onClick={onCancel}
    >
      <div
        className="w-full max-w-sm rounded-xl p-6"
        style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold mb-2" style={{ color: 'var(--text)' }}>
          {title}
        </h2>
        <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
          {message}
        </p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg px-3 py-2 text-sm cursor-pointer"
            style={{ color: 'var(--text-muted)', border: '1px solid var(--border-soft)', background: 'transparent' }}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="rounded-lg px-3 py-2 text-sm font-medium cursor-pointer"
            style={{ background: 'var(--danger)', color: 'white' }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
