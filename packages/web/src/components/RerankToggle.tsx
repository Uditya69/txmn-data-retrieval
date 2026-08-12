export interface RerankToggleProps {
  rerank: boolean
  onToggle: (next: boolean) => void
}

export default function RerankToggle({ rerank, onToggle }: RerankToggleProps) {
  return (
    <label className="inline-flex items-center gap-1.5 text-sm cursor-pointer" style={{ color: 'var(--text-faint)' }}>
      <input type="checkbox" checked={rerank} onChange={(e) => onToggle(e.target.checked)} />
      Rerank
    </label>
  )
}
