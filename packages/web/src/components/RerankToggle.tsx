export interface RerankToggleProps {
  label: string
  checked: boolean
  onToggle: (next: boolean) => void
}

export default function RerankToggle({ label, checked, onToggle }: RerankToggleProps) {
  return (
    <label className="inline-flex items-center gap-1.5 text-sm cursor-pointer" style={{ color: 'var(--text-faint)' }}>
      <input type="checkbox" checked={checked} onChange={(e) => onToggle(e.target.checked)} />
      {label}
    </label>
  )
}
