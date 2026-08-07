export interface DevModeToggleProps {
  devMode: boolean
  onToggle: (next: boolean) => void
}

export default function DevModeToggle({ devMode, onToggle }: DevModeToggleProps) {
  return (
    <label className="inline-flex items-center gap-1.5 text-sm cursor-pointer" style={{ color: 'var(--text-faint)' }}>
      <input type="checkbox" checked={devMode} onChange={(e) => onToggle(e.target.checked)} />
      Dev mode
    </label>
  )
}
