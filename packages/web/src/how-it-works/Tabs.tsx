// packages/web/src/how-it-works/Tabs.tsx
export interface TabOption {
  id: string
  label: string
}

interface TabsProps {
  options: TabOption[]
  active: string
  onChange: (id: string) => void
}

export default function Tabs({ options, active, onChange }: TabsProps) {
  return (
    <div role="tablist" className="inline-flex gap-1 rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-1">
      {options.map((option) => {
        const selected = option.id === active
        return (
          <button
            key={option.id}
            role="tab"
            type="button"
            aria-selected={selected}
            onClick={() => onChange(option.id)}
            className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors ${
              selected
                ? 'bg-[var(--accent)] text-[var(--accent-ink)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
            }`}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
