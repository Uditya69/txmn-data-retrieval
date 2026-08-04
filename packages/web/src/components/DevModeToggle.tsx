import styles from './DevModeToggle.module.css'

export interface DevModeToggleProps {
  devMode: boolean
  onToggle: (next: boolean) => void
}

export default function DevModeToggle({ devMode, onToggle }: DevModeToggleProps) {
  return (
    <label className={styles.toggle}>
      <input type="checkbox" checked={devMode} onChange={(e) => onToggle(e.target.checked)} />
      Dev mode
    </label>
  )
}
