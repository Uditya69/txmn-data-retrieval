import { useState, type FormEvent } from 'react'
import styles from './SearchBar.module.css'

export interface SearchBarProps {
  onSearch: (query: string) => void
  disabled?: boolean
}

export default function SearchBar({ onSearch, disabled }: SearchBarProps) {
  const [value, setValue] = useState('')

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) return
    onSearch(trimmed)
  }

  return (
    <form className={styles.bar} onSubmit={handleSubmit}>
      <input
        className={styles.input}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask a legal/tax question..."
        aria-label="Search query"
      />
      <button className={styles.button} type="submit" disabled={disabled}>
        Search
      </button>
    </form>
  )
}
