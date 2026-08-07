import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unhandled error in app tree', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6" style={{ background: 'var(--ink)' }}>
          <div className="max-w-md text-center" style={{ color: 'var(--text)' }}>
            <p className="text-lg font-semibold mb-2">Something went wrong.</p>
            <p className="text-sm mb-4" style={{ color: 'var(--text-faint)' }}>
              {this.state.error.message}
            </p>
            <button
              type="button"
              onClick={() => this.setState({ error: null })}
              className="text-sm px-4 py-2 rounded-full font-medium cursor-pointer"
              style={{ background: 'var(--surface-raised)', border: '1px solid var(--border-soft)', color: 'var(--text)' }}
            >
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
