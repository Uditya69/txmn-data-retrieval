import type { InstantResult, AiModeResult, TraceStep } from './api/useSearch'
import type { AgentResult } from './api/useAgentSearch'

export type ChatMode = 'classic' | 'agent'

export type ResultState = {
  status: 'loading' | 'done' | 'error'
  instant?: InstantResult | null
  aiMode?: AiModeResult | null
  agent?: AgentResult | null
  traceSteps: TraceStep[]
}

export type ChatMessage =
  | { id: string; role: 'user'; text: string }
  | {
      id: string
      role: 'assistant'
      question: string
      activeMode: ChatMode
      results: Partial<Record<ChatMode, ResultState>>
    }

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
}
