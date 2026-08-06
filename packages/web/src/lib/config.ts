// src/lib/config.ts
export function resolveWsUrl(): string {
  const fromEnv = window.__ENV__?.WS_URL
  return fromEnv && fromEnv.length > 0 ? fromEnv : 'ws://localhost:8010/ws/search'
}

export function resolveApiBaseUrl(wsUrl: string): string {
  return wsUrl.replace(/^ws/, 'http').replace(/\/ws\/(search|agent)$/, '')
}

export function resolveAgentWsUrl(): string {
  const fromEnv = window.__ENV__?.AGENT_WS_URL
  return fromEnv && fromEnv.length > 0 ? fromEnv : 'ws://localhost:8010/ws/agent'
}
