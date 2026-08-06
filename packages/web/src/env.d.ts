/// <reference types="vite/client" />

export {}

declare global {
  interface Window {
    __ENV__?: { WS_URL?: string; AGENT_WS_URL?: string }
  }
}
