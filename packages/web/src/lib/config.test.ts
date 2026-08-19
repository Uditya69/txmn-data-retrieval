import { describe, expect, it } from 'vitest'
import { resolveAdminWsUrl, resolveApiBaseUrl } from './config'

describe('resolveApiBaseUrl', () => {
  it('strips the /ws/search suffix and swaps ws for http', () => {
    expect(resolveApiBaseUrl('ws://localhost:8010/ws/search')).toBe('http://localhost:8010')
  })

  it('strips the /ws/agent suffix and swaps ws for http', () => {
    expect(resolveApiBaseUrl('ws://localhost:8010/ws/agent')).toBe('http://localhost:8010')
  })

  it('swaps wss for https', () => {
    expect(resolveApiBaseUrl('wss://example.com/ws/agent')).toBe('https://example.com')
  })
})

describe('resolveAdminWsUrl', () => {
  it('derives the admin WS url from an http api base url', () => {
    expect(resolveAdminWsUrl('http://localhost:8010')).toBe('ws://localhost:8010/ws/admin-eval')
  })

  it('derives the admin WS url from an https api base url', () => {
    expect(resolveAdminWsUrl('https://api.example.com')).toBe('wss://api.example.com/ws/admin-eval')
  })
})
