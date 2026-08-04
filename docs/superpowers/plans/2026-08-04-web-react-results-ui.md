# Web React Results UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit chat UI in `packages/web` with a React + Vite results-feed UI (Overview answer card + Documents card feed), matching the design in `docs/superpowers/specs/2026-08-04-web-react-results-ui-design.md`.

**Architecture:** Single-page React app, no router/global-state library. One `useSearch` hook owns the `/ws/search` websocket lifecycle; `App.tsx` composes `SearchBar`, `OverviewCard`, and `DocumentsFeed` around it. Two pure modules (`lib/mergeResults.ts`, `lib/citations.ts`) hold all non-trivial logic and carry the bulk of the automated test coverage, per the spec's testing section.

**Tech Stack:** React 18, TypeScript, Vite, Vitest + React Testing Library, plain CSS Modules. Served in production via a multi-stage Docker build (Vite build → static files served by nginx), replacing the Python/Streamlit Dockerfile.

## Global Constraints

(Copied from the approved design spec — every task's work implicitly includes these.)

- No ES/Milvus source labels, raw scores, or collection names shown in the normal UI — only behind the dev-mode toggle.
- ES/Milvus merge is a **presentation merge, not a ranking fusion** — never blend or re-rank by a combined score across the two sources (hard rule from `CLAUDE.md`).
- No type filter tabs (Act/Rule/Circular/Editorial), no tag chips, no sort options beyond "Relevance", no sidebar (Recent/Library/auth) — all out of scope for v1 per the spec.
- Card titles are snippet-derived by default; they upgrade to a real party-name title only when `ai_mode.citations[doc_id]` exists for that card. Do not invent metadata fields that don't exist in the backend response.
- No new backend endpoints or payload fields — this plan is frontend-only. The `/ws/search` contract is used exactly as it exists today.
- "Show detailed reasoning" ships as a static placeholder for v1 (the backend doesn't return `intent_result` today — a real backend change is explicitly deferred, not part of this plan).

---

### Task 1: Remove the Streamlit web app; stop treating `packages/web` as a uv package

**Files:**
- Delete: `packages/web/pyproject.toml`
- Delete: `packages/web/src/web/app.py`
- Delete: `packages/web/src/web/` (now-empty directory)
- Modify: `pyproject.toml:7` (root) — `[tool.uv.workspace]` members
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: nothing code-facing. This task just removes the old Python package so it stops being resolved as a uv workspace member.

- [ ] **Step 1: Delete the old Streamlit app files**

```bash
git rm packages/web/pyproject.toml packages/web/src/web/app.py
rmdir packages/web/src/web packages/web/src 2>/dev/null || true
```

- [ ] **Step 2: Stop globbing `packages/web` as a uv workspace member**

The root `pyproject.toml` currently has:
```toml
[tool.uv.workspace]
members = ["packages/*"]
```

Change it to an explicit list, since `packages/web` will no longer have a `pyproject.toml` (a bare glob over `packages/*` requires every matched directory to be a valid uv package):

```toml
[tool.uv.workspace]
members = ["packages/common", "packages/model-gateway", "packages/retrieval-api"]
```

- [ ] **Step 3: Resync the Python workspace and confirm nothing broke**

Run: `uv sync --all-packages`
Expected: succeeds, regenerates `uv.lock` without a `web` entry.

Run: `uv run pytest -q`
Expected: all existing Python tests still pass (this task touches no Python source, only workspace config).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(web): remove Streamlit app, drop web from the uv workspace

packages/web is being rebuilt as a React+Vite app (see
docs/superpowers/specs/2026-08-04-web-react-results-ui-design.md) and
will no longer be a Python package."
```

---

### Task 2: Scaffold the Vite + React + TypeScript + Vitest project

**Files:**
- Create: `packages/web/package.json`
- Create: `packages/web/tsconfig.json`
- Create: `packages/web/vite.config.ts`
- Create: `packages/web/index.html`
- Create: `packages/web/public/env-config.js`
- Create: `packages/web/src/env.d.ts`
- Create: `packages/web/src/test-setup.ts`
- Create: `packages/web/src/main.tsx`
- Create: `packages/web/src/App.tsx`
- Create: `packages/web/src/App.module.css`
- Test: `packages/web/src/App.test.tsx`

**Interfaces:**
- Produces: a runnable `App` component (placeholder content for now — fully wired in Task 11) and a working `npm run dev`/`build`/`test`/`typecheck` toolchain that every later task builds on.

- [ ] **Step 1: Write `package.json`**

```json
{
  "name": "web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "typecheck": "tsc --noEmit",
    "preview": "vite preview --host 0.0.0.0 --port 8501",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.5.0",
    "@testing-library/react": "^16.0.1",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "jsdom": "^25.0.1",
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "vitest": "^2.1.4"
  }
}
```

- [ ] **Step 2: Install dependencies**

Run: `cd packages/web && npm install`
Expected: `node_modules/` created, `package-lock.json` generated, no errors.

- [ ] **Step 3: Write `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Write `vite.config.ts`**

```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
  },
})
```

- [ ] **Step 5: Write `src/test-setup.ts`**

```typescript
import '@testing-library/jest-dom/vitest'
```

- [ ] **Step 6: Write `src/env.d.ts`**

```typescript
/// <reference types="vite/client" />

export {}

declare global {
  interface Window {
    __ENV__?: { WS_URL?: string }
  }
}
```

- [ ] **Step 7: Write `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Taxmann Retrieval</title>
  </head>
  <body>
    <div id="root"></div>
    <script src="/env-config.js"></script>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 8: Write `public/env-config.js`**

This is the dev-time default. In production it's overwritten at container startup with the real `WS_URL` (Task 12) — this is how runtime configurability is preserved without baking the URL in at build time.

```javascript
window.__ENV__ = { WS_URL: "ws://localhost:8010/ws/search" };
```

- [ ] **Step 9: Write `src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

const container = document.getElementById('root')
if (!container) {
  throw new Error('Root container #root not found')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 10: Write a placeholder `src/App.tsx`**

This gets fully rewritten in Task 11 once `useSearch`, `OverviewCard`, and `DocumentsFeed` exist. For now it just proves the toolchain works end to end.

```tsx
import styles from './App.module.css'

export default function App() {
  return (
    <div className={styles.page}>
      <h1>Taxmann Retrieval</h1>
    </div>
  )
}
```

```css
/* App.module.css */
.page {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px;
  font-family: system-ui, sans-serif;
}
```

- [ ] **Step 11: Write the failing smoke test**

```tsx
// src/App.test.tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('renders the page title', () => {
    render(<App />)
    expect(screen.getByText('Taxmann Retrieval')).toBeInTheDocument()
  })
})
```

- [ ] **Step 12: Run the test suite**

Run: `npm test`
Expected: 1 test file, 1 test, PASS.

- [ ] **Step 13: Run typecheck and build**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run build`
Expected: succeeds, produces `dist/`.

- [ ] **Step 14: Commit**

```bash
git add package.json package-lock.json tsconfig.json vite.config.ts index.html public/env-config.js src/env.d.ts src/test-setup.ts src/main.tsx src/App.tsx src/App.module.css src/App.test.tsx
git commit -m "feat(web): scaffold Vite + React + TypeScript + Vitest project"
```

---

### Task 3: `lib/mergeResults.ts` — ES/Milvus presentation merge

**Files:**
- Create: `packages/web/src/lib/mergeResults.ts`
- Test: `packages/web/src/lib/mergeResults.test.ts`

**Interfaces:**
- Produces: `EsHit`, `MilvusHit`, `MilvusByCollection`, `MergedCard`, `mergeResults(es, milvus): MergedCard[]` — used by Task 5 (`useSearch`'s `InstantResult` type reuses `EsHit`/`MilvusByCollection`) and Task 8 (`DocumentsFeed`).

- [ ] **Step 1: Write the failing tests**

```typescript
// src/lib/mergeResults.test.ts
import { describe, expect, it } from 'vitest'
import { mergeResults } from './mergeResults'

describe('mergeResults', () => {
  it('returns an empty list when both sources are empty', () => {
    expect(mergeResults([], {})).toEqual([])
    expect(mergeResults(null, null)).toEqual([])
    expect(mergeResults(undefined, undefined)).toEqual([])
  })

  it('returns ES-only cards in their own order when Milvus has nothing', () => {
    const es = [
      { doc_id: 'd1', score: 10, snippet: 'first' },
      { doc_id: 'd2', score: 8, snippet: 'second' },
    ]
    expect(mergeResults(es, {})).toEqual([
      { doc_id: 'd1', source: 'es', score: 10, snippet: 'first' },
      { doc_id: 'd2', source: 'es', score: 8, snippet: 'second' },
    ])
  })

  it('returns Milvus-only cards, deduped across collections by best score', () => {
    const milvus = {
      facts: [{ chunk_id: 'd1::facts::0', doc_id: 'd1', text: 'facts chunk', score: 5 }],
      held: [{ chunk_id: 'd1::held::0', doc_id: 'd1', text: 'held chunk', score: 9 }],
      ruling: [{ chunk_id: 'd2::ruling::0', doc_id: 'd2', text: 'ruling chunk', score: 3 }],
    }
    expect(mergeResults([], milvus)).toEqual([
      { doc_id: 'd1', source: 'milvus', collection: 'held', score: 9, snippet: 'held chunk' },
      { doc_id: 'd2', source: 'milvus', collection: 'ruling', score: 3, snippet: 'ruling chunk' },
    ])
  })

  it('puts ES cards first, then Milvus-only cards, without re-ranking either group', () => {
    const es = [{ doc_id: 'd1', score: 1, snippet: 'es hit' }]
    const milvus = {
      facts: [
        { chunk_id: 'd1::facts::0', doc_id: 'd1', text: 'already in es, ignored', score: 999 },
        { chunk_id: 'd2::facts::0', doc_id: 'd2', text: 'milvus only', score: 2 },
      ],
    }
    expect(mergeResults(es, milvus)).toEqual([
      { doc_id: 'd1', source: 'es', score: 1, snippet: 'es hit' },
      { doc_id: 'd2', source: 'milvus', collection: 'facts', score: 2, snippet: 'milvus only' },
    ])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- mergeResults`
Expected: FAIL — `Cannot find module './mergeResults'`.

- [ ] **Step 3: Write the implementation**

```typescript
// src/lib/mergeResults.ts
export interface EsHit {
  doc_id: string
  score: number
  snippet: string
}

export interface MilvusHit {
  chunk_id: string
  doc_id: string
  text: string
  score: number
}

export type MilvusByCollection = Record<string, MilvusHit[]>

export interface MergedCard {
  doc_id: string
  source: 'es' | 'milvus'
  collection?: string
  score: number
  snippet: string
}

/**
 * Presentation merge, not a ranking fusion: ES and Milvus scores live in
 * different spaces and are never blended or re-ranked against each other.
 * ES cards keep their own order; Milvus-only cards (deduped to their best
 * score per doc_id across collections) are appended after, in their own
 * first-seen order.
 */
export function mergeResults(
  es: EsHit[] | null | undefined,
  milvus: MilvusByCollection | null | undefined,
): MergedCard[] {
  const cards: MergedCard[] = []
  const seen = new Set<string>()

  for (const hit of es ?? []) {
    if (seen.has(hit.doc_id)) continue
    seen.add(hit.doc_id)
    cards.push({ doc_id: hit.doc_id, source: 'es', score: hit.score, snippet: hit.snippet })
  }

  const bestMilvusByDocId = new Map<string, MergedCard>()
  for (const [collection, hits] of Object.entries(milvus ?? {})) {
    for (const hit of hits) {
      if (seen.has(hit.doc_id)) continue
      const existing = bestMilvusByDocId.get(hit.doc_id)
      if (!existing || hit.score > existing.score) {
        bestMilvusByDocId.set(hit.doc_id, {
          doc_id: hit.doc_id,
          source: 'milvus',
          collection,
          score: hit.score,
          snippet: hit.text,
        })
      }
    }
  }

  cards.push(...bestMilvusByDocId.values())
  return cards
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- mergeResults`
Expected: 4 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/mergeResults.ts src/lib/mergeResults.test.ts
git commit -m "feat(web): add mergeResults - ES/Milvus presentation merge"
```

---

### Task 4: `lib/citations.ts` — inline citation parsing

**Files:**
- Create: `packages/web/src/lib/citations.ts`
- Test: `packages/web/src/lib/citations.test.ts`

**Interfaces:**
- Produces: `AnswerSegment`, `CitationRef`, `ParsedAnswer`, `parseCitations(answer: string): ParsedAnswer` — used by Task 9 (`OverviewCard`) and Task 8 (`DocumentsFeed`'s per-card cited count).

- [ ] **Step 1: Write the failing tests**

```typescript
// src/lib/citations.test.ts
import { describe, expect, it } from 'vitest'
import { parseCitations } from './citations'

describe('parseCitations', () => {
  it('returns the whole answer as one text segment when there are no citations', () => {
    const result = parseCitations('No citations here.')
    expect(result.segments).toEqual([{ type: 'text', text: 'No citations here.' }])
    expect(result.citations).toEqual([])
  })

  it('assigns the same number to repeated citations of the same doc_id and counts them', () => {
    const result = parseCitations('First [d1]. Second [d1].')
    expect(result.citations).toEqual([{ doc_id: 'd1', number: 1, count: 2 }])
  })

  it('numbers doc_ids in first-appearance order even when a later bracket repeats an earlier one out of order', () => {
    const result = parseCitations('See [d2] and also [d1], then again [d2].')
    expect(result.citations).toEqual([
      { doc_id: 'd2', number: 1, count: 2 },
      { doc_id: 'd1', number: 2, count: 1 },
    ])
  })

  it('handles multiple doc_ids grouped in a single bracket', () => {
    const result = parseCitations('Supported by [d1, d2].')
    expect(result.segments).toEqual([
      { type: 'text', text: 'Supported by ' },
      { type: 'citation', numbers: [1, 2] },
      { type: 'text', text: '.' },
    ])
    expect(result.citations).toEqual([
      { doc_id: 'd1', number: 1, count: 1 },
      { doc_id: 'd2', number: 2, count: 1 },
    ])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- citations`
Expected: FAIL — `Cannot find module './citations'`.

- [ ] **Step 3: Write the implementation**

```typescript
// src/lib/citations.ts
export type AnswerSegment = { type: 'text'; text: string } | { type: 'citation'; numbers: number[] }

export interface CitationRef {
  doc_id: string
  number: number
  count: number
}

export interface ParsedAnswer {
  segments: AnswerSegment[]
  citations: CitationRef[]
}

// Matches [doc_id] or [doc_id, doc_id, ...] - synthesize.py's prompt asks the
// LLM to cite doc_ids this way. \w covers both real numeric doc_ids and
// short test fixtures like "d1".
const CITATION_PATTERN = /\[(\w+(?:\s*,\s*\w+)*)\]/g

export function parseCitations(answer: string): ParsedAnswer {
  const numberByDocId = new Map<string, number>()
  const countByDocId = new Map<string, number>()
  const segments: AnswerSegment[] = []
  let lastIndex = 0

  CITATION_PATTERN.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = CITATION_PATTERN.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', text: answer.slice(lastIndex, match.index) })
    }

    const docIds = match[1].split(',').map((s) => s.trim()).filter(Boolean)
    const numbers = docIds.map((docId) => {
      let number = numberByDocId.get(docId)
      if (number === undefined) {
        number = numberByDocId.size + 1
        numberByDocId.set(docId, number)
      }
      countByDocId.set(docId, (countByDocId.get(docId) ?? 0) + 1)
      return number
    })
    segments.push({ type: 'citation', numbers })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < answer.length) {
    segments.push({ type: 'text', text: answer.slice(lastIndex) })
  }

  const citations = Array.from(numberByDocId.entries())
    .map(([doc_id, number]) => ({ doc_id, number, count: countByDocId.get(doc_id) ?? 0 }))
    .sort((a, b) => a.number - b.number)

  return { segments, citations }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- citations`
Expected: 4 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/citations.ts src/lib/citations.test.ts
git commit -m "feat(web): add parseCitations - [doc_id] bracket to numbered pill mapping"
```

---

### Task 5: `api/useSearch.ts` — websocket hook

**Files:**
- Create: `packages/web/src/api/useSearch.ts`
- Test: `packages/web/src/api/useSearch.test.ts`

**Interfaces:**
- Consumes: `EsHit`, `MilvusByCollection` (from Task 3, `../lib/mergeResults`)
- Produces: `AiModeCitation`, `InstantResult`, `AiModeResult`, `SearchState`, `useSearch(wsUrl: string): SearchState & { search(query: string): void }` — used by Task 11 (`App.tsx`), and the `AiModeCitation`/`InstantResult`/`AiModeResult` types are used by Tasks 7, 8, 9.

- [ ] **Step 1: Write the failing tests**

```typescript
// src/api/useSearch.test.ts
import { describe, expect, it, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSearch } from './useSearch'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  listeners: Record<string, Array<(event: unknown) => void>> = {}
  sent: string[] = []
  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }
  addEventListener(type: string, listener: (event: unknown) => void) {
    ;(this.listeners[type] ??= []).push(listener)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {}
  emit(type: string, event: unknown = {}) {
    for (const listener of this.listeners[type] ?? []) listener(event)
  }
}

beforeEach(() => {
  MockWebSocket.instances = []
  // @ts-expect-error - test double, not a full WebSocket implementation
  global.WebSocket = MockWebSocket
})

describe('useSearch', () => {
  it('sends the query with mode "both" once the socket opens, and stores the instant result', () => {
    const { result } = renderHook(() => useSearch('ws://test'))

    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
    })
    expect(socket.sent).toEqual([JSON.stringify({ query: 'cgst', mode: 'both' })])

    act(() => {
      socket.emit('message', {
        data: JSON.stringify({
          type: 'instant_result',
          es: [{ doc_id: 'd1', score: 1, snippet: 's' }],
          es_error: null,
          milvus: null,
          milvus_error: null,
        }),
      })
    })

    expect(result.current.instant).toEqual({
      es: [{ doc_id: 'd1', score: 1, snippet: 's' }],
      es_error: null,
      milvus: null,
      milvus_error: null,
    })
    expect(result.current.loading).toBe(true)
  })

  it('marks loading false and stores the answer on ai_mode_done', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_done', answer: 'answer text', citations: {} }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: true, answer: 'answer text', citations: {} })
  })

  it('marks loading false and stores the error on ai_mode_error', () => {
    const { result } = renderHook(() => useSearch('ws://test'))
    act(() => {
      result.current.search('cgst')
    })
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.emit('open')
      socket.emit('message', { data: JSON.stringify({ type: 'ai_mode_error', error: 'boom' }) })
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.aiMode).toEqual({ ok: false, error: 'boom' })
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- useSearch`
Expected: FAIL — `Cannot find module './useSearch'`.

- [ ] **Step 3: Write the implementation**

```typescript
// src/api/useSearch.ts
import { useCallback, useRef, useState } from 'react'
import type { EsHit, MilvusByCollection } from '../lib/mergeResults'

export type AiModeCitation = Record<string, unknown>

export interface InstantResult {
  es: EsHit[] | null
  es_error: string | null
  milvus: MilvusByCollection | null
  milvus_error: string | null
}

export type AiModeResult =
  | { ok: true; answer: string; citations: Record<string, AiModeCitation> }
  | { ok: false; error: string }

export interface SearchState {
  /** true from search() until ai_mode_done/ai_mode_error/an error/close arrives - tracks AI Mode specifically, not the Documents feed (that only depends on `instant`). */
  loading: boolean
  instant: InstantResult | null
  aiMode: AiModeResult | null
  wsError: string | null
}

const INITIAL_STATE: SearchState = { loading: false, instant: null, aiMode: null, wsError: null }

export function useSearch(wsUrl: string): SearchState & { search: (query: string) => void } {
  const [state, setState] = useState<SearchState>(INITIAL_STATE)
  const socketRef = useRef<WebSocket | null>(null)

  const search = useCallback(
    (query: string) => {
      socketRef.current?.close()
      setState({ loading: true, instant: null, aiMode: null, wsError: null })

      let socket: WebSocket
      try {
        socket = new WebSocket(wsUrl)
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, wsError: String(err) }))
        return
      }
      socketRef.current = socket

      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ query, mode: 'both' }))
      })

      socket.addEventListener('message', (event) => {
        const message = JSON.parse((event as MessageEvent).data as string)
        if (message.type === 'instant_result') {
          setState((prev) => ({
            ...prev,
            instant: {
              es: message.es ?? null,
              es_error: message.es_error ?? null,
              milvus: message.milvus ?? null,
              milvus_error: message.milvus_error ?? null,
            },
          }))
        } else if (message.type === 'ai_mode_done') {
          setState((prev) => ({
            ...prev,
            loading: false,
            aiMode: { ok: true, answer: message.answer, citations: message.citations ?? {} },
          }))
        } else if (message.type === 'ai_mode_error') {
          setState((prev) => ({ ...prev, loading: false, aiMode: { ok: false, error: message.error } }))
        }
      })

      socket.addEventListener('error', () => {
        setState((prev) => ({ ...prev, loading: false, wsError: 'Connection to the search service failed.' }))
      })

      socket.addEventListener('close', () => {
        setState((prev) => (prev.loading ? { ...prev, loading: false } : prev))
      })
    },
    [wsUrl],
  )

  return { ...state, search }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- useSearch`
Expected: 3 tests, PASS.

- [ ] **Step 5: Run typecheck**

Run: `npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/api/useSearch.ts src/api/useSearch.test.ts
git commit -m "feat(web): add useSearch websocket hook"
```

---

### Task 6: `components/SearchBar.tsx`

**Files:**
- Create: `packages/web/src/components/SearchBar.tsx`
- Create: `packages/web/src/components/SearchBar.module.css`
- Test: `packages/web/src/components/SearchBar.test.tsx`

**Interfaces:**
- Produces: `SearchBarProps { onSearch: (query: string) => void; disabled?: boolean }`, default export `SearchBar` — used by Task 11 (`App.tsx`).

- [ ] **Step 1: Write the failing tests**

```tsx
// src/components/SearchBar.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import SearchBar from './SearchBar'

describe('SearchBar', () => {
  it('calls onSearch with the trimmed query on submit', () => {
    const onSearch = vi.fn()
    render(<SearchBar onSearch={onSearch} />)

    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: '  what is cgst  ' } })
    fireEvent.click(screen.getByText('Search'))

    expect(onSearch).toHaveBeenCalledWith('what is cgst')
  })

  it('does not call onSearch for an empty query', () => {
    const onSearch = vi.fn()
    render(<SearchBar onSearch={onSearch} />)

    fireEvent.click(screen.getByText('Search'))

    expect(onSearch).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- SearchBar`
Expected: FAIL — `Cannot find module './SearchBar'`.

- [ ] **Step 3: Write the implementation**

```tsx
// src/components/SearchBar.tsx
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
```

```css
/* src/components/SearchBar.module.css */
.bar {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
}

.input {
  flex: 1;
  padding: 10px 14px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 6px;
}

.button {
  padding: 10px 20px;
  font-size: 16px;
  border: none;
  border-radius: 6px;
  background: #1a56db;
  color: white;
  cursor: pointer;
}

.button:disabled {
  opacity: 0.6;
  cursor: default;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- SearchBar`
Expected: 2 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/SearchBar.tsx src/components/SearchBar.module.css src/components/SearchBar.test.tsx
git commit -m "feat(web): add SearchBar component"
```

---

### Task 7: `components/DocumentCard.tsx`

**Files:**
- Create: `packages/web/src/components/DocumentCard.tsx`
- Create: `packages/web/src/components/DocumentCard.module.css`
- Test: `packages/web/src/components/DocumentCard.test.tsx`

**Interfaces:**
- Consumes: `MergedCard` (Task 3), `AiModeCitation` (Task 5)
- Produces: `DocumentCardProps { card: MergedCard; citedCount: number; citation?: AiModeCitation; relevance: number; devMode: boolean; highlighted?: boolean }`, default export `DocumentCard` — used by Task 8 (`DocumentsFeed`).

- [ ] **Step 1: Write the failing tests**

```tsx
// src/components/DocumentCard.test.tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import DocumentCard from './DocumentCard'
import type { MergedCard } from '../lib/mergeResults'

const baseCard: MergedCard = {
  doc_id: 'd1',
  source: 'es',
  score: 10,
  snippet:
    'A very relevant snippet about capital gains and business losses that is definitely longer than eighty characters in total length.',
}

describe('DocumentCard', () => {
  it('falls back to a truncated snippet title when no citation metadata exists', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={false} />)
    expect(screen.getByRole('heading').textContent?.endsWith('…')).toBe(true)
    expect(screen.queryByText(/Cited/)).not.toBeInTheDocument()
  })

  it('shows the cited badge and a party-name title when citation metadata exists', () => {
    render(
      <DocumentCard
        card={baseCard}
        citedCount={2}
        citation={{ otherinfo: { partyname: [{ name: 'A Ltd' }, { name: 'B Ltd' }] } }}
        relevance={80}
        devMode={false}
      />,
    )
    expect(screen.getByRole('heading')).toHaveTextContent('A Ltd vs. B Ltd')
    expect(screen.getByText('Cited 2')).toBeInTheDocument()
  })

  it('shows the source badge only in dev mode', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={true} />)
    expect(screen.getByText(/ES · score/)).toBeInTheDocument()
  })

  it('hides the source badge outside dev mode', () => {
    render(<DocumentCard card={baseCard} citedCount={0} relevance={80} devMode={false} />)
    expect(screen.queryByText(/ES · score/)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- DocumentCard`
Expected: FAIL — `Cannot find module './DocumentCard'`.

- [ ] **Step 3: Write the implementation**

```tsx
// src/components/DocumentCard.tsx
import type { MergedCard } from '../lib/mergeResults'
import type { AiModeCitation } from '../api/useSearch'
import styles from './DocumentCard.module.css'

export interface DocumentCardProps {
  card: MergedCard
  citedCount: number
  citation?: AiModeCitation
  relevance: number
  devMode: boolean
  highlighted?: boolean
}

function truncate(text: string, length: number): string {
  if (text.length <= length) return text
  return `${text.slice(0, length).trimEnd()}…`
}

function extractPartyName(citation?: AiModeCitation): string | null {
  // fetch_citations() returns ES's raw nested _source shape (confirmed
  // against packages/common/tests/test_es_client.py), e.g.
  // { otherinfo: { partyname: ... } } - NOT a flat "otherinfo.partyname" key.
  const otherinfo = citation?.otherinfo
  const raw =
    otherinfo && typeof otherinfo === 'object' ? (otherinfo as Record<string, unknown>).partyname : undefined
  if (!raw) return null
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) {
    const names = raw
      .map((entry) => (entry && typeof entry === 'object' && 'name' in entry ? String((entry as { name: unknown }).name) : null))
      .filter((name): name is string => Boolean(name))
    return names.length > 0 ? names.join(' vs. ') : null
  }
  if (typeof raw === 'object' && 'name' in (raw as Record<string, unknown>)) {
    return String((raw as { name: unknown }).name)
  }
  return null
}

export default function DocumentCard({
  card,
  citedCount,
  citation,
  relevance,
  devMode,
  highlighted = false,
}: DocumentCardProps) {
  const partyName = extractPartyName(citation)
  const title = partyName ?? truncate(card.snippet, 80)

  return (
    <li id={`document-${card.doc_id}`} className={highlighted ? `${styles.card} ${styles.highlighted}` : styles.card}>
      <div className={styles.headerRow}>
        <span className={styles.typeBadge}>Case Law</span>
        {citedCount > 0 && <span className={styles.citedBadge}>Cited {citedCount}</span>}
      </div>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.snippet}>{card.snippet}</p>
      <div className={styles.footerRow}>
        <div className={styles.relevance}>
          <div className={styles.relevanceBar} style={{ width: `${relevance}%` }} />
          <span>{relevance}</span>
        </div>
        {devMode && (
          <span className={styles.devBadge}>
            {card.source === 'es' ? 'ES' : `Milvus:${card.collection}`} · score {card.score.toFixed(3)}
          </span>
        )}
      </div>
    </li>
  )
}
```

```css
/* src/components/DocumentCard.module.css */
.card {
  border: 1px solid #e2e2e2;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
  list-style: none;
  transition: box-shadow 0.2s ease;
}

.highlighted {
  box-shadow: 0 0 0 3px #1a56db;
}

.headerRow {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.typeBadge {
  font-size: 12px;
  text-transform: uppercase;
  color: #555;
}

.citedBadge {
  font-size: 12px;
  background: #eef2ff;
  color: #1a56db;
  padding: 2px 8px;
  border-radius: 12px;
}

.title {
  margin: 0 0 8px;
  font-size: 16px;
}

.snippet {
  color: #333;
  font-size: 14px;
  line-height: 1.5;
}

.footerRow {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
  font-size: 12px;
  color: #555;
}

.relevance {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 120px;
}

.relevanceBar {
  height: 4px;
  background: #1a56db;
  border-radius: 2px;
}

.devBadge {
  font-family: monospace;
  color: #a15c00;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- DocumentCard`
Expected: 4 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/DocumentCard.tsx src/components/DocumentCard.module.css src/components/DocumentCard.test.tsx
git commit -m "feat(web): add DocumentCard component"
```

---

### Task 8: `components/DocumentsFeed.tsx`

**Files:**
- Create: `packages/web/src/components/DocumentsFeed.tsx`
- Create: `packages/web/src/components/DocumentsFeed.module.css`
- Test: `packages/web/src/components/DocumentsFeed.test.tsx`

**Interfaces:**
- Consumes: `mergeResults`, `MergedCard` (Task 3); `parseCitations` (Task 4); `InstantResult`, `AiModeResult` (Task 5); `DocumentCard` (Task 7)
- Produces: `DocumentsFeedProps { instant: InstantResult | null; aiMode: AiModeResult | null; devMode: boolean; highlightedDocId: string | null }`, default export `DocumentsFeed` — used by Task 11 (`App.tsx`).

- [ ] **Step 1: Write the failing tests**

```tsx
// src/components/DocumentsFeed.test.tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import DocumentsFeed from './DocumentsFeed'

describe('DocumentsFeed', () => {
  it('shows a placeholder before any search has been made', () => {
    render(<DocumentsFeed instant={null} aiMode={null} devMode={false} highlightedDocId={null} />)
    expect(screen.getByText('Search to see documents.')).toBeInTheDocument()
  })

  it('shows a no-results message when both legs are empty', () => {
    render(
      <DocumentsFeed
        instant={{ es: [], es_error: null, milvus: {}, milvus_error: null }}
        aiMode={null}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('No results found.')).toBeInTheDocument()
  })

  it('renders merged cards with a result count', () => {
    render(
      <DocumentsFeed
        instant={{
          es: [{ doc_id: 'd1', score: 10, snippet: 'ES snippet about capital gains' }],
          es_error: null,
          milvus: { facts: [{ chunk_id: 'd2::facts::0', doc_id: 'd2', text: 'Milvus snippet text', score: 5 }] },
          milvus_error: null,
        }}
        aiMode={null}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText(/ES snippet about capital gains/)).toBeInTheDocument()
    expect(screen.getByText(/Milvus snippet text/)).toBeInTheDocument()
  })

  it('shows a cited badge derived from the AI Mode answer text', () => {
    render(
      <DocumentsFeed
        instant={{
          es: [{ doc_id: 'd1', score: 10, snippet: 'ES snippet' }],
          es_error: null,
          milvus: null,
          milvus_error: null,
        }}
        aiMode={{ ok: true, answer: 'Cited once [d1] and again [d1].', citations: {} }}
        devMode={false}
        highlightedDocId={null}
      />,
    )
    expect(screen.getByText('Cited 2')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- DocumentsFeed`
Expected: FAIL — `Cannot find module './DocumentsFeed'`.

- [ ] **Step 3: Write the implementation**

```tsx
// src/components/DocumentsFeed.tsx
import { useMemo } from 'react'
import { mergeResults, type MergedCard } from '../lib/mergeResults'
import { parseCitations } from '../lib/citations'
import type { InstantResult, AiModeResult } from '../api/useSearch'
import DocumentCard from './DocumentCard'
import styles from './DocumentsFeed.module.css'

export interface DocumentsFeedProps {
  instant: InstantResult | null
  aiMode: AiModeResult | null
  devMode: boolean
  highlightedDocId: string | null
}

function computeRelevance(cards: MergedCard[]): number[] {
  if (cards.length === 0) return []
  const scores = cards.map((c) => c.score)
  const min = Math.min(...scores)
  const max = Math.max(...scores)
  if (max === min) return cards.map(() => 100)
  return scores.map((s) => Math.round(((s - min) / (max - min)) * 100))
}

export default function DocumentsFeed({ instant, aiMode, devMode, highlightedDocId }: DocumentsFeedProps) {
  const cards = useMemo(() => mergeResults(instant?.es, instant?.milvus), [instant])
  const relevance = useMemo(() => computeRelevance(cards), [cards])
  const citationCounts = useMemo(() => {
    if (!aiMode?.ok || !aiMode.answer) return new Map<string, number>()
    return new Map(parseCitations(aiMode.answer).citations.map((c) => [c.doc_id, c.count]))
  }, [aiMode])

  if (instant === null) {
    return <p className={styles.placeholder}>Search to see documents.</p>
  }
  if (cards.length === 0) {
    return <p className={styles.placeholder}>No results found.</p>
  }

  return (
    <section className={styles.feed}>
      <div className={styles.header}>
        <h2>Documents</h2>
        <span className={styles.count}>{cards.length}</span>
        <span className={styles.sort}>Relevance</span>
      </div>
      <ul className={styles.list}>
        {cards.map((card, index) => (
          <DocumentCard
            key={card.doc_id}
            card={card}
            citedCount={citationCounts.get(card.doc_id) ?? 0}
            citation={aiMode?.ok ? aiMode.citations?.[card.doc_id] : undefined}
            relevance={relevance[index]}
            devMode={devMode}
            highlighted={card.doc_id === highlightedDocId}
          />
        ))}
      </ul>
    </section>
  )
}
```

```css
/* src/components/DocumentsFeed.module.css */
.feed {
  margin-top: 24px;
}

.header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.count {
  color: #555;
  font-size: 14px;
}

.sort {
  margin-left: auto;
  font-size: 14px;
  color: #555;
}

.list {
  padding: 0;
  margin: 0;
}

.placeholder {
  color: #777;
  margin-top: 24px;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- DocumentsFeed`
Expected: 4 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/DocumentsFeed.tsx src/components/DocumentsFeed.module.css src/components/DocumentsFeed.test.tsx
git commit -m "feat(web): add DocumentsFeed component"
```

---

### Task 9: `components/OverviewCard.tsx`

**Files:**
- Create: `packages/web/src/components/OverviewCard.tsx`
- Create: `packages/web/src/components/OverviewCard.module.css`
- Test: `packages/web/src/components/OverviewCard.test.tsx`

**Interfaces:**
- Consumes: `parseCitations` (Task 4); `AiModeResult` (Task 5)
- Produces: `OverviewCardProps { aiMode: AiModeResult | null; loading: boolean; onCitationClick: (docId: string) => void }`, default export `OverviewCard` — used by Task 11 (`App.tsx`).

- [ ] **Step 1: Write the failing tests**

```tsx
// src/components/OverviewCard.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import OverviewCard from './OverviewCard'

describe('OverviewCard', () => {
  it('renders nothing before any AI Mode response has arrived and loading is false', () => {
    const { container } = render(<OverviewCard aiMode={null} loading={false} onCitationClick={vi.fn()} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows a loading skeleton while AI Mode is pending', () => {
    render(<OverviewCard aiMode={null} loading={true} onCitationClick={vi.fn()} />)
    expect(screen.getByTestId('overview-loading')).toBeInTheDocument()
  })

  it('renders numbered citation pills and chips, and calls onCitationClick', () => {
    const onCitationClick = vi.fn()
    render(
      <OverviewCard
        aiMode={{ ok: true, answer: 'Yes. [d1] Also see [d1, d2].', citations: {} }}
        loading={false}
        onCitationClick={onCitationClick}
      />,
    )

    expect(screen.getAllByText('1', { selector: 'sup' }).length).toBeGreaterThan(0)
    expect(screen.getAllByText('2', { selector: 'sup' }).length).toBeGreaterThan(0)

    fireEvent.click(screen.getByText('1. d1 (2)'))
    expect(onCitationClick).toHaveBeenCalledWith('d1')
  })

  it('shows an inline error message when AI Mode failed', () => {
    render(<OverviewCard aiMode={{ ok: false, error: 'boom' }} loading={false} onCitationClick={vi.fn()} />)
    expect(screen.getByText(/AI Mode is currently unavailable: boom/)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test -- OverviewCard`
Expected: FAIL — `Cannot find module './OverviewCard'`.

- [ ] **Step 3: Write the implementation**

```tsx
// src/components/OverviewCard.tsx
import { useMemo, useState } from 'react'
import { parseCitations } from '../lib/citations'
import type { AiModeResult } from '../api/useSearch'
import styles from './OverviewCard.module.css'

export interface OverviewCardProps {
  aiMode: AiModeResult | null
  loading: boolean
  onCitationClick: (docId: string) => void
}

export default function OverviewCard({ aiMode, loading, onCitationClick }: OverviewCardProps) {
  const [reasoningOpen, setReasoningOpen] = useState(false)
  const parsed = useMemo(() => {
    if (!aiMode?.ok || !aiMode.answer) return null
    return parseCitations(aiMode.answer)
  }, [aiMode])

  if (loading && !aiMode) {
    return (
      <section className={styles.card} aria-busy="true" data-testid="overview-loading">
        <h2>Overview</h2>
        <div className={styles.skeleton} />
      </section>
    )
  }

  if (!aiMode) {
    return null
  }

  if (!aiMode.ok) {
    return (
      <section className={styles.card}>
        <h2>Overview</h2>
        <p className={styles.error}>AI Mode is currently unavailable: {aiMode.error}</p>
      </section>
    )
  }

  return (
    <section className={styles.card}>
      <h2>Overview</h2>
      <p className={styles.answer}>
        {parsed?.segments.map((segment, index) =>
          segment.type === 'text' ? (
            <span key={index}>{segment.text}</span>
          ) : (
            <span key={index} className={styles.citationGroup}>
              {segment.numbers.map((n, i) => (
                <sup key={i} className={styles.pill}>
                  {n}
                </sup>
              ))}
            </span>
          ),
        )}
      </p>
      {parsed && parsed.citations.length > 0 && (
        <div className={styles.chipRow}>
          {parsed.citations.map((citation) => (
            <button
              key={citation.doc_id}
              type="button"
              className={styles.chip}
              onClick={() => onCitationClick(citation.doc_id)}
            >
              {citation.number}. {citation.doc_id} ({citation.count})
            </button>
          ))}
        </div>
      )}
      <button type="button" className={styles.reasoningToggle} onClick={() => setReasoningOpen((open) => !open)}>
        {reasoningOpen ? 'Hide' : 'Show'} detailed reasoning
      </button>
      {reasoningOpen && <p className={styles.reasoningPlaceholder}>Reasoning trace coming soon.</p>}
    </section>
  )
}
```

```css
/* src/components/OverviewCard.module.css */
.card {
  border: 1px solid #e2e2e2;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 24px;
  background: #fafbff;
}

.skeleton {
  height: 60px;
  border-radius: 6px;
  background: linear-gradient(90deg, #eee 25%, #f5f5f5 50%, #eee 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

.answer {
  line-height: 1.6;
  font-size: 15px;
}

.pill {
  background: #1a56db;
  color: white;
  border-radius: 50%;
  padding: 1px 5px;
  font-size: 10px;
  margin-left: 2px;
}

.chipRow {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.chip {
  border: 1px solid #ccc;
  border-radius: 16px;
  background: white;
  padding: 4px 12px;
  font-size: 12px;
  cursor: pointer;
}

.reasoningToggle {
  margin-top: 12px;
  background: none;
  border: none;
  color: #1a56db;
  cursor: pointer;
  padding: 0;
  font-size: 13px;
}

.reasoningPlaceholder {
  margin-top: 8px;
  color: #777;
  font-size: 13px;
}

.error {
  color: #b91c1c;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npm test -- OverviewCard`
Expected: 4 tests, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/OverviewCard.tsx src/components/OverviewCard.module.css src/components/OverviewCard.test.tsx
git commit -m "feat(web): add OverviewCard component"
```

---

### Task 10: `components/DevModeToggle.tsx`

**Files:**
- Create: `packages/web/src/components/DevModeToggle.tsx`
- Create: `packages/web/src/components/DevModeToggle.module.css`
- Test: `packages/web/src/components/DevModeToggle.test.tsx`

**Interfaces:**
- Produces: `DevModeToggleProps { devMode: boolean; onToggle: (next: boolean) => void }`, default export `DevModeToggle` — used by Task 11 (`App.tsx`).

- [ ] **Step 1: Write the failing test**

```tsx
// src/components/DevModeToggle.test.tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DevModeToggle from './DevModeToggle'

describe('DevModeToggle', () => {
  it('calls onToggle with the new checked value', () => {
    const onToggle = vi.fn()
    render(<DevModeToggle devMode={false} onToggle={onToggle} />)

    fireEvent.click(screen.getByLabelText('Dev mode'))

    expect(onToggle).toHaveBeenCalledWith(true)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- DevModeToggle`
Expected: FAIL — `Cannot find module './DevModeToggle'`.

- [ ] **Step 3: Write the implementation**

```tsx
// src/components/DevModeToggle.tsx
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
```

```css
/* src/components/DevModeToggle.module.css */
.toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #555;
  cursor: pointer;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test -- DevModeToggle`
Expected: 1 test, PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/DevModeToggle.tsx src/components/DevModeToggle.module.css src/components/DevModeToggle.test.tsx
git commit -m "feat(web): add DevModeToggle component"
```

---

### Task 11: Wire everything together in `App.tsx`

**Files:**
- Modify: `packages/web/src/App.tsx` (full rewrite, replacing Task 2's placeholder)
- Modify: `packages/web/src/App.module.css`

**Interfaces:**
- Consumes: `useSearch` (Task 5), `SearchBar` (Task 6), `OverviewCard` (Task 9), `DocumentsFeed` (Task 8), `DevModeToggle` (Task 10)
- Produces: the complete page — nothing further consumes `App` except `main.tsx` (Task 2, unchanged).

- [ ] **Step 1: Rewrite `src/App.tsx`**

```tsx
// src/App.tsx
import { useEffect, useState } from 'react'
import SearchBar from './components/SearchBar'
import OverviewCard from './components/OverviewCard'
import DocumentsFeed from './components/DocumentsFeed'
import DevModeToggle from './components/DevModeToggle'
import { useSearch } from './api/useSearch'
import styles from './App.module.css'

function resolveWsUrl(): string {
  const fromEnv = window.__ENV__?.WS_URL
  return fromEnv && fromEnv.length > 0 ? fromEnv : 'ws://localhost:8010/ws/search'
}

function readDevModeFromUrl(): boolean {
  return new URLSearchParams(window.location.search).get('dev') === '1'
}

export default function App() {
  const wsUrl = resolveWsUrl()
  const { instant, aiMode, loading, wsError, search } = useSearch(wsUrl)
  const [devMode, setDevMode] = useState(readDevModeFromUrl)
  const [highlightedDocId, setHighlightedDocId] = useState<string | null>(null)

  useEffect(() => {
    if (highlightedDocId === null) return
    const timeout = window.setTimeout(() => setHighlightedDocId(null), 2000)
    return () => window.clearTimeout(timeout)
  }, [highlightedDocId])

  function handleCitationClick(docId: string) {
    setHighlightedDocId(docId)
    document.getElementById(`document-${docId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Taxmann Retrieval</h1>
        <DevModeToggle devMode={devMode} onToggle={setDevMode} />
      </header>
      <SearchBar onSearch={search} disabled={loading} />
      {wsError && <p className={styles.wsError}>{wsError}</p>}
      <OverviewCard aiMode={aiMode} loading={loading} onCitationClick={handleCitationClick} />
      <DocumentsFeed instant={instant} aiMode={aiMode} devMode={devMode} highlightedDocId={highlightedDocId} />
    </div>
  )
}
```

- [ ] **Step 2: Update `src/App.module.css`**

```css
/* src/App.module.css */
.page {
  max-width: 800px;
  margin: 0 auto;
  padding: 24px;
  font-family: system-ui, sans-serif;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.wsError {
  color: #b91c1c;
  margin-bottom: 16px;
}
```

- [ ] **Step 3: Run the full test suite**

Run: `npm test`
Expected: every test file from Tasks 2–10 still passes (Task 2's `App.test.tsx` still asserts the "Taxmann Retrieval" heading text, which is unchanged by this rewrite).

- [ ] **Step 4: Run typecheck and build**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run build`
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.module.css
git commit -m "feat(web): wire SearchBar, OverviewCard, DocumentsFeed, and DevModeToggle into App"
```

---

### Task 12: Docker + docker-compose wiring, and full live verification

**Files:**
- Create: `packages/web/docker-entrypoint.sh`
- Modify (full rewrite): `packages/web/Dockerfile`
- Modify: `docker-compose.yml:20` (`web` service `ports`)

**Interfaces:** none — this task is infrastructure/config only.

- [ ] **Step 1: Write `docker-entrypoint.sh`**

This regenerates `env-config.js` from the container's `WS_URL` env var at startup, so `WS_URL` stays configurable via `docker-compose.yml` exactly like it was for the old Streamlit app (which read `os.environ.get("WS_URL", ...)` at runtime). The official `nginx` image auto-runs every executable `*.sh` in `/docker-entrypoint.d/` before starting nginx, so no custom `ENTRYPOINT`/`CMD` override is needed.

```sh
#!/bin/sh
set -eu
cat > /usr/share/nginx/html/env-config.js <<EOF
window.__ENV__ = { WS_URL: "${WS_URL:-ws://localhost:8010/ws/search}" };
EOF
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x packages/web/docker-entrypoint.sh`

- [ ] **Step 3: Rewrite `packages/web/Dockerfile`**

```dockerfile
FROM node:22-slim AS build
WORKDIR /app
COPY packages/web/package.json packages/web/package-lock.json ./
RUN npm ci
COPY packages/web .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY packages/web/docker-entrypoint.sh /docker-entrypoint.d/40-env-config.sh
RUN chmod +x /docker-entrypoint.d/40-env-config.sh
EXPOSE 80
```

- [ ] **Step 4: Update `docker-compose.yml`'s `web` service port mapping**

nginx listens on port 80 inside the container (its default config), not 8501 - only the host-side mapping changes:

```yaml
  web:
    build:
      context: .
      dockerfile: packages/web/Dockerfile
    ports: ["8501:80"]
    environment:
      # The websocket connection is made client-side, from the user's
      # browser on the host - not server-side inside the compose network
      # (unlike the old Streamlit app, which made this same connection
      # from Python running inside the container). Must be the host-mapped
      # port, not the internal service hostname.
      WS_URL: ws://localhost:8010/ws/search
    depends_on: [retrieval-api]
```

**Corrected during live verification:** this step originally specified `WS_URL: ws://retrieval-api:8000/ws/search` (the internal Docker-network hostname), copied from the old Streamlit setup without accounting for *where* the websocket connection actually originates. The old Streamlit app was a Python server process running inside the container, so the internal hostname resolved correctly. The new React app only ships JS to the browser - the actual `new WebSocket(...)` call happens on the user's host machine, where `retrieval-api` is not a resolvable hostname. Live browser testing caught this immediately ("Connection to the search service failed."); fixed to the host-mapped port above.

- [ ] **Step 5: Rebuild and start the stack**

Run: `docker compose up -d --build`
Expected: all three services (`model-gateway`, `retrieval-api`, `web`) build and start without errors.

- [ ] **Step 6: Verify the container serves the page and picked up the real `WS_URL`**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501`
Expected: `200`

Run: `curl -s http://localhost:8501/env-config.js`
Expected: `window.__ENV__ = { WS_URL: "ws://localhost:8010/ws/search" };` (the real container env value, not the `public/env-config.js` dev default)

- [ ] **Step 7: Live browser verification**

Using the Claude in Chrome browser tools: navigate to `http://localhost:8501`, type a real query (e.g. "what is cgst") into the search bar, submit, and confirm:
- The Documents feed populates with real card content (not a placeholder)
- The Overview card shows either a real AI Mode answer with numbered citation pills, or a clean inline error if AI Mode is unavailable
- Clicking a citation chip scrolls the matching card into view and highlights it
- Toggling Dev Mode (`?dev=1` in the URL, or the header toggle) reveals ES/Milvus source badges on the cards

Take a screenshot at each key state (initial page, after search, dev mode on) to confirm visually before moving on.

- [ ] **Step 8: Commit**

```bash
git add packages/web/docker-entrypoint.sh packages/web/Dockerfile docker-compose.yml
git commit -m "feat(web): serve the built React app via nginx in Docker

Replaces the Python/Streamlit Dockerfile with a multi-stage Node build
+ nginx static serve. WS_URL stays runtime-configurable via an
entrypoint script that regenerates env-config.js from the container's
env var at startup, matching the old Streamlit app's os.environ.get()
behavior."
```
