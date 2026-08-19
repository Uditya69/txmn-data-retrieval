// packages/web/src/how-it-works/HowItWorksApp.tsx
import { useState } from 'react'
import FlowDiagram, { type FlowStage } from './FlowDiagram'
import Tabs from './Tabs'
import ExampleWalkthrough from './ExampleWalkthrough'
import EsFallbackExample from './EsFallbackExample'

const INSTANT_STAGES: FlowStage[] = [
  {
    title: 'Query received',
    detail: 'Sent to both search backends exactly as typed — no rewriting, no LLM call on this path.',
  },
  {
    title: 'Search runs',
    branches: [
      { title: 'Elasticsearch', detail: 'Lexical (keyword) search over the full corpus. Top 20 hits.' },
      { title: 'Milvus — dense', detail: 'Vector similarity search (Voyage embeddings) across all 11 collections.' },
      { title: 'Milvus — sparse', detail: 'Milvus-native BM25 search, same 11 collections, run alongside the dense search.' },
    ],
  },
  {
    title: 'Preview returned',
    detail: 'Each source is trimmed to its natural score drop-off and shown side by side — a raw look at what the indexes hold, not a generated answer.',
  },
]

const AI_MODE_STAGES: FlowStage[] = [
  { title: 'Query received', detail: 'The question as the user typed it.' },
  {
    title: 'Intent & rewrite',
    detail: 'A small language model (SLM) reads the query and produces two things: which document types it’s about (acts, rules, caselaws, articles, commentary), and a cleaned-up search query.',
    note: 'This is what decides which of the 11 collections get searched next — a caselaw question skips acts/rules entirely.',
  },
  {
    title: 'Retrieval runs',
    branches: [
      { title: 'Dense search', detail: 'Voyage embeddings, vector similarity, scoped to the collections the intent pointed at. Up to 50 candidates per collection.' },
      { title: 'Sparse search', detail: 'Milvus-native BM25, same scoped collections, same 50-candidate cap.' },
      {
        title: 'ES sparse fallback',
        detail: '5 collections (ruling, act_section, rule_section, article_section, commentary_section) never got a native sparse index at ingestion. For those, Elasticsearch\'s highlighter picks the single best-matching fragment per document, then trims/centers it to ~1024 tokens with the same tokenizer the real ingestion pipeline uses — a same-size stand-in snippet, not a lookup of a real stored chunk.',
      },
    ],
  },
  {
    title: 'Fusion (RRF)',
    detail: 'All three result sets are merged by rank position, not by comparing their raw scores — lexical and vector scores live on different, incomparable scales, so blending them directly would be meaningless.',
  },
  {
    title: 'Rerank',
    detail: 'A dedicated reranking model can re-score the merged candidates and keep only the strongest ~5 passages before they reach the LLM.',
    note: 'Currently switched OFF in this environment — the top 5 candidates by fusion rank are passed straight to synthesis, no reranker call. It’s a one-line config toggle, kept available for when the eval numbers favor turning it back on.',
    noteTone: 'warning',
  },
  {
    title: 'Answer synthesis',
    detail: 'An LLM writes the answer from those passages. Every claim is cited inline back to the specific ruling it came from — the UI turns each citation into a clickable reference.',
  },
]

const PERSONA_STAGES: FlowStage[] = [
  {
    title: 'Query answered',
    detail: 'After AI Mode answers, a background task (never blocks the response the user sees) records a signal from this query — the same intent categories the routing step above already computed are reused directly.',
  },
  {
    title: 'Signal extracted',
    detail: 'A second, separate SLM call (same model/role as intent extraction, different prompt) classifies the query itself: expertise level (student / practitioner / expert) and query style (broad / precise-citation).',
    note: 'This is an extra model call per query for logged-in users — but it runs in the background after the answer is already sent, so it never adds latency the user feels.',
  },
  {
    title: 'Persona updated',
    detail: 'Stored per logged-in user in MongoDB. Each of the 6 document categories gets a running affinity score; expertise level and query style are updated by majority vote across every query so far.',
    note: 'A tie keeps whatever the persona already had, rather than flipping to an arbitrary new leader — one odd query can\'t swing it.',
  },
  {
    title: 'Used on future queries',
    detail: 'Once a user has 20+ recorded queries, a one-line note (e.g. "frequently asks about caselaws, acts; expertise level: practitioner") is added to that user\'s next intent-classification and answer-synthesis prompts.',
    note: 'The model is explicitly told this is a prior about the user\'s typical usage, not a fact about the current query — if the query itself disagrees, it\'s instructed to ignore the note entirely.',
  },
]

const TAB_OPTIONS = [
  { id: 'compare', label: 'Compare modes' },
  { id: 'example', label: 'Walk through an example' },
  { id: 'persona', label: 'Persona & personalization' },
]

export default function HowItWorksApp() {
  const [tab, setTab] = useState('compare')

  return (
    <div className="min-h-screen bg-[var(--ink)] text-[var(--text)]">
      <header className="max-w-5xl mx-auto px-6 pt-16 pb-8">
        <p className="text-xs font-medium uppercase tracking-wide text-[var(--accent)]">Taxmann Retrieval</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">How search works</h1>
        <p className="mt-3 text-base text-[var(--text-muted)] max-w-2xl">
          Every query takes one of two paths: an <strong className="text-[var(--text)] font-medium">instant</strong>{' '}
          raw preview straight from the search indexes, or an{' '}
          <strong className="text-[var(--text)] font-medium">AI-guided</strong> path that understands the question,
          retrieves narrowly, and writes a cited answer. Both run several searches concurrently rather than one
          after another.
        </p>

        <div className="mt-6">
          <Tabs options={TAB_OPTIONS} active={tab} onChange={setTab} />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 pb-20">
        {tab === 'compare' ? (
          <div className="grid gap-10 lg:grid-cols-2 lg:gap-8">
            <section className="lg:border-r lg:border-[var(--border-soft)] lg:pr-8">
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="text-lg font-semibold">Instant mode</h2>
                <span className="text-xs text-[var(--text-faint)]">No LLM</span>
              </div>
              <p className="mt-1 text-sm text-[var(--text-muted)]">
                A fast, unfiltered look at what the search indexes contain.
              </p>
              <div className="mt-6">
                <FlowDiagram stages={INSTANT_STAGES} compact />
              </div>
            </section>

            <section>
              <div className="flex items-baseline justify-between gap-4">
                <h2 className="text-lg font-semibold">AI Mode</h2>
                <span className="text-xs text-[var(--text-faint)]">6 stages, 2 model calls</span>
              </div>
              <p className="mt-1 text-sm text-[var(--text-muted)]">
                Query understanding, targeted retrieval, and a synthesized, cited answer.
              </p>
              <div className="mt-6">
                <FlowDiagram stages={AI_MODE_STAGES} compact />
              </div>
            </section>
          </div>
        ) : tab === 'example' ? (
          <section className="max-w-3xl">
            <h2 className="text-lg font-semibold">One query, start to finish</h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Following a single AI Mode query through every stage of the pipeline above.
            </p>
            <div className="mt-6">
              <ExampleWalkthrough />
            </div>
            <div className="mt-10">
              <EsFallbackExample />
            </div>
          </section>
        ) : (
          <section className="max-w-3xl">
            <h2 className="text-lg font-semibold">How personalization builds up</h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Guests and new users get zero personalization — this only kicks in for logged-in users with enough
              history, and it always defers to what the current query actually says.
            </p>
            <div className="mt-6">
              <FlowDiagram stages={PERSONA_STAGES} />
            </div>
          </section>
        )}

        <section className="mt-14 rounded-lg border border-[var(--border-soft)] bg-[var(--surface-raised)] px-6 py-5">
          <h2 className="text-sm font-semibold">Under the hood</h2>
          <ul className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-5 text-sm text-[var(--text-muted)]">
            <li>
              <span className="block text-[var(--text)] font-medium">11 specialized collections</span>
              Case summaries, digests, headnotes, facts, holdings, rulings, acts, rules, articles, and commentary
              are indexed separately, not as one bucket — so a query only searches what's relevant to it.
            </li>
            <li>
              <span className="block text-[var(--text)] font-medium">Rank-based fusion only</span>
              Lexical and vector scores are never blended directly. Results are always fused by rank position
              (Reciprocal Rank Fusion), so no single source can silently dominate on scale alone.
            </li>
            <li>
              <span className="block text-[var(--text)] font-medium">Citation-verified answers</span>
              Every sentence AI Mode writes is grounded in a specific ruling, linked inline — never a
              paraphrase with no traceable source.
            </li>
          </ul>
          <p className="mt-4 text-xs text-[var(--text-faint)]">
            Repeat or near-duplicate questions can be served from a semantic cache (cosine-similarity match
            against prior queries) before either path runs. It's invisible to the flow above — it only changes
            how fast a previously-seen question comes back.
          </p>
        </section>
      </main>
    </div>
  )
}
