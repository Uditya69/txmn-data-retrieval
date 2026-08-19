// packages/web/src/how-it-works/ExampleWalkthrough.tsx

interface ExampleStage {
  title: string
  caption: string
  payload: string
}

// From the eval set (docs/retrieval-eval-queries.md, Q21) - a "Direct" class query with a
// known gold doc_id, chosen so every step below could be checked against a real pass/fail
// criterion instead of a made-up example. Every payload here came from actually running this
// query end-to-end against this deployment (retrieve() live, then the full AI Mode pipeline
// live once the self-hosted LLM's request timeout was fixed) - none of it is hand-written.
const SAMPLE_QUERY = 'Shah Mohanlal Chhotalal 10 ITC 46 Bombay section 4 bonus shares reduction of capital'
const GOLD_DOC_ID = '101010000000015863'

const STAGES: ExampleStage[] = [
  {
    title: '1. Query received',
    caption: 'A citation-anchored query: a party name, a law-report citation, and a section number, run verbatim against this deployment.',
    payload: SAMPLE_QUERY,
  },
  {
    title: '2. Structural parsing',
    caption: 'Before any model sees the query, plain code (no LLM) picks out citation and section spans and classifies the query\'s shape. This is what gets handed to the SLM as a hint, not guessed at by it.',
    payload:
      'query_shape: "provision"\n' +
      'structural spans: [\n' +
      '  {"text": "10 ITC 46", "type": "citation"},\n' +
      '  {"text": "section 4", "type": "section"}\n' +
      ']',
  },
  {
    title: '3. SLM intent & rewrite',
    caption: 'With those spans as context, the SLM tags this as a caselaws query and passes the query through unchanged — there\'s little to rewrite when the citation itself is already the strongest search signal. It also pulled out a party name and court as filter hints.',
    payload:
      '{\n' +
      '  "search_query": "Shah Mohanlal Chhotalal 10 ITC 46 Bombay section 4 bonus\n' +
      '                    shares reduction of capital",\n' +
      '  "intent": ["caselaws"],\n' +
      '  "filters": {"party": "Shah Mohanlal Chhotalal", "court": "Bombay"}\n' +
      '}',
  },
  {
    title: '4. Collections routed',
    caption: 'With the "caselaws" tag above, 7 of 11 collections are searched — acts, rules, articles, and commentary are skipped.',
    payload: 'case_summary · digest · headnotes · facts · held · ruling · metadata\n(7 of 11 collections)',
  },
  {
    title: '5. Retrieval + fusion',
    caption: 'Dense, sparse, and ES-fallback searches run concurrently, then get merged by rank (RRF). "ruling" has no native sparse index, so its share of the 20 sparse-fallback hits came from Elasticsearch\'s highlighter instead of Milvus.',
    payload:
      '626 candidates fetched (350 dense · 256 native sparse · 20 ES-fallback)\n  →  merged by RRF  →  top 100 kept',
  },
  {
    title: '6. Rerank (off in this environment)',
    caption: 'With the reranker disabled (AI_MODE_RERANK_ENABLED=false), the top 5 by fusion rank are taken directly. The gold case for this query landed at rank #1, and its own chunks fill 3 of the top 5 slots — retrieval is chunk-level, so the same document can appear more than once before synthesis sees it.',
    payload:
      'top 100 (by RRF rank)  →  top 5:\n' +
      '  #1 [101010000000015863] metadata  — "[1936] 10 ITC 46 (Bombay) ... Shah\n' +
      '      Mohanlal Chhotalal vs. Commissioner of Income-tax"\n' +
      '  #2 [101010000000031998] held      — "Act requiring that the distribution...\n' +
      '      entail release of assets of the company..."\n' +
      '  #3 [101010000000015863] digest    — "HEADNOTE: Section 5 of the Income tax\n' +
      '      Act, 1961 [Corresponding to section 4 of the Indian Income-tax Act,\n' +
      '      1922] - Income - Accrual of..."\n' +
      '  #4 [101010000000015863] headnotes — "Section 5 of the Income tax Act,\n' +
      '      1961 [Corresponding to section 4 ...] - Income - Accrual of..."\n' +
      '  #5 [101010000000187753] headnotes — "II. Section 4 of the Income-tax Act,\n' +
      '      1961 - Income - Chargeable as (Bonus shares)..."',
  },
  {
    title: '7. Answer synthesis',
    caption: 'The LLM writes from those passages, citing the doc_id behind every claim. It chose to rest entirely on the gold case rather than pad the answer with the other 4 retrieved chunks — reranking being off didn\'t force it to use all of them.',
    payload:
      'The key takeaway from Shah Mohanlal Chhotalal [101010000000015863] is that\n' +
      'capitalizing reserves followed by their subsequent distribution as bonus\n' +
      'shares isn\'t illusory — it transforms the reserves into capital immediately\n' +
      'upon resolution, and when the company later reduces capital by returning\n' +
      'those shares, the tax authorities cannot reclassify the returned amount as\n' +
      'profits. [...] Crucially, since the District Court had already confirmed the\n' +
      'capital reduction, the Income Tax Department couldn\'t second-guess that\n' +
      'ruling to treat the shares as profits — the value of the shares received by\n' +
      'the assessee was never taxable [101010000000015863].',
  },
]

export default function ExampleWalkthrough() {
  return (
    <div>
      <p className="text-xs text-[var(--text-faint)] max-w-xl">
        Every step below is live data — this exact query run end-to-end against this
        deployment's real Elasticsearch and Milvus indexes and its self-hosted LLM, SLM output
        and final answer included. No synthetic numbers or hand-written prose.
      </p>
      <p className="mt-2 text-xs text-[var(--text-faint)] max-w-xl">
        This query is <code className="text-[var(--text-muted)]">Q21</code> from the retrieval eval set — a known
        case (gold <code className="text-[var(--text-muted)]">doc_id {GOLD_DOC_ID}</code>) with a documented pass
        criterion (must land in the top 5). It does — at rank #1.
      </p>

      <ol className="mt-6 relative flex flex-col">
        {STAGES.map((stage, i) => {
          const isLast = i === STAGES.length - 1
          return (
            <li key={stage.title} className="relative pl-12 pb-8 last:pb-0">
              {!isLast && (
                <span aria-hidden="true" className="absolute left-[15px] top-8 bottom-0 w-px bg-[var(--border)]" />
              )}
              <span className="absolute left-0 top-0 flex h-8 w-8 items-center justify-center rounded-full bg-[var(--accent)] text-xs font-semibold text-[var(--accent-ink)]">
                {i + 1}
              </span>

              <h3 className="text-sm font-semibold text-[var(--text)] pt-1">{stage.title}</h3>
              <p className="mt-1 text-sm text-[var(--text-muted)] max-w-xl">{stage.caption}</p>
              <pre
                className="mt-2 whitespace-pre-wrap rounded-md border border-[var(--border-soft)] bg-[var(--surface-hover)] px-3 py-2 text-xs leading-relaxed text-[var(--text)]"
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                {stage.payload}
              </pre>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
