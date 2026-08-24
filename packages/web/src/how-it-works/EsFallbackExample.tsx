// packages/web/src/how-it-works/EsFallbackExample.tsx

// Real run against this deployment: "acts" intent routes to exactly one collection
// (act_section), which has no native Milvus sparse index at all - every sparse-side hit for
// a query like this comes from Elasticsearch, not just a top-up alongside Milvus.
const QUERY = 'section 54F exemption capital gains investment in residential house'

export default function EsFallbackExample() {
  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--surface-raised)] px-6 py-5">
      <h3 className="text-sm font-semibold text-[var(--text)]">A closer look: when Elasticsearch fills in for Milvus</h3>
      <p className="mt-1 text-sm text-[var(--text-muted)] max-w-xl">
        In the walkthrough above, ES fallback was a small side contributor (20 of 626 candidates) because most of
        the routed collections have their own native sparse index. An <code className="text-[var(--text)]">acts</code>{' '}
        query routes to just one collection — <code className="text-[var(--text)]">act_section</code> — which has{' '}
        <em>no</em> native sparse index at all, so here Elasticsearch is carrying the entire sparse side by itself.
      </p>

      <pre
        className="mt-3 whitespace-pre-wrap rounded-md border border-[var(--border-soft)] bg-[var(--surface-hover)] px-3 py-2 text-xs leading-relaxed text-[var(--text)]"
        style={{ fontFamily: 'var(--font-mono)' }}
      >
        {'query: "' + QUERY + '"\n' +
          "intent: [\"acts\"]  →  routed collections: [\"act_section\"]\n\n" +
          '70 candidates fetched: 50 dense (Milvus) + 20 sparse (100% Elasticsearch —\n' +
          'act_section has no Milvus sparse_vector Function to query)\n\n' +
          'top 2 by RRF rank, from a real run:\n' +
          '  #1 [102120000000042963] milvus_dense  — "The following new section 54F\n' +
          '      shall be inserted by the Finance Act, 1982... Capital gain on\n' +
          '      transfer of certain capital assets not to be charged..."\n' +
          '  #2 [102120000000042963] es_fallback   — "...gain on transfer of certain\n' +
          '      capital assets not to be charged in case of investment in\n' +
          '      residential house. 54F. (1) Where..."'}
      </pre>

      <p className="mt-3 text-xs text-[var(--text-faint)] max-w-xl">
        Same document, found independently by two different engines (Milvus's vector search and Elasticsearch's
        keyword search) — a real instance of the two signals reinforcing each other rather than one silently
        substituting for the other.
      </p>
    </div>
  )
}
