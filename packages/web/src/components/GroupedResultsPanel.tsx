import type { EsHit } from '../lib/mergeResults'
import type { DocMeta } from '../api/useSearch'
import { highlightMatches } from '../lib/highlight'
import { CardMetaLines } from '../lib/cardMeta'

type Props = {
  groupedEs: Record<string, EsHit[]>
  docMeta: Record<string, DocMeta> | null | undefined
  query: string
  devMode: boolean
  onOpenDocument: (docId: string) => void
}

// The "Income Tax | Acts" style header is display-only decoration, not a second
// bucketing dimension - grouped_es itself only buckets by raw ES group name (see
// common.es_client.raw_search_grouped). Both halves come from doc_meta, already
// fetched for every card's own badge (common.es_client.fetch_doc_categories):
// the type half ("Acts") is deterministic per raw group name (GROUP_DISPLAY_LABELS
// is a static lookup, so every doc in the same bucket maps to the same label - just
// read it off the first doc present in doc_meta); the subject half ("Income Tax")
// is the most common category among that section's docs, since a section's docs
// aren't guaranteed to share one subject the way they share one content type.
function sectionLabel(rawGroupName: string, docs: EsHit[], docMeta: Record<string, DocMeta> | null | undefined): string {
  const metas = docs.map((d) => docMeta?.[d.doc_id]).filter((m): m is DocMeta => Boolean(m))
  const typeLabel = metas.find((m) => m.group)?.group ?? rawGroupName

  const counts = new Map<string, number>()
  for (const m of metas) {
    if (!m.category) continue
    counts.set(m.category, (counts.get(m.category) ?? 0) + 1)
  }
  let subject: string | null = null
  let best = 0
  for (const [category, count] of [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    if (count > best) {
      best = count
      subject = category
    }
  }

  return subject ? `${subject} | ${typeLabel}` : typeLabel
}

export default function GroupedResultsPanel({ groupedEs, docMeta, query, devMode, onOpenDocument }: Props) {
  const sections = Object.entries(groupedEs).filter(([, docs]) => docs.length > 0)

  if (sections.length === 0) {
    return <p className="text-sm" style={{ color: 'var(--text-faint)' }}>No matches.</p>
  }

  return (
    <div className="flex flex-col gap-4">
      {sections.map(([groupName, docs]) => (
        <div key={groupName} className="flex flex-col gap-2">
          <div
            className="text-xs font-medium uppercase tracking-wider px-1"
            style={{ color: 'var(--accent)' }}
          >
            {sectionLabel(groupName, docs, docMeta)}
          </div>
          {docs.map((doc) => (
            <button
              key={doc.doc_id}
              onClick={() => onOpenDocument(doc.doc_id)}
              className="w-full text-left rounded-lg p-3 transition-colors duration-150 cursor-pointer"
              style={{ background: 'var(--surface)', border: '1px solid var(--border-soft)' }}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>
                  {doc.heading ? highlightMatches(doc.heading, query) : doc.doc_id}
                </span>
                {devMode && (
                  <span className="text-xs shrink-0 font-mono" style={{ color: 'var(--text-faint)' }}>
                    {doc.score.toFixed(3)}
                  </span>
                )}
              </div>
              {doc.subheading && (
                <p className="text-sm mt-1 line-clamp-2" style={{ color: 'var(--text-muted)' }}>
                  {highlightMatches(doc.subheading, query)}
                </p>
              )}
              {(docMeta?.[doc.doc_id]?.act_name ??
                docMeta?.[doc.doc_id]?.tariff_name ??
                docMeta?.[doc.doc_id]?.commentary_topic) && (
                <p className="text-xs mt-1 truncate" style={{ color: 'var(--text-muted)' }}>
                  {docMeta?.[doc.doc_id]?.act_name ??
                    docMeta?.[doc.doc_id]?.tariff_name ??
                    docMeta?.[doc.doc_id]?.commentary_topic}
                </p>
              )}
              {docMeta?.[doc.doc_id]?.is_unreported && (
                <span
                  className="inline-block text-xs font-medium mt-1 px-1.5 py-0.5 rounded"
                  style={{ background: 'var(--surface-raised)', color: 'var(--text-muted)' }}
                >
                  Unreported
                </span>
              )}
              <span className="text-xs font-mono mt-1 block truncate" style={{ color: 'var(--text-faint)' }}>
                {doc.doc_id}
              </span>
              {devMode && <CardMetaLines meta={docMeta?.[doc.doc_id]} />}
            </button>
          ))}
        </div>
      ))}
    </div>
  )
}
