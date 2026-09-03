import type { DocMeta } from '../api/useSearch'

/**
 * Renders the optional per-doc metadata lines ported from the reference product's
 * result card (judge/party/date/viewcount/citation/associates) - each line is
 * conditional on that field being present, mirroring the reference product's own
 * per-field *ngIf pattern (a Rule/Act doc simply has no judge/party, same as
 * their card). Not shown: score (dev-mode only, internal) and boost-debug numbers
 * (rendered separately, dev-mode only, by the caller).
 */
export function CardMetaLines({ meta }: { meta: DocMeta | undefined }) {
  if (!meta) return null
  return (
    <>
      {(!!meta.judge?.length || !!meta.party?.length) && (
        <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
          {meta.judge?.length ? `Judge: ${meta.judge.join(', ')}` : null}
          {meta.judge?.length && meta.party?.length ? ' · ' : null}
          {meta.party?.length ? `Party: ${meta.party.join(', ')}` : null}
        </p>
      )}
      {(meta.date || meta.viewcount !== undefined) && (
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
          {meta.date ? meta.date.slice(0, 10) : null}
          {meta.date && meta.viewcount !== undefined ? ' · ' : null}
          {meta.viewcount !== undefined ? `${meta.viewcount} views` : null}
        </p>
      )}
      {meta.fullcitation && (
        <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-faint)' }}>{meta.fullcitation}</p>
      )}
      {(!!meta.referenced_act?.length || !!meta.referenced_section?.length) && (
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
          {[...(meta.referenced_act ?? []), ...(meta.referenced_section ?? [])].join(', ')}
        </p>
      )}
      {meta.cases_referred?.length ? (
        <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-faint)' }}>
          Cases referred: {meta.cases_referred.join('; ')}
        </p>
      ) : null}
    </>
  )
}
