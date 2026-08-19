// packages/web/src/how-it-works/FlowDiagram.tsx

/** One branch inside a parallel group - e.g. "Elasticsearch" running alongside "Milvus dense". */
export interface FlowBranch {
  title: string
  detail: string
}

/** One stage of the pipeline. Either a single step (title/detail) or a set of
 * branches that all run at the same time (asyncio.gather in the real code) -
 * never both. */
export interface FlowStage {
  title: string
  detail?: string
  branches?: FlowBranch[]
  /** Small callout under the stage, e.g. "Currently OFF in this environment". */
  note?: string
  noteTone?: 'info' | 'warning'
}

function ParallelIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M1 4h6M1 12h6M11 4h4M11 12h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M7 4l2.5 4L7 12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  )
}

function Note({ text, tone }: { text: string; tone: 'info' | 'warning' }) {
  const toneClass =
    tone === 'warning'
      ? 'border-[var(--danger)]/30 bg-[var(--danger-soft)] text-[var(--danger)]'
      : 'border-[var(--border)] bg-[var(--surface-hover)] text-[var(--text-muted)]'
  return <p className={`mt-2 rounded-md border px-2.5 py-1.5 text-xs leading-snug ${toneClass}`}>{text}</p>
}

export default function FlowDiagram({ stages, compact = false }: { stages: FlowStage[]; compact?: boolean }) {
  return (
    <ol className="relative flex flex-col">
      {stages.map((stage, i) => {
        const isLast = i === stages.length - 1
        return (
          <li key={stage.title} className={`relative ${compact ? 'pl-9 pb-5' : 'pl-12 pb-8'} last:pb-0`}>
            {!isLast && (
              <span
                aria-hidden="true"
                className={`absolute ${compact ? 'left-[11px] top-6' : 'left-[15px] top-8'} bottom-0 w-px bg-[var(--border)]`}
              />
            )}
            <span
              className={`absolute left-0 top-0 flex items-center justify-center rounded-full bg-[var(--accent)] font-semibold text-[var(--accent-ink)] ${
                compact ? 'h-6 w-6 text-[11px]' : 'h-8 w-8 text-xs'
              }`}
            >
              {i + 1}
            </span>

            <h3 className={`font-semibold text-[var(--text)] pt-1 ${compact ? 'text-[13px]' : 'text-sm'}`}>{stage.title}</h3>

            {stage.detail && (
              <p className={`mt-1 text-[var(--text-muted)] max-w-xl ${compact ? 'text-xs' : 'text-sm'}`}>{stage.detail}</p>
            )}

            {stage.branches && (
              <div className={`mt-2 rounded-lg border border-dashed border-[var(--border)] ${compact ? 'p-2' : 'p-3'}`}>
                <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-[var(--accent)]">
                  <ParallelIcon />
                  {stage.branches.length} running at the same time
                </div>
                <div className={`mt-2 grid gap-2 ${compact ? '' : 'sm:grid-cols-2 lg:grid-cols-3'}`}>
                  {stage.branches.map((branch) => (
                    <div
                      key={branch.title}
                      className="rounded-md border border-[var(--border-soft)] bg-[var(--surface)] px-3 py-2"
                    >
                      <div className="text-xs font-semibold text-[var(--text)]">{branch.title}</div>
                      <div className="mt-0.5 text-xs leading-snug text-[var(--text-muted)]">{branch.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {stage.note && <Note text={stage.note} tone={stage.noteTone ?? 'info'} />}
          </li>
        )
      })}
    </ol>
  )
}
