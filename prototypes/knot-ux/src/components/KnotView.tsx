// Prototype — not production code
import { useState } from 'react'
import type { KnotData, Bead, MemoryType } from '../types'

export interface KnotViewProps {
  data: KnotData
  onBeadClick?: (bead: Bead) => void
}

// Color config by memory type
const TYPE_BADGE: Record<MemoryType, string> = {
  semantic: 'bg-blue-100 text-blue-700 border-blue-200',
  episodic: 'bg-green-100 text-green-700 border-green-200',
  procedural: 'bg-purple-100 text-purple-700 border-purple-200',
}

const TYPE_CARD_BORDER: Record<MemoryType, string> = {
  semantic: 'border-blue-200',
  episodic: 'border-green-200',
  procedural: 'border-purple-200',
}

const TYPE_SELECTED_RING: Record<MemoryType, string> = {
  semantic: 'ring-2 ring-blue-400',
  episodic: 'ring-2 ring-green-400',
  procedural: 'ring-2 ring-purple-400',
}

function truncate(text: string, maxLen: number): string {
  return text.length <= maxLen ? text : text.slice(0, maxLen) + '…'
}

interface BeadCardProps {
  bead: Bead
  isSelected: boolean
  hasCrossing: boolean
  crossingLabel?: string
  onClick: () => void
}

function BeadCard({ bead, isSelected, hasCrossing, crossingLabel, onClick }: BeadCardProps) {
  return (
    <div
      className={`cursor-pointer rounded-lg border bg-white p-3 shadow-sm transition-all hover:shadow-md ${TYPE_CARD_BORDER[bead.type]} ${isSelected ? TYPE_SELECTED_RING[bead.type] : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick() }}
      aria-pressed={isSelected}
    >
      {/* Content */}
      <p className="text-sm text-slate-700 leading-snug mb-2">
        {truncate(bead.content, 80)}
      </p>

      {/* Meta row */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className={`inline-block px-1.5 py-0.5 text-xs font-semibold rounded border ${TYPE_BADGE[bead.type]}`}>
          {bead.type}
        </span>
        {bead.sessionDate && (
          <span className="text-xs text-slate-400 font-mono">{bead.sessionDate}</span>
        )}
        <span className="text-xs text-slate-400 ml-auto">
          ★ {bead.importance.toFixed(2)}
        </span>
      </div>

      {/* Crossing indicator */}
      {hasCrossing && (
        <div className="mt-1.5 text-xs text-amber-600 italic truncate">
          → crossing{crossingLabel ? `: ${crossingLabel}` : ''}
        </div>
      )}
    </div>
  )
}

interface DetailPanelProps {
  bead: Bead
  onClose: () => void
}

function DetailPanel({ bead, onClose }: DetailPanelProps) {
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-5 shadow-md">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs text-slate-400 font-mono mb-0.5">Bead detail</p>
          <p className="text-xs font-mono text-slate-500">{bead.id}</p>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-600 text-lg leading-none flex-none"
          aria-label="Close detail panel"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Content</p>
        <p className="text-sm text-slate-800 leading-relaxed">{bead.content}</p>
      </div>

      {/* Type + importance */}
      <div className="flex items-center gap-3">
        <span className={`inline-block px-2 py-0.5 text-xs font-semibold rounded border ${TYPE_BADGE[bead.type]}`}>
          {bead.type}
        </span>
        <span className="text-sm text-slate-600">importance: <strong>{bead.importance.toFixed(2)}</strong></span>
      </div>

      {/* Dates */}
      <div className="space-y-1 text-sm text-slate-600">
        <div>
          <span className="text-slate-400">session date: </span>
          {bead.sessionDate ?? <span className="text-slate-300">—</span>}
        </div>
        <div>
          <span className="text-slate-400">event date: </span>
          {bead.eventDate ?? <span className="text-slate-300">—</span>}
        </div>
      </div>

      {/* Tags */}
      {bead.tags.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Tags</p>
          <div className="flex flex-wrap gap-1">
            {bead.tags.map((tag) => (
              <span key={tag} className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded-full text-xs">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Entity */}
      <div className="text-xs text-slate-400 font-mono">
        entity: {bead.entityId}
      </div>
    </div>
  )
}

export function KnotView({ data, onBeadClick }: KnotViewProps) {
  const [selectedBead, setSelectedBead] = useState<Bead | null>(null)

  // Build a fast lookup: beadId → crossing labels originating from it
  const crossingsBySource = new Map<string, string[]>()
  for (const crossing of data.crossings) {
    const existing = crossingsBySource.get(crossing.sourceBeadId) ?? []
    existing.push(crossing.label ?? crossing.edgeType)
    crossingsBySource.set(crossing.sourceBeadId, existing)
  }

  function handleBeadClick(bead: Bead) {
    setSelectedBead((prev) => (prev?.id === bead.id ? null : bead))
    onBeadClick?.(bead)
  }

  // Sort beads newest-first (by sessionDate desc, undefined last)
  function sortedBeads(beads: Bead[]): Bead[] {
    return [...beads].sort((a, b) => {
      if (!a.sessionDate && !b.sessionDate) return 0
      if (!a.sessionDate) return 1
      if (!b.sessionDate) return -1
      return b.sessionDate.localeCompare(a.sessionDate)
    })
  }

  return (
    <div className="flex gap-4 w-full">
      {/* Strand columns */}
      <div className="flex-1 overflow-x-auto">
        <div
          className="grid gap-4 min-w-0"
          style={{ gridTemplateColumns: `repeat(${data.strands.length}, minmax(200px, 1fr))` }}
        >
          {data.strands.map((strand) => (
            <div key={strand.entityId} className="flex flex-col gap-2 min-w-0">
              {/* Strand header */}
              <div className="sticky top-0 z-10 bg-slate-50 pb-1">
                <h3 className="text-sm font-semibold text-slate-700 px-2 py-1 bg-slate-100 rounded-lg border border-slate-200 truncate text-center">
                  {strand.label}
                </h3>
                <p className="text-xs text-slate-400 text-center mt-0.5">
                  {strand.beads.length} bead{strand.beads.length !== 1 ? 's' : ''}
                </p>
              </div>

              {/* Bead list */}
              <div className="flex flex-col gap-2">
                {sortedBeads(strand.beads).map((bead) => {
                  const crossingLabels = crossingsBySource.get(bead.id)
                  return (
                    <BeadCard
                      key={bead.id}
                      bead={bead}
                      isSelected={selectedBead?.id === bead.id}
                      hasCrossing={!!crossingLabels && crossingLabels.length > 0}
                      crossingLabel={crossingLabels?.[0]}
                      onClick={() => handleBeadClick(bead)}
                    />
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {selectedBead && (
        <div className="flex-none w-72 xl:w-80">
          <DetailPanel bead={selectedBead} onClose={() => setSelectedBead(null)} />
        </div>
      )}
    </div>
  )
}

export default KnotView
