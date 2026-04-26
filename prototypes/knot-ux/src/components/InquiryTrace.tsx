// Prototype — not production code
import { useState } from 'react'
import type { TraceResult } from '../types'

export interface InquiryTraceProps {
  result: TraceResult
  question: string
  collapsed?: boolean
}

const INTENT_BADGE: Record<string, string> = {
  factual: 'bg-blue-100 text-blue-700',
  aggregational: 'bg-purple-100 text-purple-700',
  exploratory: 'bg-teal-100 text-teal-700',
  temporal: 'bg-orange-100 text-orange-700',
  comparative: 'bg-pink-100 text-pink-700',
}

const Dash = () => <span className="text-gray-400">—</span>

interface StepCardProps {
  step: number
  icon: string
  title: string
  children: React.ReactNode
}

function StepCard({ step, icon, title, children }: StepCardProps) {
  return (
    <div className="flex gap-3 py-3 border-b border-slate-100 last:border-0">
      <div className="flex-none w-7 h-7 rounded-full bg-slate-100 flex items-center justify-center text-xs font-semibold text-slate-500">
        {step}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-base leading-none">{icon}</span>
          <span className="text-xs font-semibold text-slate-600 uppercase tracking-wide">{title}</span>
        </div>
        <div className="text-sm text-slate-700">{children}</div>
      </div>
    </div>
  )
}

export function InquiryTrace({ result, question, collapsed: initialCollapsed = false }: InquiryTraceProps) {
  const [collapsed, setCollapsed] = useState(initialCollapsed)
  const { trace } = result

  // Stage 1 breakdown
  const bm25Count = trace.stage1_candidates?.from_bm25?.length ?? 0
  const rareCount = trace.stage1_candidates?.from_rare_tokens?.length ?? 0
  const hopCount = trace.stage1_candidates?.from_entity_hop?.length ?? 0
  const totalUnique =
    trace.stage1_candidates?.total ??
    new Set([
      ...(trace.stage1_candidates?.from_bm25 ?? []),
      ...(trace.stage1_candidates?.from_rare_tokens ?? []),
      ...(trace.stage1_candidates?.from_entity_hop ?? []),
    ]).size

  // Stage 0 lexical bridge
  const lexBridge = (trace as Record<string, unknown>).stage0_lexical_bridge as
    | { frames_applied?: string[]; terms_added?: string[] }
    | null
    | undefined

  // Evidence confidence
  const evidenceCount = result.pack_fact_ids?.length ?? 0
  const hasDense = Array.isArray(trace.stage3b_dense_guarantee) && trace.stage3b_dense_guarantee.length > 0

  // Context preview
  const contextPreview = result.context ? result.context.slice(0, 200) : null

  // Intent badge
  const intentClass = trace.intent ? (INTENT_BADGE[trace.intent] ?? 'bg-gray-100 text-gray-600') : ''

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      {/* Header — clickable to toggle */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-start justify-between gap-3 px-5 py-4 text-left hover:bg-slate-50 transition-colors"
      >
        <div className="min-w-0">
          <p className="text-xs text-slate-400 font-mono mb-0.5">Recall trace</p>
          <p className="text-sm font-medium text-slate-800 truncate">{question}</p>
        </div>
        <span className="flex-none text-slate-400 text-lg leading-none mt-0.5">
          {collapsed ? '▶' : '▼'}
        </span>
      </button>

      {/* Steps */}
      {!collapsed && (
        <div className="px-5 pb-4">
          {/* Step 1: Intent */}
          <StepCard step={1} icon="🎯" title="Intent">
            {trace.intent ? (
              <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${intentClass}`}>
                {trace.intent}
              </span>
            ) : (
              <Dash />
            )}
          </StepCard>

          {/* Step 2: Lexical Expansion */}
          <StepCard step={2} icon="🔤" title="Lexical Expansion">
            {lexBridge ? (
              <div className="space-y-1">
                {lexBridge.frames_applied && lexBridge.frames_applied.length > 0 ? (
                  <div>
                    <span className="text-slate-500">Frames: </span>
                    {lexBridge.frames_applied.map((f) => (
                      <span key={f} className="inline-block mr-1 px-1.5 py-0.5 bg-indigo-50 text-indigo-700 rounded text-xs">
                        {f}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-500">No frames applied</span>
                )}
                {lexBridge.terms_added && lexBridge.terms_added.length > 0 && (
                  <div className="text-slate-500">
                    Terms added: {lexBridge.terms_added.join(', ')}
                  </div>
                )}
              </div>
            ) : (
              <span className="text-slate-400 italic">Bridge inactive</span>
            )}
          </StepCard>

          {/* Step 3: Candidates */}
          <StepCard step={3} icon="🔍" title="Candidates">
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs">
                BM25 <strong>{bm25Count}</strong>
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-violet-50 text-violet-700 rounded text-xs">
                Rare tokens <strong>{rareCount}</strong>
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded text-xs">
                Entity hop <strong>{hopCount}</strong>
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-semibold">
                Total unique {totalUnique}
              </span>
            </div>
          </StepCard>

          {/* Step 4: Evidence Pack */}
          <StepCard step={4} icon="📦" title="Evidence Pack">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-slate-700">
                <strong>{evidenceCount}</strong> fact{evidenceCount !== 1 ? 's' : ''} selected
              </span>
              {hasDense && (
                <span className="inline-block px-2 py-0.5 bg-cyan-50 text-cyan-700 border border-cyan-200 rounded text-xs">
                  dense path active
                </span>
              )}
              {evidenceCount === 0 && (
                <span className="inline-block px-2 py-0.5 bg-red-50 text-red-600 border border-red-200 rounded text-xs font-semibold">
                  No evidence found
                </span>
              )}
            </div>
          </StepCard>

          {/* Step 5: Context */}
          <StepCard step={5} icon="📄" title="Context">
            {contextPreview ? (
              <p className="text-slate-600 text-xs leading-relaxed font-mono bg-slate-50 rounded p-2 break-words">
                {contextPreview}
                {result.context.length > 200 && (
                  <span className="text-slate-400"> …[{result.context.length - 200} more chars]</span>
                )}
              </p>
            ) : (
              <Dash />
            )}
          </StepCard>

          {/* Step 6: Confidence */}
          <StepCard step={6} icon="✅" title="Confidence">
            {evidenceCount === 0 ? (
              <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-50 text-red-700 border border-red-200 rounded text-xs font-semibold">
                🔴 No evidence found
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-50 text-green-700 border border-green-200 rounded text-xs font-semibold">
                🟢 {evidenceCount} evidence item{evidenceCount !== 1 ? 's' : ''} retrieved
              </span>
            )}
          </StepCard>
        </div>
      )}
    </div>
  )
}

export default InquiryTrace
