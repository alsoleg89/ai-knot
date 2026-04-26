// Prototype — not production code
import { useState } from 'react'
import InquiryTrace from './components/InquiryTrace'
import KnotView from './components/KnotView'
import MemoryTimeTravel from './components/MemoryTimeTravel'
import traceExamples from './data/mock-trace.json'
import knotData from './data/mock-knot.json'
import type { TraceExample, KnotData } from './types'

type Tab = 'trace' | 'knot' | 'timetravel'

const TAB_LABELS: Record<Tab, string> = {
  trace: 'InquiryTrace',
  knot: 'KnotView',
  timetravel: 'MemoryTimeTravel',
}

const examples = traceExamples as TraceExample[]
const knot = knotData as KnotData

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('trace')

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Knot UX Prototype</h1>
            <p className="text-sm text-slate-500 mt-0.5">ai-knot · Cycle U.trace</p>
          </div>
          <span className="inline-flex items-center rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800 border border-amber-200">
            Prototype — not production
          </span>
        </div>
      </header>

      {/* Tab navigation */}
      <nav className="bg-white border-b border-slate-200 px-6">
        <div className="max-w-5xl mx-auto flex gap-0">
          {(Object.keys(TAB_LABELS) as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-slate-800 text-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </div>
      </nav>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* InquiryTrace tab */}
        {activeTab === 'trace' && (
          <section>
            <div className="mb-6">
              <h2 className="text-base font-semibold text-slate-700 mb-1">Recall pipeline trace</h2>
              <p className="text-sm text-slate-500">
                Glass-box viewer for the 6-stage recall pipeline. Click a header to collapse/expand.
              </p>
            </div>
            <div className="flex flex-col gap-4">
              {examples.map((ex, i) => (
                <InquiryTrace
                  key={i}
                  result={ex.result}
                  question={ex.question}
                  collapsed={i > 0}
                />
              ))}
            </div>
          </section>
        )}

        {/* KnotView tab */}
        {activeTab === 'knot' && (
          <section>
            <div className="mb-6">
              <h2 className="text-base font-semibold text-slate-700 mb-1">Entity strand view</h2>
              <p className="text-sm text-slate-500">
                Click a bead to see full detail. Beads are sorted newest-first within each strand.
                Crossings are indicated with a → marker.
              </p>
            </div>
            <KnotView data={knot} />
          </section>
        )}

        {/* MemoryTimeTravel tab */}
        {activeTab === 'timetravel' && (
          <section>
            <div className="mb-6">
              <h2 className="text-base font-semibold text-slate-700 mb-1">Memory Time Travel</h2>
              <p className="text-sm text-slate-500">
                Phase D component — UI shell only.
              </p>
            </div>
            <MemoryTimeTravel />
          </section>
        )}
      </main>
    </div>
  )
}
