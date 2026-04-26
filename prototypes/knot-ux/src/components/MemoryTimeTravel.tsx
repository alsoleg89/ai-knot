// Prototype — not production code
// STUB: UI shell only. Backend not implemented.

export default function MemoryTimeTravel() {
  return (
    <div className="p-6 border-2 border-dashed border-amber-400 rounded-lg bg-amber-50">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-2xl">⏳</span>
        <h2 className="text-xl font-bold text-amber-800">Memory Time Travel</h2>
        <span className="ml-2 px-2 py-0.5 text-xs bg-amber-200 text-amber-800 rounded-full font-mono">
          STUB — NOT IMPLEMENTED
        </span>
      </div>
      <p className="text-amber-700 mb-4">
        <strong>Backend not yet implemented.</strong> Requires Phase E:{" "}
        <code className="bg-amber-100 px-1 rounded">fact_history</code> table +{" "}
        <code className="bg-amber-100 px-1 rounded">recall_as_of(query, timestamp)</code> API.
      </p>
      {/* Placeholder UI */}
      <div className="opacity-40 pointer-events-none">
        <input
          type="range"
          className="w-full"
          disabled
          placeholder="← past | future →"
        />
        <div className="mt-4 p-4 bg-white rounded border border-amber-200 text-gray-500 text-sm">
          Answer as of [date] would appear here...
        </div>
      </div>
    </div>
  )
}
