# knot-ux — Prototype Scaffold

> **Prototype only — not production code.**
> This directory contains a standalone UX prototype for ai-knot.
> It is not connected to any production backend.

## Quick start

```bash
cd prototypes/knot-ux
npm install
npm run dev
```

Vite starts a dev server at `http://localhost:5173`.

## Stack

| Tool | Version |
|------|---------|
| Vite | ^5.4 |
| React | ^18.3 |
| Tailwind CSS | ^3.4 |
| TypeScript | ^5.4 |

## Mock data

All fixtures are **synthetic** — names, events, and dates are invented.
No data from real benchmark datasets (LOCOMO or otherwise) is included.

| File | Contents |
|------|----------|
| `src/data/mock-trace.json` | 3 recall traces: FACTUAL / AGGREGATIONAL / EXPLORATORY. Entities: Sarah, Tom, Apollo (dog), Camping Trip. |
| `src/data/mock-knot.json` | KnotData: 4 entity strands, 18 beads, 6 crossings. Date range 2024-01 to 2024-06. |

## Planned components and backend dependencies

| Component | Description | Requires |
|-----------|-------------|---------|
| **InquiryTrace** | Multi-stage recall pipeline visualiser (BM25 → entity-hop → RRF → MMR) | `recall_with_trace` MCP tool (Phase C) |
| **KnotView** | Interactive entity-strand graph with bead type / importance overlay | Versioned KnotData API (Phase C) |
| **MemoryTimeTravel** | Timeline scrubber over memory snapshots | Snapshot store (Phase D) |
| **PromiseLedger** | Procedural commitment tracker with fulfilment status | Procedural materialiser (Phase D) |

## No external dependencies on ai-knot Python packages

This prototype is entirely standalone. `npm install` installs only the
packages listed in `package.json`. No `ai_knot` Python package, no
`aiknotbench` workspace dependency.

## Development notes

- TypeScript types for trace and Knot view are in `src/types.ts`.
- The app entry point is `src/main.tsx` → `src/App.tsx`.
- Tailwind config scans `index.html` and `src/**/*.{ts,tsx}`.
