#!/usr/bin/env bash
# Launch the ai-knot MCP server using THIS worktree's src (shadows the editable
# install), so the LongMemEval harness exercises the worktree's event_time
# persistence + MCP event_time wiring rather than the shared checkout's code.
WT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${WT_ROOT}/src:${PYTHONPATH}"
exec /Users/alsoleg/Documents/github/ai-knot/.venv/bin/ai-knot-mcp "$@"
