"""Causal Witness Path (CWP) framework — bench-side experimental implementation.

CWP is a path-based memory primitive: each belief is reconstructed by traversing
a deterministic chain (atom + supporting raw observations + persistence weight),
not retrieved as a flat result. See `/Users/alsoleg/.claude/plans/giggly-humming-lightning.md`
for the full theoretical program.

bench/ only — no LLM calls in core/ ops/ store/ api/.
"""
