# Operational team cycle implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Runnable event/timer hosting composes return, context, guarded work and experience.

**Architecture:** Public primitives, local SQLite inbox, bounded inspection/diagnostics and a common two-agent scenario. No new library policy/transport.

**Tech Stack:** Python>=3.11, existing SQLite/PostgreSQL; no dependency/schema/version change.

**Spec:** `docs/superpowers/specs/2026-10-28-team-cycle-design.md`

## Global Constraints

- Preserve protected checkout/release bytes; no publication/push/merge.
- A wake is not a lease/permit; no quiet-state model call or unknown-effect auto retry.

## Review Focus

- Cache commit/ack crashes must not lose work or return stale text.
- Expiry/freshness changes without journal events must still be inspected.
- Unknown/live lease/deadline/manual blocks cannot become automatic retries.
- Concurrent READY observations still need a single successful claim before inference.
- Source/outcome correction must retract experience copies through the same workflow.

### Task1: Bounded host inspection

Create `examples/always_on_team.py`, tests `tests/always_on/test_team_cycle.py`.
Produce Inbox, TurnObservation and inspect_turn with exact spec signature.

- [x] RED SQL tests: idle/wait, filtered backlog, retention/overflow, cache failure/replay, missing/freshness/unknown/lease/deadline/block and claim competition.
- [x] Implement public API composition and diagnostic counts; focused SQL tests GREEN.

### Task2: Executable workflow and guide

Extend the example with a shared workflow callable on either SQL store and a temporary SQLite CLI; integrated/restart/guard/experience tests.
Write `docs/always-on-team-cycle.md`, update example/docs navigation and schema-history references.

- [x] RED integrated/CLI tests; common-corpus return→replan→guard→experience/retraction scenario GREEN.
- [x] Document event/timer one-shot hosting, durability/statuses/limits; execute commands and check links.

### Task3: Whole-program acceptance

Create current validation/completion audit; review from3618bfb through all four stages.

- [x] Full required-PG Python/static/TS/recovery/MA gates and all three SQL load reports; fixed-data/source-hash verification.
- [x] Fresh Astra/Ponytail review; material regressions RED→GREEN and final full suite.
- [x] Requirement-by-requirement completion audit, protected-byte check, feature-only local commit and clean worktree.
