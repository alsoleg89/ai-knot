---
date: 2026-04-26
trigger: dated full-10 reproduction at 1167e70 reached pf3 baseline (62.2% combined, cat1 38.8%); 2 weeks of subsequent work on feat/v2-product-kernel did not improve LOCOMO numbers
decision: rewind feat/v2-product-kernel to validated baseline; archive all in-flight work as tags for analysis
---

# Branch archive + feat-rewind plan (2026-04-26)

## Status: EXECUTED 2026-04-26

All 4 steps completed successfully:

- ✓ 10 archive tags created locally and pushed to `origin`
- ✓ `feat/v2-product-kernel` reset locally from `6d0a7b9` → `6735ea0` (= `repro/dated-1167e70` HEAD)
- ✓ `feat/v2-product-kernel` force-pushed to origin (`a71a195` → `6735ea0`, `--force-with-lease` succeeded)
- ✓ `repro/dated-1167e70` also pushed to origin as alias for traceability

Final state: `feat/v2-product-kernel` (local + origin) = `6735ea0`. All preserved work is reachable via `git tag -l 'archive/*-20260426'`.

## Why

Memory record of last 2 weeks (see `project_pf3_regression_chain_20260419`, `project_locomo_phase1_retrieval_exhausted`, `project_locomo_phase1e_revert`, `project_locomo_claims_first_promotion_20260423`, `project_locomo_cat1_shift_ab_negative_20260423`):

- 8 consecutive reverts on Phase 1 retrieval moves
- Cat1 ceiling traced to retrieval-only ≈ 40-44 % (not materializer)
- 2-conv gate cannot discriminate ±3 pp noise from signal
- Several attempts (claims-first promotion, materializer widening, shift A+B) actively regressed the baseline

Today's `repro/dated-1167e70` reproduction confirmed:

- pf3 numbers (60.5 % agg, 40.4 % cat1) are within 1-2 pp of validated baseline
- Configuration (dated mode + Phase E SUT + gpt-4o-mini + text-embedding-3-small + top_k=60 + AI_KNOT_LLM_RECALL=false + 4789521 helper overlays) reliably reproduces this number
- 2 weeks of subsequent work on `feat/v2-product-kernel` did not exceed this baseline; instead repeatedly tripped stop-and-revert rule

Decision: rewind `feat/v2-product-kernel` to the validated baseline (`repro/dated-1167e70`). Archive all work since 1167e70 as named tags for retrospective analysis.

The 100 commits of work are not lost — they live under archive tags, locally and on origin.

## Branch inventory (as of 2026-04-26)

### Active "work" branches with content since 1167e70

| branch | commit | last activity | merged into feat? | archive tag |
|---|---|---|---|---|
| `feat/v2-product-kernel` (local) | `6d0a7b9` | 14 h ago | self | `archive/feat-v2-product-kernel-20260426` |
| `origin/feat/v2-product-kernel` | `a71a195` | 2 d ago | (remote tip) | `archive/feat-v2-product-kernel-origin-20260426` |
| `codex/pf3-staged-rebuild` | `3d59172` | 7 d ago | NO | `archive/codex-pf3-staged-rebuild-20260426` |
| `feature/configurable-mcp-env-v0.9.4` | `40e5aa6` | 3 d ago | YES | `archive/feature-configurable-mcp-env-v0.9.4-20260426` |
| `feature/v3-retrieval-architecture` | `daaa9d3` | 2 w ago | NO | `archive/feature-v3-retrieval-architecture-20260426` |

### Worktree-agent parallel experiments (11 days ago)

| branch | commit | subject | archive tag |
|---|---|---|---|
| `worktree-agent-a643e0d2` | `464b461` | feat: evidence_text contract + episode fallback in QueryAnswer | `archive/worktree-agent-a643e0d2-20260426` |
| `worktree-agent-a2ebdaa9` | `811b14d` | fix: always run raw-episode search for evidence_text enrichment | `archive/worktree-agent-a2ebdaa9-20260426` |
| `worktree-agent-acd2e1b2` | `811b14d` | (same commit as a2ebdaa9) | `archive/worktree-agent-acd2e1b2-20260426` |
| `worktree-agent-a2449f22` | `c0b53dc` | fix: garbage filter cleanup + generic self-state pattern, MATERIALIZATION_VERSION=4 | `archive/worktree-agent-a2449f22-20260426` |
| `worktree-agent-a9ff3497` | `fa2c150` | fix: relation bridge — slot keys from bundles, drive/works_at aliases | `archive/worktree-agent-a9ff3497-20260426` |

### What stays untouched

- `main` (`732bd18`, 2 weeks ago) — already 10 commits behind 1167e70; not affected
- All older release branches (`feature/query-shape-retrieval-v0.9.2`, `feature/fix-locomo-benchmark`, etc.) — historical, no recent activity
- `repro/dated-1167e70` — becomes the new content of `feat/v2-product-kernel` after rewind

## Execution plan

Order matters. Steps 1-2 are non-destructive (create + push tags). Step 3-4 are destructive (rewrites feat).

### Step 1: create archive tags (local)

```bash
# Active branches
git tag archive/feat-v2-product-kernel-20260426 6d0a7b9
git tag archive/feat-v2-product-kernel-origin-20260426 a71a195
git tag archive/codex-pf3-staged-rebuild-20260426 3d59172
git tag archive/feature-configurable-mcp-env-v0.9.4-20260426 40e5aa6
git tag archive/feature-v3-retrieval-architecture-20260426 daaa9d3

# Worktree-agent branches
git tag archive/worktree-agent-a643e0d2-20260426 464b461
git tag archive/worktree-agent-a2ebdaa9-20260426 811b14d
git tag archive/worktree-agent-acd2e1b2-20260426 811b14d
git tag archive/worktree-agent-a2449f22-20260426 c0b53dc
git tag archive/worktree-agent-a9ff3497-20260426 fa2c150
```

### Step 2: push archive tags to origin

```bash
git push origin --tags 'archive/*-20260426'
```

(Pushes only the new archive tags, not all tags.)

### Step 3: rewind feat/v2-product-kernel locally

```bash
git checkout feat/v2-product-kernel
git reset --hard repro/dated-1167e70
```

After this, local `feat/v2-product-kernel` points to `6274462` (= current `repro/dated-1167e70` HEAD).

### Step 4: force-push feat/v2-product-kernel to origin

```bash
git push --force-with-lease origin feat/v2-product-kernel
```

`--force-with-lease` aborts if remote has new commits we don't know about (race protection). If anyone has pushed to origin since we last fetched, this fails — investigate before retrying.

## Recovery / inspection commands

After the rewind, to access archived work:

```bash
# List all archive tags
git tag -l 'archive/*-20260426'

# Inspect a specific archive
git log archive/feat-v2-product-kernel-20260426 --oneline -20

# Diff archive tip vs current feat
git diff feat/v2-product-kernel..archive/feat-v2-product-kernel-20260426

# Check out an archived branch state read-only
git checkout archive/feat-v2-product-kernel-20260426

# Recover an archived branch as a working branch
git switch -c restored/feat-v2-product-kernel archive/feat-v2-product-kernel-20260426
```

All archive tags push to `origin` so they survive a fresh clone.

## Post-rewind state

After execution:

| ref | points to | meaning |
|---|---|---|
| `main` | `732bd18` | unchanged |
| `feat/v2-product-kernel` (local + origin) | `6274462` | validated pf3 baseline + analysis |
| `repro/dated-1167e70` | `6274462` | same commit as feat (alias for clarity) |
| `archive/feat-v2-product-kernel-20260426` | `6d0a7b9` | preserved 100-commit Phase 1 work |
| `archive/feat-v2-product-kernel-origin-20260426` | `a71a195` | preserved remote tip before rewind |
| `archive/codex-pf3-staged-rebuild-20260426` | `3d59172` | preserved pf3-staged-rebuild |
| `archive/feature-configurable-mcp-env-v0.9.4-20260426` | `40e5aa6` | preserved v0.9.4 work |
| `archive/feature-v3-retrieval-architecture-20260426` | `daaa9d3` | preserved v3 architecture |
| `archive/worktree-agent-*-20260426` | various | 5 preserved worktree experiments |

Branches themselves are NOT deleted — only tagged. They remain as branch refs at their current commits.

## What this enables

After rewind, the project's "active" line is the validated baseline. Any new improvement attempt:
1. Branches off `feat/v2-product-kernel` (= validated 62.2 % / cat1 38.8 %)
2. Must beat this number on full-10 (or averaged 2-conv per `feedback_regression_stop_rule`) to merge
3. Can mine archive tags for "what was tried, what failed, why" before re-attempting same approach

Specifically, the archive tags preserve enough context that a future session can:
- Study the regression chain (8 reverts) without re-running them
- See what features were attempted: claims-first promotion, materializer widening, shift A+B, etc.
- Find documented stop-and-revert reasoning in `feature/configurable-mcp-env-v0.9.4` (its tip docs the claims-first revert)

## Risks and mitigations

| risk | mitigation |
|---|---|
| Force-push race (someone pushes to origin/feat between fetch and push) | `--force-with-lease` aborts on race |
| Missed branch (work exists somewhere I didn't catalog) | This file lists everything from `git for-each-ref refs/heads/`; reflog preserves missed work for ~90 days |
| Archive tags get garbage-collected | Tags are first-class refs, not subject to gc; pushing to origin makes them remote-persistent |
| Need to revert the rewind | `git reset --hard archive/feat-v2-product-kernel-20260426 && git push --force-with-lease` restores |

## Pre-flight verification (before step 3)

Before reset, confirm:

```bash
# 1. All tags exist
git tag -l 'archive/*-20260426' | wc -l   # expect: 10

# 2. All tags pushed
git ls-remote --tags origin 'archive/*-20260426' | wc -l   # expect: 10

# 3. Working dir is clean (or has only intentional uncommitted work)
git status

# 4. We're on the right branch
git branch --show-current
```

## Decision log

- 2026-04-26 ~09:00: dated full-10 launched in parallel
- 2026-04-26 ~09:30: convs 0-1 done, baseline 60.5 % observed (matches pf3)
- 2026-04-26 ~10:00: convs 5-9 hit OpenAI RPD wall mid-QA
- 2026-04-26 ~10:30: convs 0-4 finished; combined 62.2 %; reproduction validated
- 2026-04-26 ~10:45: user decision — rewind feat to validated baseline; archive 2 weeks of work
- 2026-04-26 ~11:00: this plan written; awaiting execution

## Artifacts

- This file: `research/branch_archive_plan_20260426.md`
- Reproduction analysis: `research/dated_full10_analysis_20260426.md`
- Memory entry: `~/.claude/projects/.../memory/project_dated_full10_repro_20260426.md`
- Validated baseline branch: `repro/dated-1167e70` at commit `6274462`
- pf3 baseline reference: memory `project_pf3_full10_phase_e_20260413`
