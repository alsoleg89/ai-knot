What this change does: It adds an explicit host-authored procedural lesson tied to an action's exact recorded outcome and revision. It retains observation roots, gives repeated calls a durable identity, and withdraws registered lessons when the action revision changes.

Review scope: uncommitted outcome-bound experience milestone against `f877440`, reviewed on 2026-10-28. I read the spec/plan, all new production code and direct callers, the existing lineage/admission and SQL transaction code, workspace operation aliases, private/shared publication and recall, snapshot/restore, HTTP/MCP decoding, TypeScript mappings, and the focused tests. I assumed the documented small shared working set with concurrent agents/processes on SQLite/PostgreSQL, not an unbounded production-scale corpus. No production or repository tests were edited by this review.

**Must fix**

1. **Published copies keep the old action outcome** (`src/ai_knot/always_on/experience.py:44`, `src/ai_knot/pool.py:504`). Priority P1.
   - **What this is:** A private lesson can cite accessible shared observations and be copied into the shared pool through the existing `publish` API.
   - **Problem:** The copy retains its observation dependencies but initially received no action binding. After failed→succeeded reconciliation, the private lesson is withdrawn while the shared copy remains selectable as “Recorded outcome: failed.”
   - **Fix:** The smallest safe contract is to skip direct action-bound lessons in ordinary `pool.publish`, before any shared-slot mutation. Require `record_experience(namespace='__shared__')` for a shared lesson so the existing maintain permission and binding transaction apply.
   - **If we skip it:** Another agent can continue using obsolete failure advice after correction. The public private→publish route also avoids the shared experience operation's maintain check.

2. **Workspace result aliases can produce obsolete experience** (`src/ai_knot/always_on/experience.py:109`, `src/ai_knot/always_on/workspace.py:205`). Priority P1.
   - **What this is:** Preparing the same completed workspace operation for another agent creates an action containing a saved terminal result. It is an alias, not the authoritative action that controls the operation.
   - **Problem:** Recording checked that the alias was accessible, but not that it still owned the workspace operation. After the authoritative action is corrected from failed to succeeded, the alias's existing lesson remains live and a new failed lesson can still be recorded from it.
   - **Fix:** Call the existing workspace ownership check inside the recording transaction, before accepting the action. Reject aliases for current-outcome experience; they can still be retained as historical action records.
   - **If we skip it:** Agents can repeatedly reintroduce an obsolete operation outcome without an evidence or revision error.

3. **Normal snapshot restore resurrects withdrawn lessons** (`src/ai_knot/always_on/experience.py:44`, `src/ai_knot/knowledge.py:1510`, `src/ai_knot/always_on/memory.py:293`). Priority P1.
   - **What this is:** `KnowledgeBase.snapshot` and `restore` save and replace facts, including their support flags and exact revisions, without rolling back durable actions.
   - **Problem:** Snapshot a failed lesson and a derived copy, reconcile the action to succeeded, then restore the snapshot. Both old refs become accepted and selected again because withdrawal only changed the stored fact's support flag; the action remains succeeded.
   - **Fix:** Enforce the persisted action binding at the common SQL fact boundary or all admission paths, including restored/imported originals used by descendant checks. A facts-only restore must not restore authority that the durable action has already revoked.
   - **If we skip it:** An ordinary recovery operation silently restores stale outcome advice and the same evidence checks used before actions accept it again.

All three were independently reproduced on both SQLite and PostgreSQL through public APIs. The reproductions live in `/tmp/ai-knot-team-experience/astra_repros.py`; they create their own temporary SQLite database and disposable PostgreSQL schema, then clean them up. The script's publication and alias cases intentionally document the initially vulnerable behavior; after fixes, rejection or no publication is the safe outcome.

Evidence from the initial public-path reproduction:

```text
sqlite published_copy: private_live=false, published_live=true,
  peer_selected=["Recorded outcome: failed.\nRestore checkpoint: verify backup before switching traffic."]
postgres published_copy: same
sqlite workspace_alias: alias_outcome=failed, existing_alias_lesson_live=true,
  new_alias_lesson_after_correction=failed
postgres workspace_alias: same
```

Evidence from the later snapshot case:

```text
sqlite snapshot_restore: restored_lesson_and_copy_live=2, selected_count=2
postgres snapshot_restore: restored_lesson_and_copy_live=2, selected_count=2
```

Commands and observed results (the dedicated test DSN was supplied through the environment):

```sh
PYTHONPATH=/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot/src \
  /tmp/ai-knot-pr-ci/py312/bin/python \
  /tmp/ai-knot-team-experience/astra_repros.py

AI_KNOT_REQUIRE_POSTGRES=1 \
PYTHONPATH=/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot/src \
  /tmp/ai-knot-pr-ci/py312/bin/python -m pytest \
  tests/always_on/test_team_experience.py \
  tests/always_on/test_surfaces.py tests/always_on/test_mcp_surface.py --no-cov -q
# Initial result: 69 passed, 2 failed in 11.32s.
# The failures were a workspace test fixture that had not reserved a workspace
# operation. The implementation owner corrected the fixture and reported GREEN.

AI_KNOT_REQUIRE_POSTGRES=1 \
PYTHONPATH=/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot/src \
  /tmp/ai-knot-pr-ci/py312/bin/python -m pytest \
  tests/always_on/test_team_experience.py tests/always_on/test_lineage.py \
  tests/always_on/test_publication_grounding.py \
  tests/always_on/test_pool_lineage_sync.py \
  tests/always_on/test_workspace_operations.py \
  tests/always_on/test_surfaces.py tests/always_on/test_mcp_surface.py --no-cov -q
# Intermediate implementation result: 174 passed, 4 failed in 26.40s.
# The four failures belonged to the temporary binding-propagation publication
# approach, subsequently abandoned in favor of rejecting direct publication.
# This is not a test result for the final implementation.
```

During the review, the implementation owner corrected the workspace alias path; my independent rerun then rejected the alias on both SQL backends. An intermediate publication fix withdrew both copies in my public-path rerun, but added a maintain-authority bypass and a rejection after slot mutation. The owner chose the smaller direct-publication rejection above and is implementing persistent snapshot/import handling. Per the owner's instruction, this review records the original findings and does not claim independent verification of those final repairs; the owner will run the final RED/GREEN and full gates.

The original direct recording transaction, action revision update/retraction rollback, source/read-ACL/freshness checks, root deduplication, idempotent replay, input type checks, bounded canonical request payload, same-store namespace validation, and no-pool invalidation discovery were traced. No further concrete finding was established in those paths. The permanent binding rows preserve replay identity; they and full-namespace fact replacement keep the existing small-working-set scaling limits.

Verdict: Revise. Fix 1, 2, and 3 before shipping; this report is not a final repaired-code ship sign-off.

Not checked: Final repairs, final example/workload/docs, TypeScript execution, full repository/static/recovery/MA acceptance, and production-scale load are owned by the implementation task. No claim is made about remote tool success, semantic entailment, learned generalization, or model-answer quality.

## Implementation owner's resolution

All three P1 findings were reproduced with repository regressions before repair. Final behavior skips direct marked publication before any slot mutation, checks current workspace ownership, and applies persisted action bindings at the common SQLite/PostgreSQL fact-write boundary so snapshot/import cannot resurrect withdrawn marked facts. Shared creation uses the dedicated maintain-checked writer. Matching replay remains stable; corrections emit normal fact-change events and withdraw exact-source copies. The temporary publication-propagation helper was removed.

The owner verified 56 core cases on both SQL backends and the final full suite: 1,954 passed, two expected skips, 89.37% coverage; 59 TypeScript tests plus build/typecheck, format/lint/strict mypy, 12/12 recovery scenarios per SQL backend and 11/11 binding multi-agent gates passed. The eight-process default workload was independently rerun by the owner on both SQL backends. See the [validation record](../validation/2026-10-28-team-experience.json). These are post-review owner checks, not a new independent Astra Ship verdict. No review finding remains deferred.
