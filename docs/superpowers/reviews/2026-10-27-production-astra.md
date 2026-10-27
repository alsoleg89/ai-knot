# Independent production review — P1–P3

Reviewed 2026-10-27 against the supplied canonical always-on design and uncommitted implementation on `codex/always-on-p1-p3` (base `69e2adf`). Inspection concentrated on the actual state transitions, transaction/session implementation, subscriptions, maintenance and HTTP dispatch, including their connected callers. Existing test counts were not treated as proof. No production/test files were changed.

**Initial verdict (superseded by the targeted re-review below): changes required.** Two reproducible concurrency defects undermined the intended stale-result protections. Both reproduced on SQLite using real `KnowledgeBase`, `SQLiteStorage`, and `AlwaysOnMemory` instances. The affected logic is backend-independent; PostgreSQL was inspected but these additional repros were not run on PostgreSQL.

## P1 — Journal pruning removes the consolidation conflict guard

**Location:** `src/ai_knot/always_on/maintenance.py:324–341` (especially the `ao_changes` lookup at lines 327–331).

The commit guard assumes absence of a retained journal entry means an output slot has not changed since `anchor_seq`. Neither normal `prune_changes()` nor maintenance cleanup retains events for pending consolidation preparations. While a provider is awaited, another maintenance run (or the public prune method) can remove a foreground update from the journal. The stale proposal then passes the guard and closes the newer foreground slot value at line 341. The input episode can remain unchanged throughout, so evidence-reference validation does not catch it.

**Reproduction:** Seed a two-day-old episodic fact and run maintenance with a consolidator. In `propose`, after the run captured its anchor, add a foreground resolved fact for slot `deploy::target`, call `ao.prune_changes()` with its defaults and no subscriptions, then return a valid semantic proposal for the same slot derived from the episode. The run reports `completed`, `consolidated=1`; the foreground fact is inactive and the stale background fact is active. The reproduction below produced exactly this result. Normal pruning is sufficient; `force=True` is unnecessary.

This violates the design's revision-guarded maintenance commit and prohibition on overwriting newer foreground results. Locks serialize the individual transactions but do not preserve the information needed across the asynchronous preparation gap.

**Minimal correction:** Under the commit lock, reject/retry if the stream's `min_retained_seq` is greater than `anchor_seq + 1`, before relying on event absence. Alternatively pin the anchor in the pruning frontier for the preparation lifetime, or use a durable per-slot revision independent of journal retention. Add a race regression test covering ordinary pruning/another maintenance run between preparation and commit on both backends.

## P1 — Old-source verification confirms a newly configured policy

**Location:** `src/ai_knot/always_on/maintenance.py:368–375`; related request construction in `src/ai_knot/always_on/memory.py:655–663` and unguarded policy replacement in `set_freshness`.

A `VerificationRequest` captures the old policy's `source_ref`, but `_verify` reloads whatever policy exists at commit and passes it straight to `record_verification`. Only the fact revision and maintenance-run generation are guarded. If `set_freshness` changes the source or resets the policy while the external verifier is awaited, a response for the old source marks the replacement policy confirmed and uses the replacement interval. This defeats the explicit rule that policy registration/reconfiguration is unverified and can permit actions without checking the newly required source.

**Reproduction:** Configure one current fact with source `old-source`, interval 60. Start maintenance verification. Inside `verify(request)`, reconfigure that same fact with source `new-source`, interval 999, then return a confirmed result carrying evidence `{'source': 'old-source'}`. Maintenance reports `verified=1`. The resulting policy has `source_ref='new-source'`, `last_outcome=CONFIRMED`, old-source evidence and a 999-second verified window. The reproduction below produced exactly this combination.

**Minimal correction:** Capture a freshness-policy generation/revision with the verification request and CAS it at commit. Increment it on `set_freshness` and reject stale results. Checking source identity alone addresses the shown source-change case but does not guard an intentional reset of the same source/configuration. Apply the same token contract to host-driven asynchronous `record_verification` so that path cannot confirm a reset policy with an outstanding response either. Add concurrent reconfiguration and same-policy-reset tests.

## Executed reproduction

Run with `.venv/bin/python /tmp/astra_repro.py` from the repository. Standalone source follows; it uses temporary SQLite files and makes no repository edits.

```python
import asyncio,json,tempfile
from datetime import datetime,UTC,timedelta
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.always_on import AlwaysOnMemory
from ai_knot.always_on.maintenance import *
from ai_knot.always_on.models import VerificationResult,VerificationOutcome
now=datetime.now(UTC)
s=SQLiteStorage(tempfile.mktemp(suffix='.db')); kb=KnowledgeBase('owner',storage=s); ao=AlwaysOnMemory(kb,clock=lambda:now)
e=Fact('old event',type=MemoryType.EPISODIC,created_at=now-timedelta(days=2),valid_from=now-timedelta(days=2)); s.save('owner',[e])
class Race:
 config_key='race';model_cost=0
 async def propose(self,episodes):
  refs=ao.capture_evidence(tuple(('owner',x.id) for x in episodes))
  kb.add_resolved([Fact('foreground',slot_key='deploy::target',value_text='foreground')])
  ao.prune_changes()
  return ConsolidationProposal((Fact('stale background',slot_key='deploy::target',value_text='stale',qualifiers={'consolidation_ids':json.dumps([e.id])}),),(e.id,),refs)
r=asyncio.run(run_maintenance(ao,run_key='race',budget=MaintenanceBudget(),consolidator=Race()))
print('prune race',r,[(f.content,f.is_active(now)) for f in s.load('owner')])
f=Fact('source-bound',created_at=now,valid_from=now);s.save('owner',s.load('owner')+[f]); ref=ao.capture_evidence((('owner',f.id),))[0];ao.set_freshness(ref,source_ref='old-source',recheck_after_seconds=60)
class V:
 config_key='v';model_cost=0
 async def verify(self,request):
  ao.set_freshness(ref,source_ref='new-source',recheck_after_seconds=999)
  return VerificationResult(VerificationOutcome.CONFIRMED,{'source':'old-source'},'old-v1')
r=asyncio.run(run_maintenance(ao,run_key='verify',budget=MaintenanceBudget(),verifier=V()))
with ao._transaction(write=False) as session: p=ao._policy(session,'owner',f)
print('policy race',r,p)
```

Observed first case: `status='completed', consolidated=1`, with `('foreground', False)` and `('stale background', True)`.

Observed second case: `status='completed', verified=1`, and a confirmed `new-source` policy containing `{'source': 'old-source'}`, `source_version='old-v1'`, and interval 999.

No further material finding was established in the inspected paths. This is not a claim that every line of the large patch was exhaustively reviewed or that the full final test gate was rerun.


## Targeted re-review after corrections — 2026-10-27

**Current verdict: both findings resolved for the built-in maintenance path; no remaining blocking finding in this targeted re-review.** This supersedes the initial verdict above. The original findings and repro are retained as historical evidence. This re-review does not extend approval to unrelated evolving changes or replace the final project gate.

1. **Journal pruning guard:** inspected `maintenance.py:343–348`, its following slot-event lookup and the pruning frontier updates. The commit now reads `min_retained_seq` under the same namespace write lock and rejects if any potentially relevant post-anchor history was lost (`min_retained_seq > anchor_seq + 1`). Pruning through the anchor remains safe; pruning later history is conservatively rejected. The provider input and foreground fact remain intact on conflict.
2. **Freshness policy guard:** inspected `maintenance.py:391–432` and `569–613`, plus `state_storage.py:90–103` and `362–395`. The verifier captures the full policy and internal revision in one consistent snapshot before invocation. Its success, error and cancellation write paths all compare both under the namespace lock. Every saved freshness record increments the internal revision, so an identical reset at an identical clock value also fences an outstanding result. A newer verdict is protected as well. Freshness migration adds the revision column with a non-null default for existing rows; repeated initialization does not reset it.
3. **Public manual attestation contract:** inspected `docs/always-on.md:53–57`. `record_verification(ref,result)` explicitly means a trusted host attestation against the current policy; hosts performing their own asynchronous verification coordinate policy changes themselves. Built-in maintenance supplies the stronger snapshot/generation guard. With this explicit boundary, the unchanged public signature is not an outstanding finding from this review.

**Reviewer-executed validation:**

- `.venv/bin/pytest --no-cov -q tests/always_on/test_maintenance.py tests/always_on/test_state_contracts.py -k 'pruning_during or outstanding_verifier or freshness_internal_revision'` against SQLite and the local PostgreSQL service: **30 passed, 96 deselected**. These cover public-prune and maintenance-prune races; source change, interval change, newer changed verdict, identical reset crossed with confirmed/error/cancelled verifier outcomes; and monotonic revision updates.
- Additional SQLite old-schema simulation: created a populated policy, removed the new revision/dirty columns, initialized again, verified the complete policy was preserved and revision initialized to 1, saved again and reinitialized, verified revision remained 2. **Passed.** PostgreSQL additive DDL was inspected; a separate pre-column PostgreSQL migration simulation was not run.
- An initial SQLite-only targeted invocation omitted `--no-cov`: all 14 selected SQLite tests passed, 14 PostgreSQL cases skipped, but the command exited nonzero solely because a targeted subset cannot meet the configured whole-project 80% coverage threshold. The subsequent no-coverage runs above passed. No full-suite coverage claim is made here.
