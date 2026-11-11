# Operational always-on team cycle

The final program stage makes durable return, explicit task context and outcome-bound experience usable together. Existing APIs suffice; provide an executable host recipe instead of another scheduler/public policy API. The user authorized autonomous design/implementation.

## Contract

Create `examples/always_on_team.py` with a bounded `inspect_turn(memory, inbox, *, intent_id, subscription_id, requirements, task_key="", role="", max_pages=4, max_facts=3, max_chars=600)` helper and local SQLite `Inbox`. A typed observation carries current context, resume packet and content-free diagnostics. The helper calls no model/effect and claims no lease.

- Inbox keys include owner/subscription. Selected current context/refs and the delivery receipt commit together before ack. Failure before commit leaves delivery unacknowledged; failure after commit before ack safely replays. Previously received context cannot suppress unfinished READY work.
- Drain at most max_pages deliveries. Empty filtered pages are not completion; debounce with backlog and no token returns to the timer. CursorExpired/resync_required gets a bounded complete snapshot; SnapshotTooLarge returns an actionable narrow-subscription status without acknowledging a partial set.
- Check saved refs on every invocation, select current context after each page and again after resume. Timers must notice expiry/freshness without new journal events. Reject stale cached text rather than appending it. Respect whole-fact count/character bounds.
- Wake-model candidates require drained journal, complete context, no live lease/unknown effect/deadline/manual block, and READY or stale-basis replanning. A manual block remains quiet even when its sources change. Waiting/terminal/missing-context/verification-due/reconciliation states remain distinct. Host must claim before inference; prepare/begin_action recheck evidence afterward. Inspection is point-in-time, not an action permit.

## Approach and acceptance

A new combined library packet/transport family would duplicate explicit operations and bake in host wake policies; three disconnected demos would not test their interaction. Use one common two-agent corpus and runnable event/timer recipe, with no default KB/recall/dependency/schema/version change.

Tests on SQLite/PostgreSQL: quiet idle ticks; >1000 filtered changes before relevant updates; bounded drain; retention recovery/overflow; cache commit/ack crash; stale context, missing evidence, verification deadline without an event, live lease/unknown effect/deadline/manual block; claim competition. Integrated workflow reopens independent connections, replans, rejects an action whose evidence changed, records a shared successful lesson, then retracts it for peers. CLI proof on SQLite; existing process recovery/load covers real crashes/uncertain effects. No actual model/embedding/remote operation.

Run full local required-PostgreSQL/static/TS/recovery/MA acceptance and all three load workloads on both SQL backends; verify fixed-data comparison controls/source hashes. Fresh independent Astra/Ponytail review inspects the full program from3618bfb through this integration. Update navigation and a requirement-to-evidence completion audit. Preserve30protected original files/HEAD/status and deferred releases; local feature commit only.
