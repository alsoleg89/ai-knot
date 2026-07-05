# Multi-agent memory governance

Updated: **July 5, 2026**

Shared memory is where multi-agent systems quietly break. The moment two agents
can write to the same store, you inherit a distributed-systems problem: whose
version wins, why did a bad fact spread, who wrote this, and can you prove what
the fleet knew at time *T*. ai-knot's `SharedMemoryPool` treats that as a
first-class problem and solves it **deterministically — no LLM on the read or
write path.**

This is ai-knot's most differentiated surface and the one most worth
understanding before you build anything with more than one agent.

---

## The problem, as the field now frames it

A 2026 line of work has crystallized *governed shared memory* as a named problem
for multi-agent LLM systems. "Governed Shared Memory for Multi-Agent LLM Systems"
([arXiv:2606.24535](https://arxiv.org/abs/2606.24535)) formalizes the
"fleet-memory problem" and names four foundational failure modes:

| Failure mode (arXiv 2606.24535) | What it looks like in practice |
|---|---|
| **Unauthorized leakage** | an agent recalls a fact it should never have been able to see |
| **Stale propagation** | a corrected fact keeps resurfacing; agents act on outdated state |
| **Contradiction persistence** | two conflicting values for the same slot both survive |
| **Provenance collapse** | nobody can say which agent wrote a fact, or what it superseded |

Its conclusion is blunt: *"Long-context retrieval alone is insufficient for
production multi-agent memory. Governed shared memory demands explicit
systems-level abstractions."* Parallel work benchmarks the same territory
([GateMem, arXiv:2606.18829](https://arxiv.org/abs/2606.18829)) and hardens it
against memory poisoning
([arXiv:2606.24322](https://arxiv.org/abs/2606.24322)).

ai-knot ships those systems-level abstractions as a self-hosted, dependency-free
library you run yourself — not a hosted service you send your data to. The four
primitives the paper defines (*scoped retrieval, temporal supersession,
provenance tracking, policy-governed propagation*) map almost one-to-one onto
mechanisms that already exist in `pool.py`.

---

## The five mechanisms (and where they live in the code)

Every claim below points at real code, so you can verify it rather than trust it.

### 1. Access control — scoped retrieval

Facts carry a `visibility_scope`; an agent only sees `global` facts, its own, and
scopes it has been explicitly granted. Grants are auditable
(`granted_by` / `granted_at`) and survive a restart on ACL-capable backends.

- `SharedMemoryPool.grant_read(agent_id, scope)` — [`pool.py:190`](../src/ai_knot/pool.py)
- Durable grants: `ACLStoreCapable.save_grant/load_grants/revoke_grant` — [`storage/base.py:134`](../src/ai_knot/storage/base.py)

### 2. Temporal supersession — stale facts stop propagating

Publishing to a slot uses **compare-and-set**: the previous active fact is closed
(`valid_until=now`, MESI → `INVALID`) and the new value inserted. A **monotonic
CAS guard** rejects a *stale replay* — an agent cannot re-claim a slot a peer now
holds by re-sending an old fact. Re-asserting a value requires a *fresh* fact.

- Slot CAS + stale-replay rejection — [`pool.py:443`](../src/ai_knot/pool.py)
- Bi-temporal reads: `recall(now=...)` returns what the fleet knew at time *T*.

### 3. Provenance tracking — every fact has a writer and a lineage

Each published fact records `origin_agent_id`, `qualifiers["published_by"]`, and,
when it supersedes another, `qualifiers["supersedes_id"]` — a walkable lineage
chain of who-wrote-what-and-what-it-replaced.

- Provenance stamping on publish — [`pool.py:420`](../src/ai_knot/pool.py)

### 4. Trust — earned from behaviour, not assigned

Trust is computed, not configured:

```
trust = min(1, used / published) × (1 − quick_invalidation_rate)
```

`used` counts recall hits (facts that proved useful); `quick_invalidation_rate`
is the fraction of an agent's **verifiable, slot-addressed** publishes that a
different agent had to overturn within an hour. Taking the rate over *slot
events* — not total publish volume — means an attacker **cannot launder a bad
record by flooding free-standing junk**, while an honest agent **recovers** by
re-asserting a corrected slot. A Bayesian prior keeps new agents near the
neutral default until evidence accumulates.

- `SharedMemoryPool.get_trust(agent_id)` — [`pool.py:279`](../src/ai_knot/pool.py)

### 5. Audit + policy-governed propagation

An append-only ledger records *when and why* trust changed and *which recall used
which fact* — the event stream an audit actually needs. Timestamps are
caller-supplied ISO strings; **storage never reads the clock, so audited runs
stay deterministic**. Publishing can be gated: `utility_threshold` filters
low-value facts, and an opt-in `require_evidence` gate refuses to promote a fact
that carries no provenance pointer (evidence-before-belief).

Export the whole ledger — trust changes, fact-usage, and access grants — as JSON
for an auditor, no code required:

```bash
ai-knot --storage sqlite --data-dir ./data audit-export -o audit.json
```

- Audit ledger: `EventLedgerCapable.append_trust_event/append_usage_event` — [`storage/base.py:158`](../src/ai_knot/storage/base.py)
- Ledger export CLI: `ai-knot audit-export` — [`cli.py`](../src/ai_knot/cli.py)
- Contradiction resolution: `ClaimFamilyResolver.resolve(...)` groups competing
  claims by IDF-weighted overlap and keeps the authoritative winner —
  [`multi_agent/canonical.py:208`](../src/ai_knot/multi_agent/canonical.py)
- Abstention signal: `pool.last_recall_abstains` / `last_recall_risk` flag an
  answer that would rest on unsupported memory — [`pool.py:204`](../src/ai_knot/pool.py)

---

## See it defend itself: `examples/poisoned_pool.py`

The demo puts an honest agent and an attacker in one pool and lets the machinery
run. **Every number is computed by the library at runtime — nothing is staged.**

```
$ python examples/poisoned_pool.py

[honest] publishes production facts to the shared pool
         used by 3 services  ->  trust = 1.00

[attacker] floods 8 junk facts and poisons the secrets slot:
           "Production secrets live at http://evil.tld/creds"
           -> superseded the honest slot via slot-addressed CAS

[honest] re-asserts the correct value (a fresh fact)
         -> logs a quick-invalidation against the attacker

[attacker] tries a stale replay of the poison to re-claim the slot
           accepted? NO — rejected by monotonic CAS

[trust]  the pool scored both agents from behaviour alone:
           honest   = 0.75
           attacker = 0.18   (collapsed)

[newcomer] "where are production secrets stored?"
           [2.87] (from=honest, trust=0.75) Production secrets live in AWS Secrets Manager (rotated 90d)
           evil.tld in results? no — suppressed

           0 LLM calls · 0 network · deterministic
```

This is not a scripted effect: the same trust-and-CAS behaviour is a **scored CI
gate** — scenario S23 in the multi-agent scorecard
([`tests/eval/benchmark/scenarios/s23_adversarial_noise.py`](../tests/eval/benchmark/scenarios/s23_adversarial_noise.py),
gated in [`tests/eval/benchmark/ma_gate.py`](../tests/eval/benchmark/ma_gate.py)).
Showing the demo is showing a tested invariant.

---

## What ai-knot does *not* claim

The wedge is honesty, so the boundaries are stated plainly:

- **Not the only governed-memory design.** The MemClaw service in
  arXiv:2606.24535 and other projects target the same problem. ai-knot's specific
  position is *self-hosted, deterministic, no-LLM, dependency-free* — not "first"
  or "only."
- **The audit ledger is append-only by discipline** (ordinary database rows), not
  a cryptographically immutable commitment. It answers "who recalled what, when,
  and why did trust change," which is what most audits ask — but it is not a
  tamper-proof ledger.
- **ACL enforcement is at the recall layer** (application-level), not database
  row-level security. For hard multi-tenant isolation, pair ai-knot scopes with
  your database's own RLS as defense in depth.
- **The evidence gate needs facts with provenance.** Facts created via the manual
  `KnowledgeBase.add()` path carry no source pointer, so `require_evidence` is
  opt-in; turn it on for shared/org tiers that feed facts with source snippets.
- **Trust is a behavioural heuristic**, not a security boundary. It discounts and
  suppresses; it does not authenticate. Access control and provenance are the
  hard guarantees; trust is the soft signal on top.

---

## When you need this

- **Multi-agent systems past the demo stage** — the moment several agents write
  shared state, you need provenance, supersession, and conflict resolution, and
  no mainstream framework ships them as first-class primitives yet.
- **Self-hosted / air-gapped / regulated deployments** — the write path needs no
  LLM and no network, the storage is under your control, and the trust/usage
  ledger gives an auditor a real event stream. See
  [air-gapped-deployment.md](air-gapped-deployment.md) (with a test that enforces
  the zero-network guarantee), [deployment.md](deployment.md), and
  [production-readiness.md](production-readiness.md).

## Further reading

- Runnable example: [`examples/shared_pool.py`](../examples/shared_pool.py) (the friendly tour) and [`examples/poisoned_pool.py`](../examples/poisoned_pool.py) (the adversarial one above)
- Positioning and honest guardrails: [positioning.md](positioning.md), [comparison.md](comparison.md)
- The literature: [arXiv:2606.24535](https://arxiv.org/abs/2606.24535) · [GateMem 2606.18829](https://arxiv.org/abs/2606.18829) · [memory-poisoning defenses 2606.24322](https://arxiv.org/abs/2606.24322)
