"""SharedMemoryPool — shared memory pool for multi-agent knowledge exchange."""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ai_knot._pool_helpers import _extract_claim_key
from ai_knot._pool_recall import _SHARED_NAMESPACE, _PoolRecallMixin
from ai_knot._query_intent import _RecallMeta
from ai_knot.multi_agent.canonical import ClaimFamilyResolver, SemanticConflictResolver
from ai_knot.multi_agent.recall_service import SharedPoolRecallService
from ai_knot.multi_agent.router import QueryShapeRouter
from ai_knot.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from ai_knot.storage.base import (
    ACLStoreCapable,
    AtomicUpdateCapable,
    EventLedgerCapable,
    PoolStatsCapable,
    StorageBackend,
    TemporalStorageCapable,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MESIState, SlotDelta

if TYPE_CHECKING:
    from ai_knot.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

_PROVENANCE_DISCOUNT = 0.8
# Facts superseded by a different agent within this window count as "quick invalidations".
_QUICK_INV_WINDOW_S = 3600.0  # 1 hour


def _has_evidence(fact: Fact) -> bool:
    """Whether *fact* carries a provenance pointer fit for the shared pool.

    Implements the evidence-before-belief invariant (Eywa / MemMachine): a fact
    promoted to shared state must trace to immutable evidence rather than being a
    free-floating assertion.  A fact qualifies when it has not been flagged
    unsupported by the faithfulness filter AND carries at least one source pointer
    — verbatim source text, a source snippet, or a source span.

    Note: facts created via the manual ``KnowledgeBase.add()`` path carry no source
    pointer, so the ``require_evidence`` gate on :meth:`SharedMemoryPool.publish`
    is opt-in (default off) to preserve existing publish behaviour; governed
    deployments turn it on for shared/org tiers.
    """
    if fact.supported is False:
        return False
    return bool(fact.source_verbatim or fact.source_snippets or fact.source_spans)


class SharedMemoryPool(_PoolRecallMixin):
    """Shared memory pool for multi-agent knowledge exchange.

    Provides a shared namespace (``__shared__``) where agents can publish
    facts for cross-agent retrieval. Each published fact retains its
    ``origin_agent_id`` for provenance tracking.

    Inspired by CommNet (Sukhbaatar et al., 2016): a shared communication
    channel with selective read access. Facts from other agents receive a
    provenance discount reflecting per-agent trust (Marsh 1994).

    Trust is computed automatically from observed behaviour:
        ``trust = min(1, used / published) × (1 − quick_invalidation_rate)``

    where *published* is the number of facts an agent has contributed,
    *used* is the total number of recall hits across all queries, and
    *quick_invalidation_rate* is the fraction of the agent's *verifiable*
    (slot-addressed) publish events that a different agent superseded within
    ``_QUICK_INV_WINDOW_S`` seconds — a signal that the original data was stale or
    low-quality.  Taking the rate over slot events (not total publish volume) means
    free-standing facts cannot launder a poor track record, while re-asserting a
    slot with a corrected value lets a penalised agent recover.

    Usage::

        pool = SharedMemoryPool(storage=SQLiteStorage("mem.db"))
        pool.register("devops_agent")
        pool.register("coding_agent")

        # DevOps agent publishes a fact
        pool.publish("devops_agent", [fact_id], kb=devops_kb)

        # Coding agent queries the shared pool
        results = pool.recall("what database?", "coding_agent", top_k=5)

    Args:
        storage: Backend used to persist the shared namespace.
    """

    # Auto-promotion: promote facts used by >= N distinct agents to "org" tier.
    _AUTO_PROMOTE_THRESHOLD: int = 3
    # Pool TTL: expire pool facts unused for this many seconds (30 days).
    _POOL_TTL_SECONDS: float = 30 * 24 * 3600.0
    # Tier score boost: multiplicative bonus for higher-tier facts in recall.
    _TIER_BOOST: dict[str, float] = {"private": 1.0, "pool": 1.0, "org": 1.05}

    def __init__(
        self,
        storage: StorageBackend | None = None,
        *,
        persist_stats: bool = False,
        semantic_resolver: SemanticConflictResolver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage: StorageBackend = storage or YAMLStorage()
        # Clock for audit-ledger timestamps and grant times; injectable so tests
        # are deterministic. Defaults to wall-clock UTC.
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._bm25 = BM25Retriever(skip_prf=True)
        self._dense = DenseRetriever()
        # Pool retrieval has a wider semantic gap than private KB recall:
        # queries use abstract terms while facts contain technical specifics.
        # Weight dense higher to bridge this gap (2:3 vs KnowledgeBase's 2:1).
        self._retriever = HybridRetriever(
            self._bm25, self._dense, bm25_weight=2.0, dense_weight=3.0
        )
        self._agents: set[str] = set()
        self._publish_count: dict[str, int] = {}
        self._used_count: dict[str, int] = {}
        self._quick_inv_count: dict[str, int] = {}
        # Count of verifiable (slot-addressed) publish events per agent — the
        # denominator of the quick-invalidation penalty.  Counting slot *events*
        # (not total publishes, and not distinct slot keys) does two things:
        #   * free-standing publishes do not count → an agent cannot launder a poor
        #     track record by flooding the pool with unverifiable facts;
        #   * re-asserting a slot with a corrected value DOES count → an invalidated
        #     agent rehabilitates by making good verifiable claims that survive.
        # Stale replays rejected by the monotonic-CAS guard are never published, so
        # they never count here either.
        self._slot_publish_count: dict[str, int] = {}
        # MESI: per-agent high-water mark of versions pulled from shared pool.
        self._known_version: dict[str, int] = {}
        # Serialise concurrent publish() calls on the same pool instance.
        self._publish_lock = threading.Lock()
        # Internal recall metadata (not part of public API).
        self._last_recall_meta: _RecallMeta | None = None
        # Per-fact usage tracking: fact_id -> set of agent_ids that recalled it.
        self._fact_consumers: dict[str, set[str]] = {}
        # Tracks which fact IDs have been embedded to avoid re-embedding.
        self._embedded_ids: set[str] = set()
        # Transient query vector set by arecall() for hybrid search.
        self._query_vector: list[float] | None = None
        self._recall_service = SharedPoolRecallService()
        self._claim_resolver = ClaimFamilyResolver()
        # Opt-in semantic conflict resolver (e.g. LLM-backed) for value-conflicts
        # the deterministic resolver cannot detect.  Default None → deterministic,
        # dependency-free path; core never imports an LLM.
        self._semantic_resolver = semantic_resolver
        self._query_router = QueryShapeRouter()
        # Per-agent read-access grants for named visibility scopes (Collaborative Memory
        # access-control projection). agent_id -> set of scopes it may read. In-memory.
        self._read_scopes: dict[str, set[str]] = {}
        # Opt-in durable trust/usage telemetry. When enabled and the backend supports
        # it, social memory is restored on init and flushed after publish so it
        # survives a process restart. Default off → no new I/O for existing callers.
        self._persist_stats = persist_stats
        if persist_stats and isinstance(self._storage, PoolStatsCapable):
            self._restore_stats(self._storage.load_pool_stats())
        # Restore durable read-scope grants (ACL projection) when supported, so
        # access grants survive a process restart.
        if persist_stats and isinstance(self._storage, ACLStoreCapable):
            self._read_scopes = self._storage.load_grants()

    def register(self, agent_id: str) -> None:
        """Register an agent to participate in the shared pool.

        Args:
            agent_id: Unique identifier for the agent.
        """
        self._agents.add(agent_id)

    @property
    def agents(self) -> set[str]:
        """Return the set of registered agent IDs."""
        return set(self._agents)

    @property
    def read_scopes(self) -> dict[str, set[str]]:
        """Return a copy of the per-agent read-scope grants (ACL projection)."""
        return {agent_id: set(scopes) for agent_id, scopes in self._read_scopes.items()}

    def grant_read(self, agent_id: str, scope: str) -> None:
        """Grant *agent_id* read access to a named visibility scope.

        Facts published with ``visibility_scope=scope`` then become visible to this
        agent in :meth:`recall`, in addition to public (``"global"``) facts and the
        agent's own.  Scopes are matched verbatim; ``"global"`` needs no grant.
        Grants persist across restarts when the pool was created with
        ``persist_stats=True`` and the backend implements ``ACLStoreCapable``;
        otherwise they are in-memory (per pool instance).
        """
        self._read_scopes.setdefault(agent_id, set()).add(scope)
        if self._persist_stats and isinstance(self._storage, ACLStoreCapable):
            self._storage.save_grant(agent_id, scope, granted_at=self._clock().isoformat())

    @property
    def last_recall_abstains(self) -> bool:
        """Whether the most recent recall recommends abstaining (Synthius-Mem).

        True when the recall returned nothing, coverage was low, or no returned fact
        carried an evidence pointer — i.e. an answer would rest on unsupported memory.
        Returns False before any recall.  See :attr:`last_recall_risk` for the score.
        """
        meta = self._last_recall_meta
        return bool(meta and meta.should_abstain)

    @property
    def last_recall_risk(self) -> float:
        """Unsupported-answer risk in [0,1] from the most recent recall (0.0 if none)."""
        meta = self._last_recall_meta
        return meta.unsupported_answer_risk if meta else 0.0

    def _stats_snapshot(self) -> dict[str, object]:
        """Serializable snapshot of the pool's trust/usage telemetry."""
        return {
            "publish_count": dict(self._publish_count),
            "used_count": dict(self._used_count),
            "quick_inv_count": dict(self._quick_inv_count),
            "fact_consumers": {k: sorted(v) for k, v in self._fact_consumers.items()},
            "slot_publish_count": dict(self._slot_publish_count),
        }

    def _restore_stats(self, data: dict[str, object]) -> None:
        """Restore telemetry from a persisted snapshot (best-effort; ignores junk)."""
        pc = data.get("publish_count")
        if isinstance(pc, dict):
            self._publish_count = {str(k): int(v) for k, v in pc.items()}
        uc = data.get("used_count")
        if isinstance(uc, dict):
            self._used_count = {str(k): int(v) for k, v in uc.items()}
        qi = data.get("quick_inv_count")
        if isinstance(qi, dict):
            self._quick_inv_count = {str(k): int(v) for k, v in qi.items()}
        fc = data.get("fact_consumers")
        if isinstance(fc, dict):
            self._fact_consumers = {str(k): set(v) for k, v in fc.items()}
        sk = data.get("slot_publish_count")
        if isinstance(sk, dict):
            self._slot_publish_count = {str(k): int(v) for k, v in sk.items()}

    def flush_stats(self) -> None:
        """Persist the pool's trust/usage telemetry if persistence is enabled.

        No-op unless the pool was created with ``persist_stats=True`` and the
        storage backend implements :class:`PoolStatsCapable`.  ``publish()`` calls
        this automatically; call it directly to also capture recall-derived
        ``used_count`` increments between publishes.
        """
        if self._persist_stats and isinstance(self._storage, PoolStatsCapable):
            self._storage.save_pool_stats(self._stats_snapshot())

    def _log_trust_event(
        self, agent_id: str, event_type: str, delta: float, reason: str = ""
    ) -> None:
        """Append a trust-change event to the durable audit ledger.

        No-op unless the pool was created with ``persist_stats=True`` and the
        backend implements :class:`EventLedgerCapable`.  Unlike the aggregate
        snapshot, this records *when and why* trust changed (publish / quick
        invalidation), the event stream an audit needs.
        """
        if self._persist_stats and isinstance(self._storage, EventLedgerCapable):
            self._storage.append_trust_event(
                ts=self._clock().isoformat(),
                agent_id=agent_id,
                event_type=event_type,
                delta=delta,
                reason=reason,
            )

    def get_trust(self, agent_id: str) -> float:
        """Return the current auto-computed trust score for an agent.

        Formula:
            ``trust = min(1, used / published) × (1 − quick_inv_rate)``
        where ``quick_inv_rate = quick_inv / verifiable_slot_publishes``.  Taking the
        rate over verifiable (slot-addressed) publish events — not total publish
        volume — means free-standing facts cannot dilute the penalty, while
        re-asserting a slot with a corrected value lets a penalised agent recover.

        Falls back to ``_PROVENANCE_DISCOUNT`` when the agent has not yet
        published any facts (no track record).

        Args:
            agent_id: The agent to query.

        Returns:
            Trust score in [0.1, 1.0].
        """
        published = self._publish_count.get(agent_id, 0)
        if published == 0:
            return _PROVENANCE_DISCOUNT
        used = self._used_count.get(agent_id, 0)
        quick_inv = self._quick_inv_count.get(agent_id, 0)
        # Use Bayesian prior: start at _PROVENANCE_DISCOUNT and adjust toward
        # observed quality as evidence accumulates.  Without this, agents with
        # published facts but no retrievals yet get trust=0.1 (the floor),
        # which is far below the old flat 0.8 default and distorts ranking.
        _PRIOR_WEIGHT = 3  # pseudocount — ~3 observations before prior fades
        quality = min(
            1.0, (used + _PRIOR_WEIGHT * _PROVENANCE_DISCOUNT) / (published + _PRIOR_WEIGHT)
        )
        # The quick-invalidation penalty is the rate of peer overturns over the
        # agent's VERIFIABLE (slot-addressed) publish events — not over total publish
        # volume.  This makes trust resistant to reputation-laundering (free-standing
        # spam does not lower the rate) while still allowing recovery: re-asserting a
        # slot with a corrected value adds a verifiable event and dilutes the penalty.
        verifiable = self._slot_publish_count.get(agent_id, 0)
        inv_penalty = min(1.0, quick_inv / verifiable) if verifiable else 0.0
        return max(0.1, quality * (1.0 - inv_penalty))

    def publish(
        self,
        agent_id: str,
        fact_ids: list[str],
        *,
        kb: KnowledgeBase,
        utility_threshold: float = 0.0,
        require_evidence: bool = False,
        visibility_scope: str | None = None,
    ) -> list[Fact]:
        """Copy facts from an agent's private KB into the shared pool.

        Uses slot-addressed CAS: for each fact with a ``slot_key``, the
        existing active fact for that slot is closed (valid_until=now,
        mesi_state='I') and the new fact is inserted (mesi_state='M' if
        replacing, 'S' if new). Facts with no slot fall back to ID-based dedup.

        An optional publish gate filters facts by utility score before
        inserting:
            ``utility = state_confidence × importance``
        Only facts with ``utility >= utility_threshold`` are published.
        Threshold 0.0 (default) = no gating.

        Args:
            agent_id: The agent publishing the facts.
            fact_ids: IDs of facts to publish from the agent's KB.
            kb: The agent's KnowledgeBase instance.
            utility_threshold: Minimum utility score (0.0–1.0) to publish.
                Facts below the threshold are silently skipped.
            require_evidence: When True, apply the evidence-before-belief invariant
                — only facts carrying a provenance pointer (verbatim/snippet/span)
                and not flagged unsupported enter the pool (see :func:`_has_evidence`).
                Default False preserves the legacy behaviour (``add()``-created facts
                carry no source pointer); governed deployments enable it for shared/org.

        Returns:
            List of facts that were published (copies, not mutations of private KB).

        Raises:
            ValueError: If agent_id is not registered.

        Note:
            Thread-safe within a single process (protected by ``_publish_lock``).
            Cross-process atomicity is not guaranteed — concurrent writers from
            separate processes can interleave; use an external advisory lock if
            cross-process CAS semantics are required.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id!r} is not registered. Call register() first.")

        private_facts = kb.list_facts()
        id_set = set(fact_ids)
        to_publish = [
            f
            for f in private_facts
            if f.id in id_set
            and f.state_confidence * f.importance >= utility_threshold
            and (not require_evidence or _has_evidence(f))
        ]

        if not to_publish:
            return []

        with self._publish_lock:
            return self._publish_locked(agent_id, to_publish, visibility_scope)

    def _publish_locked(
        self, agent_id: str, to_publish: list[Fact], visibility_scope: str | None = None
    ) -> list[Fact]:
        """Execute publish while holding ``_publish_lock``.  Called only from :meth:`publish`."""
        now = datetime.now(UTC)
        published: list[Fact] = []
        quick_inv_updates: dict[str, int] = {}
        # Verifiable (slot-addressed) publish events the agent makes in this call —
        # applied to the slot-publish ledger after the storage transaction (mirrors the
        # quick_inv_updates pattern so atomic_update retries stay side-effect-free).
        slot_events: list[str] = []

        def _merge(shared: list[Fact]) -> list[Fact]:
            # Index active shared facts by slot_key for O(1) CAS lookup.
            # Falls back to (entity, attribute) for pre-Phase-3 facts without slot_key.
            active_by_slot: dict[str, Fact] = {}
            existing_ids: set[str] = set()
            for f in shared:
                existing_ids.add(f.id)
                if f.is_active(now):
                    if f.slot_key:
                        active_by_slot[f.slot_key] = f
                    elif f.entity and f.attribute:
                        legacy_key = f"{f.entity.lower().strip()}::{f.attribute.lower().strip()}"
                        active_by_slot.setdefault(legacy_key, f)

            # Global monotonic version counter for the pool.
            # Each published or superseded fact gets a strictly increasing version
            # so that load_since_version() and sync_slot_deltas() work correctly
            # when multiple facts are published in separate batches.
            next_version = max((f.version for f in shared), default=0) + 1

            for fact in to_publish:
                new_fact = copy.deepcopy(fact)
                new_fact.origin_agent_id = agent_id
                new_fact.visibility = "pool"
                new_fact.memory_tier = "pool"
                new_fact.valid_from = now
                new_fact.valid_until = None
                # Provenance lineage (persisted via qualifiers): who published this fact.
                new_fact.qualifiers["published_by"] = agent_id
                # Optional publish-time scope override; otherwise the fact keeps the
                # visibility_scope assigned at add() time.
                if visibility_scope is not None:
                    new_fact.visibility_scope = visibility_scope

                # For unslotted facts, extract a claim fingerprint for conflict detection.
                if not new_fact.slot_key and not new_fact.claim_key:
                    new_fact.claim_key = _extract_claim_key(new_fact.content)

                # Resolve CAS key: prefer slot_key, fall back to entity+attribute.
                cas_key = fact.slot_key or (
                    f"{fact.entity.lower().strip()}::{fact.attribute.lower().strip()}"
                    if fact.entity and fact.attribute
                    else ""
                )

                if cas_key and cas_key in active_by_slot:
                    old = active_by_slot[cas_key]
                    if old.id == fact.id:
                        # Same fact already published as the active version — no-op.
                        continue
                    # Monotonic CAS: reject a stale replay.  If this incoming fact was
                    # already published and then superseded (its id is in the pool as an
                    # inactive copy), it must not re-claim a slot a *different* agent now
                    # holds.  Otherwise an agent could undo a peer's legitimate update —
                    # or re-poison a corrected slot — simply by re-sending its old fact.
                    # Re-asserting a value requires a fresh fact, not a replay.
                    if old.origin_agent_id != agent_id and new_fact.id in existing_ids:
                        prior = next((f for f in shared if f.id == new_fact.id), None)
                        if prior is not None and not prior.is_active(now):
                            continue
                    # Slot-addressed CAS: close old version, insert new.
                    old.valid_until = now
                    old.mesi_state = MESIState.INVALID
                    new_fact.mesi_state = MESIState.MODIFIED
                    new_fact.version = next_version
                    next_version += 1
                    # Provenance lineage: record the fact this one superseded via CAS.
                    new_fact.qualifiers["supersedes_id"] = old.id
                    # Quick invalidation: a different agent superseded this slot within the window.
                    if old.origin_agent_id and old.origin_agent_id != agent_id:
                        age_s = (now - old.valid_from).total_seconds()
                        if age_s < _QUICK_INV_WINDOW_S:
                            quick_inv_updates[old.origin_agent_id] = (
                                quick_inv_updates.get(old.origin_agent_id, 0) + 1
                            )
                    # If the fact was previously published (INVALID), remove the stale
                    # copy so that re-activating it via CAS does not create a duplicate.
                    if new_fact.id in existing_ids:
                        shared[:] = [f for f in shared if f.id != new_fact.id]
                elif new_fact.id not in existing_ids:
                    new_fact.mesi_state = MESIState.SHARED
                    new_fact.version = next_version
                    next_version += 1
                else:
                    # ID already in pool and no slot key — skip duplicate.
                    continue

                shared.append(new_fact)
                published.append(new_fact)
                # Record the verifiable (slot-addressed) publish event for trust
                # accounting.  Stale replays rejected above never reach this point.
                if cas_key:
                    slot_events.append(cas_key)

            return shared

        # AtomicUpdateCapable backends (SQLite) protect the full load→merge→save
        # cycle with an EXCLUSIVE transaction, preventing cross-process lost updates.
        # Other backends fall back to the in-process lock only.
        if isinstance(self._storage, AtomicUpdateCapable):
            self._storage.atomic_update(_SHARED_NAMESPACE, _merge)
        else:
            current = self._storage.load(_SHARED_NAMESPACE)
            _merge(current)
            if published:
                if isinstance(self._storage, TemporalStorageCapable):
                    self._storage.save_atomic(_SHARED_NAMESPACE, current)
                else:
                    self._storage.save(_SHARED_NAMESPACE, current)

        # Apply trust-tracking side effects outside the storage transaction.
        for agt, count in quick_inv_updates.items():
            self._quick_inv_count[agt] = self._quick_inv_count.get(agt, 0) + count
            self._log_trust_event(agt, "quick_invalidation", float(count), reason="slot superseded")
        if slot_events:
            self._slot_publish_count[agent_id] = self._slot_publish_count.get(agent_id, 0) + len(
                slot_events
            )

        if published:
            self._publish_count[agent_id] = self._publish_count.get(agent_id, 0) + len(published)
            self._log_trust_event(agent_id, "publish", float(len(published)))
            logger.info(
                "Agent '%s' published %d facts to shared pool",
                agent_id,
                len(published),
            )

        # Persist trust/usage telemetry (no-op unless persist_stats=True).
        self.flush_stats()
        return published

    def promote(self, agent_id: str, fact_ids: list[str], *, tier: str = "pool") -> int:
        """Manually promote pool facts to a higher memory tier.

        Only facts currently in the shared pool (origin_agent_id == agent_id)
        are eligible.  Returns the number of facts actually promoted.

        Args:
            agent_id: The agent requesting the promotion.
            fact_ids: IDs of pool facts to promote.
            tier: Target tier — ``"pool"`` (default) or ``"org"`` (future).

        Raises:
            ValueError: If *tier* is not a valid tier name.
        """
        if tier not in ("pool", "org"):
            raise ValueError(f"Invalid tier {tier!r}; must be 'pool' or 'org'.")
        shared = self._storage.load(_SHARED_NAMESPACE)
        promoted = 0
        for fact in shared:
            if fact.id in fact_ids and fact.is_active() and fact.origin_agent_id == agent_id:
                fact.memory_tier = tier
                fact.qualifiers["promoted_by"] = agent_id  # provenance lineage
                promoted += 1
        if promoted:
            self._storage.save(_SHARED_NAMESPACE, shared)
            logger.info(
                "Promoted %d facts from agent '%s' to tier '%s'",
                promoted,
                agent_id,
                tier,
            )
        return promoted

    def gc_pool(self, *, now: datetime | None = None) -> int:
        """Garbage-collect stale pool facts that exceed the pool TTL.

        Facts in the ``"org"`` tier are exempt from TTL (they are considered
        permanently promoted).  Only ``"pool"`` tier facts whose
        ``last_accessed`` is older than ``_POOL_TTL_SECONDS`` are expired.

        Returns:
            Number of facts expired.
        """
        now_dt = now or datetime.now(UTC)
        cutoff = now_dt - timedelta(seconds=self._POOL_TTL_SECONDS)
        shared = self._storage.load(_SHARED_NAMESPACE)
        expired = 0
        for fact in shared:
            if (
                fact.is_active(now_dt)
                and fact.memory_tier == "pool"
                and fact.last_accessed < cutoff
            ):
                fact.valid_until = now_dt
                expired += 1
        if expired:
            self._storage.save(_SHARED_NAMESPACE, shared)
            logger.info(
                "Pool GC: expired %d stale facts (TTL=%ds)", expired, self._POOL_TTL_SECONDS
            )
        return expired

    def sync_dirty(self, agent_id: str) -> list[Fact]:
        """Pull facts changed by other agents since the last sync (MESI lazy invalidation).

        Implements the Modified/Invalid state pull from MESI protocol.
        Token savings: ~95% vs broadcast when only a small subset of facts
        changes between syncs (arXiv 2603.15183).

        Uses ``TemporalStorageCapable.load_since_version()`` for index-accelerated
        queries on SQLite/Postgres; falls back to Python filtering on YAML.

        Args:
            agent_id: The agent requesting dirty facts.

        Returns:
            Facts changed by other agents since the last sync call for this agent.
        """
        since = self._known_version.get(agent_id, 0)

        if isinstance(self._storage, TemporalStorageCapable):
            dirty = self._storage.load_since_version(_SHARED_NAMESPACE, since, agent_id)
        else:
            now_dt = datetime.now(UTC)
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            dirty = [
                f
                for f in all_shared
                if f.version > since and f.origin_agent_id != agent_id and f.is_active(now_dt)
            ]

        if dirty:
            self._known_version[agent_id] = max(f.version for f in dirty)
        return dirty

    def sync_slot_deltas(self, agent_id: str) -> list[SlotDelta]:
        """Pull slot-level changes since the last sync as lightweight ``SlotDelta`` records.

        Semantically equivalent to :meth:`sync_dirty` but returns compact
        ``SlotDelta`` objects instead of full ``Fact`` copies.  Each delta
        carries only ``slot_key``, ``version``, ``op``, ``fact_id``,
        ``content``, and ``prompt_surface`` — roughly 10–30× less data than
        a full ``Fact`` for a typical memory entry.

        On SQLite/Postgres backends the query is index-accelerated; on YAML
        backends it falls back to Python-level filtering.

        Args:
            agent_id: The agent requesting delta records.

        Returns:
            ``SlotDelta`` records for slot changes by other agents since the
            last call for this agent.  The per-agent high-water mark is
            updated on each call.
        """
        since = self._known_version.get(agent_id, 0)

        if isinstance(self._storage, TemporalStorageCapable):
            deltas = self._storage.load_slot_deltas_since(_SHARED_NAMESPACE, since, agent_id)
        else:
            # Python-level fallback for YAML backends.
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            deltas = []
            for f in all_shared:
                if f.version <= since or f.origin_agent_id == agent_id:
                    continue
                if f.valid_until is not None:
                    op = "invalidate"
                elif f.mesi_state == MESIState.MODIFIED:
                    op = "supersede"
                else:
                    op = "new"
                deltas.append(
                    SlotDelta(
                        slot_key=f.slot_key,
                        version=f.version,
                        op=op,
                        fact_id=f.id,
                        content=f.content,
                        prompt_surface=f.prompt_surface,
                    )
                )

        if deltas:
            self._known_version[agent_id] = max(d.version for d in deltas)
        return deltas

    def list_shared_facts(self) -> list[Fact]:
        """Return all facts in the shared pool.

        Returns:
            List of all shared Facts (including closed/invalidated).
        """
        return self._storage.load(_SHARED_NAMESPACE)
