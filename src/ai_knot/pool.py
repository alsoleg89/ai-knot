"""SharedMemoryPool — shared memory pool for multi-agent knowledge exchange."""

from __future__ import annotations

import copy
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ai_knot._pool_helpers import _extract_claim_key
from ai_knot._pool_recall import _SHARED_NAMESPACE, _PoolRecallMixin
from ai_knot._query_intent import _RecallMeta
from ai_knot.multi_agent.canonical import ClaimFamilyResolver
from ai_knot.multi_agent.recall_service import SharedPoolRecallService
from ai_knot.multi_agent.router import QueryShapeRouter
from ai_knot.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from ai_knot.storage.base import AtomicUpdateCapable, StorageBackend, TemporalStorageCapable
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MESIState, SlotDelta

if TYPE_CHECKING:
    from ai_knot.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

_PROVENANCE_DISCOUNT = 0.8
# Facts superseded by a different agent within this window count as "quick invalidations".
_QUICK_INV_WINDOW_S = 3600.0  # 1 hour


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
    *quick_invalidation_rate* is the fraction of the agent's facts that were
    superseded by a different agent within ``_QUICK_INV_WINDOW_S`` seconds —
    a signal that the original data was stale or low-quality.

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

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage: StorageBackend = storage or YAMLStorage()
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
        self._query_router = QueryShapeRouter()

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

    def get_trust(self, agent_id: str) -> float:
        """Return the current auto-computed trust score for an agent.

        Formula:
            ``trust = min(1, used / published) × (1 − quick_inv_rate)``

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
        inv_penalty = quick_inv / published
        return max(0.1, quality * (1.0 - inv_penalty))

    def publish(
        self,
        agent_id: str,
        fact_ids: list[str],
        *,
        kb: KnowledgeBase,
        utility_threshold: float = 0.0,
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
            if f.id in id_set and f.state_confidence * f.importance >= utility_threshold
        ]

        if not to_publish:
            return []

        with self._publish_lock:
            return self._publish_locked(agent_id, to_publish)

    def _publish_locked(self, agent_id: str, to_publish: list[Fact]) -> list[Fact]:
        """Execute publish while holding ``_publish_lock``.  Called only from :meth:`publish`."""
        now = datetime.now(UTC)
        published: list[Fact] = []
        quick_inv_updates: dict[str, int] = {}

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
                    # If this fact was previously published (now INVALID after CAS by
                    # another agent), do not re-activate it.  Agents must create a new
                    # fact via add_structured to reassert a superseded claim.
                    if new_fact.id in existing_ids:
                        continue
                    # Slot-addressed CAS: close old version, insert new.
                    old.valid_until = now
                    old.mesi_state = MESIState.INVALID
                    new_fact.mesi_state = MESIState.MODIFIED
                    new_fact.version = next_version
                    next_version += 1
                    # Quick invalidation: a different agent superseded this slot within the window.
                    if old.origin_agent_id and old.origin_agent_id != agent_id:
                        age_s = (now - old.valid_from).total_seconds()
                        if age_s < _QUICK_INV_WINDOW_S:
                            quick_inv_updates[old.origin_agent_id] = (
                                quick_inv_updates.get(old.origin_agent_id, 0) + 1
                            )
                elif new_fact.id not in existing_ids:
                    new_fact.mesi_state = MESIState.SHARED
                    new_fact.version = next_version
                    next_version += 1
                else:
                    # ID already in pool and no slot key — skip duplicate.
                    continue

                shared.append(new_fact)
                published.append(new_fact)

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

        if published:
            self._publish_count[agent_id] = self._publish_count.get(agent_id, 0) + len(published)
            logger.info(
                "Agent '%s' published %d facts to shared pool",
                agent_id,
                len(published),
            )

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
