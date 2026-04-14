"""Learning pipeline mixin for KnowledgeBase."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from ai_knot.extractor import (
    Extractor,
    resolve_against_existing,
    resolve_by_slot,
    resolve_structured,
)
from ai_knot.providers import LLMProvider
from ai_knot.storage.base import StorageBackend
from ai_knot.types import (
    CONFLICT_POLICIES,
    ConversationTurn,
    Fact,
    MemoryOp,
    MemoryType,
)

logger = logging.getLogger(__name__)

_LEARN_DEBUG = False  # overridden by KnowledgeBase after import


class _LearningMixin:
    """Mixin providing learn/alearn/learn_async/consolidate_episodic to KnowledgeBase.

    Depends on the following attributes being set by KnowledgeBase.__init__:
        _agent_id, _storage, _default_provider, _default_api_key, _default_model,
        _default_provider_kwargs, _episodic_ttl_hours, _embed_url, _embed_model
    """

    # These are declared here only for type-checker awareness; they are set by
    # KnowledgeBase.__init__ at runtime.
    _agent_id: str
    _storage: StorageBackend
    _default_provider: str | LLMProvider | None
    _default_api_key: str | None
    _default_model: str | None
    _default_provider_kwargs: dict[str, str]
    _episodic_ttl_hours: float
    _embed_url: str
    _embed_model: str
    _embed_api_key: str | None

    def learn(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Extract and store facts from a conversation using an LLM.

        Resolution uses a three-phase pipeline:

        1. **Slot-based** (deterministic): facts with ``slot_key`` are matched
           against existing active facts by exact ``slot_key`` equality.
           Same slot + same value → *reinforce* (bump confidence, no insert).
           Same slot + new value → *supersede* (temporal close + versioned insert).
           No slot match → *branch* (insert as new).

        2. **Entity-addressed CAS** (fuzzy, backward compat): unslotted facts
           with ``entity`` + ``attribute`` are matched via ``resolve_structured``
           (Jaccard entity matching) to close superseded pre-Phase-1 facts.

        3. **Lexical dedup**: remaining unslotted facts are checked against active
           facts with ``resolve_against_existing`` (combined Jaccard + containment).

        Provider credentials default to those passed at :meth:`__init__` when
        not specified per-call.

        Args:
            turns: Conversation messages to extract knowledge from.
            api_key: LLM API key. Falls back to the value set at init, then to
                environment variables.
            provider: Provider name or a pre-configured ``LLMProvider`` instance.
                Falls back to the value set at init, then to ``"openai"``.
                Supported names: openai, anthropic, gigachat, yandex, qwen,
                openai-compat.
            model: Override the default model for this provider. Falls back to
                the value set at init.
            conflict_threshold: Jaccard similarity threshold for the lexical
                dedup pass (phase 3). Does not affect slot-based resolution.
            timeout: Per-request timeout in seconds for LLM calls. ``None``
                uses the provider's built-in default (30 s).
            batch_size: Maximum conversation turns sent per LLM call. Longer
                conversations are split into batches to prevent JSON truncation.
            **provider_kwargs: Extra args forwarded to the provider constructor
                (e.g. ``folder_id`` for Yandex, ``base_url`` for openai-compat).
                Merged with any defaults set at init, with per-call values taking
                precedence.

        Returns:
            List of inserted Facts (new inserts + versioned replacements).
            Reinforced facts (same value) are excluded from the return value.
        """
        if not turns:
            return []

        # Stage 1: extract raw candidate facts from LLM.
        candidates = self._extract_phase(
            turns,
            provider=provider,
            api_key=api_key,
            model=model,
            timeout=timeout,
            batch_size=batch_size,
            **provider_kwargs,
        )
        if not candidates:
            return []

        # Stage 2: candidate verification (dedup, future ATC checks).
        verified = self._candidate_phase(candidates)

        # Stage 3: resolve against existing knowledge.
        existing = self._storage.load(self._agent_id)
        to_insert = self._resolve_phase(verified, existing, conflict_threshold)

        # Stage 3.5: create/update entity-level aggregate facts.
        aggregates = self._consolidate_phase(to_insert, existing)
        to_insert.extend(aggregates)

        # Stage 4: commit to storage.
        self._commit_phase(existing, to_insert)
        return to_insert

    # ------------------------------------------------------------------
    # learn() pipeline stages
    # ------------------------------------------------------------------

    def _extract_phase(
        self,
        turns: list[ConversationTurn],
        *,
        provider: str | LLMProvider | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Stage 1: extract raw candidate facts from an LLM."""
        resolved_provider = provider or self._default_provider or "openai"
        resolved_api_key = api_key or self._default_api_key
        resolved_model = model or self._default_model
        resolved_kwargs: dict[str, str] = {**self._default_provider_kwargs, **provider_kwargs}

        if isinstance(resolved_provider, str):
            if not resolved_api_key:
                import os

                resolved_api_key = os.environ.get(
                    {
                        "openai": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY",
                        "gigachat": "GIGACHAT_API_KEY",
                        "yandex": "YANDEX_API_KEY",
                        "qwen": "QWEN_API_KEY",
                    }.get(resolved_provider, "LLM_API_KEY"),
                    "",
                )
            if not resolved_api_key:
                raise ValueError(
                    f"No API key for provider {resolved_provider!r}. "
                    "Pass api_key= or set the environment variable "
                    f"(e.g. OPENAI_API_KEY for openai, ANTHROPIC_API_KEY for anthropic)."
                )
            extractor = Extractor(
                resolved_provider,
                api_key=resolved_api_key,
                model=resolved_model,
                timeout=timeout,
                batch_size=batch_size,
                **resolved_kwargs,
            )
        else:
            extractor = Extractor(
                resolved_provider,
                model=resolved_model,
                timeout=timeout,
                batch_size=batch_size,
            )

        facts = extractor.extract(turns)
        if _LEARN_DEBUG:
            logger.debug(
                "learn/_extract_phase: %d candidates from %d turns",
                len(facts),
                len(turns),
            )
        return facts

    def _candidate_phase(self, candidates: list[Fact]) -> list[Fact]:
        """Stage 2: verify and deduplicate candidate facts.

        Currently a pass-through. Future: ATC (Assertion-Truth-Checking),
        source grounding, confidence calibration.
        """
        if _LEARN_DEBUG:
            logger.debug("learn/_candidate_phase: %d candidates (pass-through)", len(candidates))
        return candidates

    def _resolve_phase(
        self,
        verified: list[Fact],
        existing: list[Fact],
        conflict_threshold: float,
    ) -> list[Fact]:
        """Stage 3: resolve new facts against existing knowledge.

        Three-phase resolution:
        1. Slot-based CAS (deterministic, exact slot_key match).
        2. Entity-addressed CAS (fuzzy, for pre-Phase-1 facts).
        3. Lexical dedup (Jaccard + containment).

        Returns:
            Facts to insert (new + versioned replacements).
            Mutates ``existing`` in place (closing superseded facts).
        """
        now_close = datetime.now(UTC)
        active_existing = [f for f in existing if f.is_active(now_close)]

        to_insert: list[Fact] = []
        handled_ids: set[str] = set()
        n_reinforce = n_supersede = n_branch = n_delete = n_noop = 0

        # Phase 1: slot-based resolution (deterministic, exact slot_key match).
        # Consults the per-type ConflictPolicy to decide supersession behaviour.
        slotted_facts = [f for f in verified if f.slot_key]
        for new_fact in slotted_facts:
            if new_fact.op == MemoryOp.NOOP:
                n_noop += 1
                continue

            policy = CONFLICT_POLICIES.get(new_fact.type, CONFLICT_POLICIES[MemoryType.SEMANTIC])

            if new_fact.op == MemoryOp.DELETE:
                unhandled = [f for f in active_existing if f.id not in handled_ids]
                _, matched = resolve_by_slot(new_fact, unhandled)
                if matched is not None:
                    matched.valid_until = now_close
                    handled_ids.add(matched.id)
                n_delete += 1
                continue

            slot_op, matched = resolve_by_slot(new_fact, active_existing)
            if new_fact.op == MemoryOp.UPDATE and slot_op == "reinforce":
                slot_op = "supersede"

            # Policy override: if the policy says don't supersede, branch instead.
            # Explicit UPDATE ops bypass the policy (user intent is authoritative).
            if (
                slot_op == "supersede"
                and matched is not None
                and new_fact.op != MemoryOp.UPDATE
                and not policy.should_supersede(new_fact, matched)
            ):
                slot_op = "branch"

            if slot_op == "reinforce":
                assert matched is not None
                matched.state_confidence = min(1.0, matched.state_confidence + 0.05)
                matched.importance = min(1.0, matched.importance + 0.02)
                matched.last_accessed = now_close
                # Accumulate evidence snippets from the new observation.
                if new_fact.source_snippets:
                    existing_snips = set(matched.source_snippets)
                    for s in new_fact.source_snippets:
                        if s not in existing_snips:
                            matched.source_snippets.append(s)
                            existing_snips.add(s)
                    matched.source_snippets = matched.source_snippets[:5]
                handled_ids.add(matched.id)
                n_reinforce += 1
            elif slot_op == "supersede":
                assert matched is not None
                matched.valid_until = now_close
                handled_ids.add(matched.id)
                new_fact.importance = min(1.0, matched.importance + 0.05)
                new_fact.version = matched.version + 1
                # Carry over evidence trail from the old fact.
                if matched.source_snippets:
                    existing_snips = set(new_fact.source_snippets)
                    carried = [s for s in matched.source_snippets if s not in existing_snips]
                    new_fact.source_snippets = (new_fact.source_snippets + carried)[:5]
                to_insert.append(new_fact)
                n_supersede += 1
            else:  # branch — new slot, insert as-is
                to_insert.append(new_fact)
                n_branch += 1

        # Phase 2: entity-addressed CAS for unslotted facts with entity+attribute.
        unslotted_facts = [f for f in verified if not f.slot_key and f.op != MemoryOp.NOOP]
        unslotted_with_entity = [f for f in unslotted_facts if f.entity and f.attribute]
        entity_candidates = [f for f in active_existing if f.id not in handled_ids]
        for new_fact in unslotted_with_entity:
            available = [f for f in entity_candidates if f.id not in handled_ids]
            matched_fact = resolve_structured(new_fact, available)
            if matched_fact is not None:
                matched_fact.valid_until = now_close
                handled_ids.add(matched_fact.id)
                # Carry over evidence trail from the old entity fact.
                if matched_fact.source_snippets and new_fact.op != MemoryOp.DELETE:
                    existing_snips = set(new_fact.source_snippets)
                    carried = [s for s in matched_fact.source_snippets if s not in existing_snips]
                    new_fact.source_snippets = (new_fact.source_snippets + carried)[:5]
            if new_fact.op == MemoryOp.DELETE:
                n_delete += 1

        # Phase 3: lexical dedup for remaining unslotted facts.
        remaining_active = [f for f in entity_candidates if f.id not in handled_ids]
        unslotted_to_insert = [f for f in unslotted_facts if f.op != MemoryOp.DELETE]
        unslotted_inserted, _ = resolve_against_existing(
            unslotted_to_insert, remaining_active, threshold=conflict_threshold
        )
        to_insert.extend(unslotted_inserted)

        if _LEARN_DEBUG:
            logger.debug(
                "learn/_resolve_phase: slot=%d(r=%d s=%d b=%d d=%d n=%d) "
                "entity=%d lexical=%d→%d insert=%d",
                len(slotted_facts),
                n_reinforce,
                n_supersede,
                n_branch,
                n_delete,
                n_noop,
                len(unslotted_with_entity),
                len(unslotted_to_insert),
                len(unslotted_inserted),
                len(to_insert),
            )
        return to_insert

    def _consolidate_phase(
        self,
        to_insert: list[Fact],
        existing: list[Fact],
    ) -> list[Fact]:
        """Stage 3.5: create/update entity-level aggregate facts.

        Groups extracted facts by entity and creates a single keyword-dense
        aggregate fact per entity (when >= 2 facts share the same entity).
        This ensures that queries like "tell me about X" or "what do you know
        about the user" can surface all known values for an entity with a single
        BM25 hit instead of relying on top-k to cover every individual slot.

        Aggregate format is intentionally compact (values only, no attr= prefixes)
        to avoid BM25 length normalization penalising the aggregate vs atomic facts.

        Args:
            to_insert: New facts being added this learn() call.
            existing: All previously stored facts for this agent.

        Returns:
            List of aggregate facts to append to the commit batch.
            May include supersession markers on existing aggregates.
        """
        now = datetime.now(UTC)

        # Build a unified view: active existing facts + incoming new facts.
        active_existing = [f for f in existing if f.is_active(now)]
        all_active = active_existing + to_insert

        # Group by entity — skip facts without entity field or with _aggregate attribute
        # (prevent aggregate-on-aggregate chaining).
        entity_groups: dict[str, list[Fact]] = defaultdict(list)
        for f in all_active:
            if f.entity and f.attribute and not f.attribute.startswith("_aggregate"):
                entity_groups[f.entity.lower()].append(f)

        aggregates: list[Fact] = []
        for entity, group in entity_groups.items():
            if len(group) < 2:
                continue

            # Sort by importance descending, cap at 30 pairs.
            group_sorted = sorted(group, key=lambda f: f.importance, reverse=True)[:30]

            # Build compact content: "entity: val1, val2, val3, ..."
            values = [f.value_text.strip() or f.content for f in group_sorted]
            content = f"{entity}: {', '.join(v for v in values if v)}"
            canonical = f"{entity} {' '.join(v for v in values if v)}"
            slot_key = f"{entity}::_agg"

            # CAS against existing aggregate with the same slot_key.
            existing_agg = next(
                (f for f in active_existing if f.slot_key == slot_key),
                None,
            )
            if existing_agg is not None:
                if existing_agg.content == content:
                    continue  # identical — nothing to do
                existing_agg.valid_until = now  # supersede stale aggregate

            agg = Fact(
                content=content,
                type=MemoryType.SEMANTIC,
                importance=max(f.importance for f in group_sorted),
                entity=entity,
                attribute="_aggregate",
                slot_key=slot_key,
                canonical_surface=canonical,
                tags=["aggregate"],
                verification_source="aggregate",
            )
            aggregates.append(agg)

        if _LEARN_DEBUG:
            logger.debug(
                "learn/_consolidate_phase: %d aggregates for %d entities",
                len(aggregates),
                len(entity_groups),
            )
        return aggregates

    def _commit_phase(self, existing: list[Fact], to_insert: list[Fact]) -> None:
        """Stage 4: persist resolved facts to storage."""
        self._storage.save(self._agent_id, existing + to_insert)
        if _LEARN_DEBUG:
            logger.debug(
                "learn/_commit_phase: saved %d total facts (%d new) for '%s'",
                len(existing) + len(to_insert),
                len(to_insert),
                self._agent_id,
            )
        logger.info(
            "Learned %d facts for agent '%s'",
            len(to_insert),
            self._agent_id,
        )

    async def alearn(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Async variant of :meth:`learn` — non-blocking for asyncio applications.

        Runs ``learn()`` in a thread-pool executor so the event loop is never
        blocked during the LLM HTTP call.  All parameters are identical to
        :meth:`learn`.

        Example::

            # FastAPI handler — does not block the event loop
            facts = await kb.alearn(turns, provider="openai", api_key="sk-...")

            # Concurrent extraction for multiple agents
            results = await asyncio.gather(
                kb_a.alearn(turns_a, ...),
                kb_b.alearn(turns_b, ...),
            )
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.learn(
                turns,
                api_key=api_key,
                provider=provider,
                model=model,
                conflict_threshold=conflict_threshold,
                timeout=timeout,
                batch_size=batch_size,
                **provider_kwargs,
            ),
        )

    async def learn_async(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        embed_api_key: str | None = None,
        semantic_threshold: float = 0.80,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Like :meth:`alearn` but adds a semantic deduplication pass after extraction.

        After the standard LLM extraction + lexical dedup (``resolve_against_existing``),
        a second pass embeds remaining new facts alongside existing facts and merges
        any pair with cosine similarity ≥ ``semantic_threshold``.  This detects
        topic-evolution updates (e.g. "I use Airflow" followed by "I switched to
        Prefect") that share almost no tokens but are semantically near-duplicate.

        On merge the *newer* fact's content replaces the existing one (temporal
        consolidation).  Importance is bumped by ``+0.05`` (clamped to 1.0).

        Graceful degradation: if the Ollama embedding endpoint is unreachable the
        semantic pass is silently skipped and behaviour is identical to
        :meth:`alearn`.

        Args:
            turns: Conversation messages to extract knowledge from.
            embed_url: Base URL of the Ollama server for embeddings.
            embed_model: Embedding model name (must support /v1/embeddings).
            semantic_threshold: Cosine similarity above which two facts are
                considered the same topic (0.0–1.0, default 0.82).
            All other args: same as :meth:`learn`.
        """
        from ai_knot.embedder import cosine, embed_texts

        loop = asyncio.get_running_loop()
        new_facts: list[Fact] = await loop.run_in_executor(
            None,
            lambda: self.learn(
                turns,
                api_key=api_key,
                provider=provider,
                model=model,
                conflict_threshold=conflict_threshold,
                timeout=timeout,
                batch_size=batch_size,
                **provider_kwargs,
            ),
        )

        if not new_facts:
            return new_facts

        existing = await loop.run_in_executor(None, lambda: self._storage.load(self._agent_id))
        if not existing:
            return new_facts

        # Build a set of already-inserted new fact IDs to avoid comparing against
        # themselves (learn() already saved them to storage).
        new_ids = {f.id for f in new_facts}
        prior_facts = [f for f in existing if f.id not in new_ids]
        if not prior_facts:
            return new_facts

        effective_embed_api_key = embed_api_key or getattr(self, "_embed_api_key", None)
        new_texts = [f.content for f in new_facts]
        all_embeddings = await embed_texts(
            new_texts + [f.content for f in prior_facts],
            base_url=embed_url,
            model=embed_model,
            api_key=effective_embed_api_key,
        )

        if not all_embeddings:
            # Ollama unavailable — graceful degradation.
            return new_facts

        new_embs = all_embeddings[: len(new_facts)]
        prior_embs = all_embeddings[len(new_facts) :]

        # For each new fact find its nearest prior fact.  If above threshold,
        # close the old fact (set valid_until = now) and keep the new fact as
        # the current version — proper temporal versioning instead of mutation.
        closed_ids: set[str] = set()
        updated_prior: list[Fact] = []
        for _new_fact, new_emb in zip(new_facts, new_embs, strict=True):
            updated_prior_ids = {p.id for p in updated_prior}
            best_score = 0.0
            best_prior: Fact | None = None
            for prior_fact, prior_emb in zip(prior_facts, prior_embs, strict=True):
                if prior_fact.id in updated_prior_ids:
                    continue  # already merged into this slot — don't double-merge
                score = cosine(new_emb, prior_emb)
                if score > best_score:
                    best_score = score
                    best_prior = prior_fact

            if best_prior is not None and best_score >= semantic_threshold:
                # Temporal close: mark prior fact as superseded instead of mutating content.
                best_prior.valid_until = datetime.now(UTC)
                best_prior.importance = min(1.0, best_prior.importance + 0.05)
                updated_prior.append(best_prior)
                closed_ids.add(best_prior.id)

        if closed_ids:
            await loop.run_in_executor(None, lambda: self._storage.save(self._agent_id, existing))
            logger.info(
                "Temporal consolidation closed %d prior fact(s) for agent '%s'",
                len(closed_ids),
                self._agent_id,
            )

        return new_facts

    async def consolidate_episodic(
        self,
        *,
        older_than_hours: float = 24.0,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        embed_api_key: str | None = None,
        semantic_threshold: float = 0.80,
    ) -> int:
        """Promote episodic facts to semantic memory (async "sleep consolidation").

        Finds episodic facts older than ``older_than_hours``, runs LLM extraction
        to produce structured semantic facts, deduplicates against existing
        semantic memory, and marks episodic facts as consolidated (valid_until=now).

        Inspired by CLS theory (McClelland et al. 1995): hippocampal → neocortical
        transfer happens offline (during "sleep"), not during encoding.

        Args:
            older_than_hours: Only consolidate episodic facts created more than
                this many hours ago (avoids consolidating too-recent episodes).
            embed_url: Ollama base URL for semantic dedup embeddings.
            embed_model: Embedding model for dedup pass.
            semantic_threshold: Cosine threshold for semantic dedup.

        Returns:
            Number of new semantic facts created from consolidation.
        """
        if not self._default_provider:
            logger.warning(
                "consolidate_episodic() requires a provider; "
                "configure KnowledgeBase with provider= to enable this."
            )
            return 0

        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=older_than_hours)

        all_facts = self._storage.load(self._agent_id)
        to_consolidate = [
            f
            for f in all_facts
            if f.type == MemoryType.EPISODIC and f.is_active(now) and f.created_at <= cutoff
        ]

        if not to_consolidate:
            return 0

        # Run LLM extraction on the episodic batch (treat as conversation turns)
        turns = [
            ConversationTurn(role="user", content=f.source_verbatim or f.content)
            for f in to_consolidate
        ]
        new_semantic = await self.learn_async(
            turns,
            embed_url=embed_url,
            embed_model=embed_model,
            embed_api_key=embed_api_key,
            semantic_threshold=semantic_threshold,
        )

        # Reload after learn_async() (which internally saved new semantic facts)
        # to avoid overwriting them with the stale snapshot.
        all_facts = self._storage.load(self._agent_id)
        consolidated_ids = {f.id for f in to_consolidate}
        for fact in all_facts:
            if fact.id in consolidated_ids:
                fact.valid_until = now

        self._storage.save(self._agent_id, all_facts)
        logger.info(
            "Consolidated %d episodic facts → %d new semantic facts for agent '%s'",
            len(to_consolidate),
            len(new_semantic),
            self._agent_id,
        )
        return len(new_semantic)
