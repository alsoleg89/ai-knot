"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import ai_knot._spreading_activation as _ddsa
from ai_knot._bm25 import _rrf_fuse
from ai_knot._date_enrichment import enrich_date_tags
from ai_knot._inverted_index import InvertedIndex, _char_trigrams, _slot_exact_score
from ai_knot._query_intent import classify_recall_intent, get_pipeline_config
from ai_knot._spreading_activation import spreading_activation
from ai_knot.extractor import Extractor as Extractor  # noqa: F401  re-exported for tests
from ai_knot.extractor import split_enumerations
from ai_knot.forgetting import apply_decay
from ai_knot.learning import _LearningMixin
from ai_knot.providers import LLMProvider, create_provider
from ai_knot.query_expander import LLMQueryExpander
from ai_knot.retriever import DenseRetriever, HybridRetriever, TFIDFRetriever
from ai_knot.storage.base import (
    SnapshotCapable,
    StorageBackend,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import (
    Fact,
    MemoryType,
    SnapshotDiff,
)

# LLM expansion token weight — higher than PRF (0.5) to reflect
# the semantic advantage of LLM-based query understanding.
_LLM_EXPANSION_WEIGHT: float = 0.6

_LEARN_DEBUG = bool(os.environ.get("AI_KNOT_LEARN_DEBUG", ""))

logger = logging.getLogger(__name__)

# Patch the _LEARN_DEBUG flag into the learning module so pipeline stages log correctly.
import ai_knot.learning as _learning_mod  # noqa: E402

_learning_mod._LEARN_DEBUG = _LEARN_DEBUG


class KnowledgeBase(_LearningMixin):
    """Agent knowledge store with extraction, retrieval, and forgetting.

    Usage::

        kb = KnowledgeBase(agent_id="my_agent")
        kb.add("User prefers Python", importance=0.9)
        context = kb.recall("what language?")
        # → "[procedural] User prefers Python"

        # Configure provider once at init — no need to repeat credentials:
        kb = KnowledgeBase(agent_id="my_agent", provider="openai", api_key="sk-...")
        kb.learn(turns)
        kb.learn(more_turns)

    Args:
        agent_id: Unique identifier for this agent's memory namespace.
        storage: Storage backend (defaults to YAMLStorage in .ai_knot/).
        provider: Default LLM provider name or instance used by :meth:`learn`.
            When set, ``learn()`` calls do not need to repeat the provider name.
        api_key: Default API key for the provider. Falls back to environment
            variables when not set.
        model: Default model override for the provider.
        **provider_kwargs: Extra provider arguments stored as defaults for
            ``learn()`` (e.g. ``folder_id`` for Yandex, ``base_url`` for
            openai-compat).
    """

    def __init__(
        self,
        agent_id: str,
        storage: StorageBackend | None = None,
        *,
        provider: str | LLMProvider | None = None,
        api_key: str | None = None,
        model: str | None = None,
        decay_config: dict[str, float] | None = None,
        llm_recall: bool | None = None,
        rrf_weights: tuple[float, ...] | None = None,
        expansion_weight: float | None = None,
        episodic_ttl_hours: float = 72.0,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        embed_api_key: str | None = None,
        **provider_kwargs: str,
    ) -> None:
        self._agent_id = agent_id
        self._storage: StorageBackend = storage or YAMLStorage()
        self._bm25 = TFIDFRetriever(
            rrf_weights=rrf_weights or (5.0, 3.0, 2.0, 1.5, 1.5, 1.0),
        )
        self._dense = DenseRetriever()
        self._hybrid = HybridRetriever(self._bm25, self._dense)
        self._retriever: TFIDFRetriever | HybridRetriever = self._bm25
        self._embed_url = embed_url
        self._embed_model = embed_model
        self._embed_api_key = embed_api_key
        self._embedded_ids: set[str] = set()
        self._default_provider = provider
        self._default_api_key = api_key
        self._default_model = model
        self._decay_config = decay_config
        self._llm_recall = llm_recall if llm_recall is not None else (provider is not None)
        self._expansion_weight = expansion_weight
        self._query_expander: LLMQueryExpander | None = None
        self._default_provider_kwargs: dict[str, str] = dict(provider_kwargs)
        self._episodic_ttl_hours = episodic_ttl_hours

    # Jaccard threshold for near-duplicate detection in add().
    # 0.7 suppresses sliding-window stride-1 overlap (~0.84) while preserving
    # legitimate re-mentions of a topic from different contexts (~0.3–0.5).
    _DEDUP_THRESHOLD: float = 0.7
    # Number of recent facts to compare against (catches consecutive window overlap).
    _DEDUP_WINDOW: int = 5

    def add(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.8,
        tags: list[str] | tuple[str, ...] = (),
    ) -> Fact:
        """Add a fact manually to the knowledge base.

        Near-duplicate detection: if the new content has Jaccard similarity
        ≥ 0.7 with any of the last 5 stored facts, the existing fact is
        returned without writing a duplicate.  This suppresses the 84%-overlap
        near-duplicates produced by 3-turn stride-1 sliding windows while
        preserving legitimate re-mentions of a topic (typically Jaccard < 0.5).

        Args:
            content: The knowledge string.
            type: Classification (semantic/procedural/episodic).
            importance: How critical (0.0-1.0).
            tags: Optional labels.

        Returns:
            The created Fact (or the existing near-duplicate if one was found).
        """
        if not content.strip():
            raise ValueError("content must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

        facts = self._storage.load(self._agent_id)

        # Fuzzy near-duplicate check against the last _DEDUP_WINDOW facts.
        if facts:
            new_tokens = frozenset(_tokenize(content.lower()))
            if new_tokens:
                for existing_fact in facts[-self._DEDUP_WINDOW :]:
                    existing_tokens = frozenset(_tokenize(existing_fact.content.lower()))
                    if existing_tokens:
                        union_size = len(new_tokens | existing_tokens)
                        jaccard = len(new_tokens & existing_tokens) / union_size
                        if jaccard >= self._DEDUP_THRESHOLD:
                            logger.debug(
                                "Skipped near-duplicate (jaccard=%.2f): '%s'",
                                jaccard,
                                content[:50],
                            )
                            return existing_fact

        fact = Fact(
            content=content,
            type=type,
            importance=importance,
            tags=list(tags),
        )
        # C6c: inject canonical date tags for temporal recall (mode-agnostic).
        enrich_date_tags(fact)

        # C6b v2: split enumeration/aggregation lists into atomic child facts
        # so raw/dated windows get the same treatment as learn/dated-learn.
        all_facts = split_enumerations([fact])  # [fact, *children]
        accepted_children: list[Fact] = []
        # Reference for child near-dup check: previously STORED facts only.
        # We intentionally exclude both the parent (fact) and sibling children
        # from the reference window:
        # - Parent: children are derived from it, so they always share most tokens.
        # - Siblings: they differ only in the final enumerated item but share the
        #   date prefix and verb, making their pairwise Jaccard artificially high
        #   (≈ 0.71 for 4-item dated windows) which would suppress valid children.
        reference_window = facts[-self._DEDUP_WINDOW :]
        for child in all_facts[1:]:  # skip original (index 0 = fact)
            child_tokens = frozenset(_tokenize(child.content.lower()))
            if not child_tokens:
                continue
            is_dup = False
            # Check only against previously stored facts (not parent or siblings).
            for ref in reference_window:
                ref_tokens = frozenset(_tokenize(ref.content.lower()))
                if not ref_tokens:
                    continue
                union_size = len(child_tokens | ref_tokens)
                if len(child_tokens & ref_tokens) / union_size >= self._DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                enrich_date_tags(child)  # inherit date from [date] prefix in content
                accepted_children.append(child)

        facts.append(fact)
        facts.extend(accepted_children)
        self._storage.save(self._agent_id, facts)
        logger.info(
            "Added fact '%s' (+%d derived) to agent '%s'",
            content[:50],
            len(accepted_children),
            self._agent_id,
        )
        return fact

    def add_many(
        self,
        facts: Sequence[str | dict[str, Any]],
        *,
        type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.8,
        tags: list[str] | tuple[str, ...] = (),
    ) -> list[Fact]:
        """Add multiple pre-extracted facts at once without an LLM call.

        Each item can be a plain string (content only) or a dict with any of the
        keys ``content`` (required), ``type``, ``importance``, ``tags``.  Dict
        values take precedence over the method-level defaults.

        Args:
            facts: Sequence of fact strings or dicts.
            type: Default memory type applied to string items and to dicts that
                do not specify a type.
            importance: Default importance applied to string items and to dicts
                that do not specify importance.
            tags: Default tags applied to string items and to dicts that do not
                specify tags.

        Returns:
            List of created Facts in the same order as the input.

        Raises:
            ValueError: If any fact has empty content or an invalid importance.
        """
        if not facts:
            return []

        # Build and validate all Fact objects before touching storage so that
        # a validation error on item N does not leave the first N-1 persisted.
        new_facts: list[Fact] = []
        for item in facts:
            if isinstance(item, str):
                content = item
                item_type = type
                item_importance = importance
                item_tags: list[str] = list(tags)
            else:
                raw_content = item.get("content")
                if not raw_content:
                    raise ValueError("each dict item must include a non-empty 'content' key")
                content = str(raw_content)
                raw_type = item.get("type")
                item_type = MemoryType(raw_type) if raw_type else type
                item_importance = float(item.get("importance", importance))
                raw_tags = item.get("tags", list(tags))
                item_tags = raw_tags if isinstance(raw_tags, (list, tuple)) else list(tags)  # type: ignore[assignment]

            if not content.strip():
                raise ValueError("content must not be empty")
            if not 0.0 <= item_importance <= 1.0:
                raise ValueError(f"importance must be between 0.0 and 1.0, got {item_importance}")
            new_facts.append(
                Fact(content=content, type=item_type, importance=item_importance, tags=item_tags)
            )

        # Single load + save: O(1) storage round-trips regardless of list length.
        existing = self._storage.load(self._agent_id)
        existing.extend(new_facts)
        self._storage.save(self._agent_id, existing)
        logger.info("Added %d facts to agent '%s'", len(new_facts), self._agent_id)
        return new_facts

    def add_episodic(
        self,
        content: str,
        *,
        importance: float = 0.3,
        tags: list[str] | tuple[str, ...] = (),
        ttl_hours: float | None = None,
    ) -> Fact:
        """Add a short-lived episodic fact (L1 hippocampus-like buffer).

        Episodic facts have a time-to-live: they expire after ``ttl_hours``
        (defaults to ``episodic_ttl_hours`` set at init, typically 72h).
        They are excluded from default recall() and recall_facts() results
        (which only return active semantic/procedural facts), but are visible
        to consolidate_episodic() for promotion to semantic memory.

        Use for: raw conversation snippets, session context, unverified claims
        that need consolidation before becoming durable knowledge.
        """
        if not content.strip():
            raise ValueError("content must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

        ttl = ttl_hours if ttl_hours is not None else self._episodic_ttl_hours
        now = datetime.now(UTC)
        fact = Fact(
            content=content,
            type=MemoryType.EPISODIC,
            importance=importance,
            tags=list(tags),
            valid_from=now,
            valid_until=now + timedelta(hours=ttl),
        )
        facts = self._storage.load(self._agent_id)
        facts.append(fact)
        self._storage.save(self._agent_id, facts)
        logger.info("Added episodic fact (TTL=%.0fh) to agent '%s'", ttl, self._agent_id)
        return fact

    async def arecall(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> str:
        """Async variant of :meth:`recall` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recall(
                query, top_k=top_k, now=now, include_unsupported=include_unsupported
            ),
        )

    async def arecall_facts(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> list[Fact]:
        """Async variant of :meth:`recall_facts` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recall_facts(
                query, top_k=top_k, now=now, include_unsupported=include_unsupported
            ),
        )

    def _expand_query_for_embed(self, query: str) -> str:
        """Expand query via LLM for semantic embedding.

        LLM expansion adds synonyms/related terms that help the embedding model
        bridge vocabulary gaps (e.g. "database" → "PostgreSQL SQL storage").

        BM25 does NOT use LLM expansion — it uses corpus-aware PRF instead,
        which avoids off-target term pollution from the LLM's training data.

        Returns the expanded text (or the original query if expansion is
        unavailable).
        """
        if not self._llm_recall:
            return query
        if not self._default_provider:
            logger.warning(
                "llm_recall=True but no provider configured — "
                "query expansion skipped, returning original query"
            )
            return query
        if self._query_expander is None:
            provider = self._default_provider
            if isinstance(provider, str):
                provider = create_provider(
                    provider, self._default_api_key, **self._default_provider_kwargs
                )
            self._query_expander = LLMQueryExpander(provider, self._default_model)
        return self._query_expander.expand(query)

    # Embedding batch size — avoids Ollama timeout on large fact sets.
    _EMBED_BATCH: int = 256

    def _embed_for_recall(
        self,
        facts: list[Fact],
        query: str,
    ) -> list[float] | None:
        """Embed new facts and the query for hybrid retrieval.

        Facts are embedded in batches of ``_EMBED_BATCH`` to avoid HTTP
        timeouts when the fact set is large (e.g. 5 000+ facts on first
        recall).  The query is always embedded in its own final batch.

        Returns the query vector, or ``None`` if embedding is unavailable
        (Ollama down, timeout, etc.) — callers fall back to BM25-only.
        """
        from ai_knot.embedder import embed_texts

        new_facts = [f for f in facts if f.id not in self._embedded_ids]
        fact_texts = [f.content for f in new_facts]

        # Build batches: fact texts in chunks, then query alone.
        batches: list[list[str]] = []
        for i in range(0, len(fact_texts), self._EMBED_BATCH):
            batches.append(fact_texts[i : i + self._EMBED_BATCH])
        batches.append([query])

        try:
            import contextlib

            loop: asyncio.AbstractEventLoop | None = None
            with contextlib.suppress(RuntimeError):
                loop = asyncio.get_running_loop()

            all_vectors: list[list[float]] = []

            if loop is not None and loop.is_running():
                # Already inside an event loop (e.g. Jupyter, MCP server) —
                # can't nest asyncio.run().  Use a thread to avoid deadlock.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    for batch in batches:
                        vecs = pool.submit(
                            asyncio.run,
                            embed_texts(
                                batch,
                                base_url=self._embed_url,
                                model=self._embed_model,
                                api_key=self._embed_api_key,
                            ),
                        ).result(timeout=120)
                        if not vecs:
                            logger.warning(
                                "Embedding batch failed (%d texts) — falling back to BM25-only",
                                len(batch),
                            )
                            return None
                        all_vectors.extend(vecs)
            else:
                for batch in batches:
                    vecs = asyncio.run(
                        embed_texts(
                            batch,
                            base_url=self._embed_url,
                            model=self._embed_model,
                            api_key=self._embed_api_key,
                        )
                    )
                    if not vecs:
                        logger.warning(
                            "Embedding batch failed (%d texts) — falling back to BM25-only",
                            len(batch),
                        )
                        return None
                    all_vectors.extend(vecs)
        except Exception:
            logger.warning("Embedding unavailable — falling back to BM25-only")
            return None

        if not all_vectors:
            return None

        # Last vector is the query; everything before is fact embeddings.
        query_vector = all_vectors[-1]
        fact_vectors = all_vectors[:-1]
        for f, vec in zip(new_facts, fact_vectors, strict=True):
            self._dense.add_embeddings({f.id: vec})
            self._embedded_ids.add(f.id)

        return query_vector

    def _build_entity_dictionary(self, facts: list[Fact]) -> set[str]:
        """Collect known entity names from structured entity/value_text fields."""
        entities: set[str] = set()
        for f in facts:
            if f.entity and len(f.entity) > 2:
                entities.add(f.entity.lower())
            if (
                f.value_text
                and len(f.value_text) > 2
                and not f.value_text.replace(".", "").replace(",", "").isdigit()
            ):
                entities.add(f.value_text.strip().lower())
        return entities

    @staticmethod
    def _build_entity_mention_index(
        facts: list[Fact],
        entity_dict: set[str],
    ) -> dict[str, set[str]]:
        """Map entity names → fact IDs that mention them in content."""
        index: dict[str, set[str]] = defaultdict(set)
        for f in facts:
            content_lower = f.content.lower()
            for entity in entity_dict:
                if entity in content_lower:
                    index[entity].add(f.id)
        return index

    def _execute_recall(
        self,
        query: str,
        *,
        top_k: int,
        now: datetime | None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
        trace: dict[str, Any] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Staged retrieval pipeline.

        STAGE 0 — Classify intent and load PipelineConfig (Phase E Router).
          Determines RRF weights, MMR-lambda, DDSA gating, sort strategy,
          memory_type_filter, and field_weights_override for this query.

        STAGE 1 — Collect candidates:
          Channel A: all facts with positive raw BM25F score.
          Channel B: exhaustive posting lookup for rare query tokens
                     (IDF > median_idf), language-agnostic.
          Channel C: entity-hop via value_text → entity links (token-intersection
                     match; learn-ON only; graceful no-op in learn-OFF / dated mode).
          Channel D: dense cosine matches (optional; no-op without embeddings).

        STAGE 2 — Score: raw BM25F per candidate with optional field_weights_override.

        STAGE 3 — Intent-weighted RRF fusion (Phase E replacement for greedy).
          Six signals: BM25, slot-exact, trigram Jaccard, importance,
          retention, recency — weighted by PipelineConfig.rrf_weights.

        STAGE 4 — MMR dedup (Jaccard-only, slot-aware, lambda from PipelineConfig).

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to exclude from results.
            include_unsupported: Include facts with ``supported=False``.
        """
        # --- STAGE 0: Intent classification ---
        intent = classify_recall_intent(query)
        config = get_pipeline_config(intent)

        # Stage 0 — Lexical Bridge (opt-in via AI_KNOT_LEXICAL_BRIDGE=1)
        _lexical_expansion: Any = None
        if os.environ.get("AI_KNOT_LEXICAL_BRIDGE") == "1":
            from ai_knot.query_lexicon import expand_query_lexically

            _lexical_expansion = expand_query_lexically(
                query,
                intent.value,
                max_terms_per_intent=config.lexical_expansion_max,
            )

        all_facts = self._storage.load(self._agent_id)
        if not all_facts:
            return []

        now_dt = now or datetime.now(UTC)
        # Temporal filter + exclude episodic (raw buffer) + verification gate.
        # E2.1: memory_type_filter restricts to a specific MemoryType (e.g. PROCEDURAL).
        facts = [
            f
            for f in all_facts
            if f.is_active(now_dt)
            and f.type != MemoryType.EPISODIC
            and (include_unsupported or f.supported is not False)
            and (config.memory_type_filter is None or f.type == config.memory_type_filter)
        ]
        if not facts:
            return []

        facts = apply_decay(facts, type_exponents=self._decay_config, now=now_dt)
        candidate_facts = [f for f in facts if f.id not in excluded_ids] if excluded_ids else facts
        if not candidate_facts:
            return []

        # Build InvertedIndex once — used for BM25 scoring, IDF lookup, Channel B.
        index = InvertedIndex(candidate_facts)

        # Build entity index for Channel C (learn-ON only; no-op in dated mode).
        entity_index: dict[str, list[Fact]] = defaultdict(list)
        fact_map: dict[str, Fact] = {}
        for f in candidate_facts:
            fact_map[f.id] = f
            if f.entity:
                entity_index[f.entity].append(f)

        # --- STAGE 1: Candidate Collection ---

        # Channel A: all facts with positive raw BM25F score.
        # E2.2: field_weights_override lets NAVIGATIONAL intent boost tags/canonical.
        # Stage 0: lexical expansion terms (if bridge active) passed as expansion_weights.
        _lex_expansion_weights: dict[str, float] = (
            _lexical_expansion.expansion_weights if _lexical_expansion is not None else {}
        )
        bm25_raw = index.score(
            query,
            field_weights_override=config.field_weights_override,
            expansion_weights=_lex_expansion_weights if _lex_expansion_weights else None,
        )
        candidate_ids: set[str] = {fid for fid, s in bm25_raw.items() if s > 0}
        _trace_ch_a = set(candidate_ids) if trace is not None else None

        # Channel B: exhaustive posting lookup for rare query tokens.
        # Captures entity-scoped facts even when the entity name is high-DF
        # (IDF≈0 after high-DF filter) by targeting only truly rare tokens.
        med_idf = index.median_idf()
        for token in _tokenize(query):
            if index.idf(token) > med_idf:
                candidate_ids |= set(index._content_postings.get(token, {}).keys())
        _trace_ch_b = (
            (candidate_ids - _trace_ch_a) if trace is not None and _trace_ch_a is not None else None
        )

        # Channel C: entity-hop via value_text → entity links (learn-ON facts only).
        # Uses token intersection (frozenset) instead of exact string lookup so that
        # "pottery" matches "pottery class", while avoiding false positives like
        # "car" matching "oscar" — only tokens with len > 2 are considered significant.
        for fid in list(candidate_ids):
            f_hop = fact_map.get(fid)
            if f_hop and f_hop.value_text:
                hop_key = f_hop.value_text.strip().lower()
                if (
                    hop_key
                    and len(hop_key) > 2
                    and not hop_key.replace(".", "").replace(",", "").isdigit()
                ):
                    hop_tokens = frozenset(_tokenize(hop_key))
                    for known_entity, related_facts in entity_index.items():
                        entity_tokens = frozenset(_tokenize(known_entity))
                        shared = hop_tokens & entity_tokens
                        if any(len(t) > 2 for t in shared):
                            for related in related_facts:
                                candidate_ids.add(related.id)
        _trace_ch_c = (
            candidate_ids - _trace_ch_a - _trace_ch_b
            if trace is not None and _trace_ch_a is not None and _trace_ch_b is not None
            else None
        )

        # Channel D: dense retrieval (no-op when embeddings unavailable).
        # Track dense scores so the top-1 semantic match can be guaranteed
        # even when it has zero BM25 (vocabulary gap / terminology mismatch).
        dense_scores: dict[str, float] = {}
        embed_text = self._expand_query_for_embed(query)
        query_vector = self._embed_for_recall(candidate_facts, embed_text)
        if query_vector is not None and self._dense.has_embeddings():
            for f, sim in self._dense.search(query_vector, candidate_facts, top_k=top_k):
                dense_scores[f.id] = sim
                candidate_ids.add(f.id)

        if trace is not None:
            trace["stage0_lexical_bridge"] = (
                {
                    "frames_applied": _lexical_expansion.frames_applied,
                    "terms_added": _lexical_expansion.terms_added,
                    "expansion_weights": _lexical_expansion.expansion_weights,
                }
                if os.environ.get("AI_KNOT_LEXICAL_BRIDGE") == "1"
                else None
            )
            _prev = (_trace_ch_a or set()) | (_trace_ch_b or set()) | (_trace_ch_c or set())
            trace["stage1_candidates"] = {
                "from_bm25": sorted(_trace_ch_a or set()),
                "from_rare_tokens": sorted(_trace_ch_b or set()),
                "from_entity_hop": sorted(_trace_ch_c or set()),
                "from_dense": sorted(candidate_ids - _prev),
                "total": len(candidate_ids),
                "dense_scores": dict(dense_scores),
            }

        if not candidate_ids:
            return []

        # --- STAGE 3: Intent-weighted RRF fusion (Phase E — replaces greedy) ---
        # Six ranked signals fused via Reciprocal Rank Fusion with per-intent weights.
        # Weights: (BM25, slot, trigram, importance, retention, recency).
        candidate_id_list = list(candidate_ids)
        query_tokens_frozen = frozenset(_tokenize(query))
        query_trigrams = _char_trigrams(query)

        # Ranker 1: BM25 score (field_weights_override already applied above).
        bm25_ranked = sorted(
            candidate_id_list, key=lambda fid: bm25_raw.get(fid, 0.0), reverse=True
        )

        # Ranker 2: Slot-exact — fraction of slot-address tokens in query.
        slot_scores_rrf: dict[str, float] = {
            fid: _slot_exact_score(query_tokens_frozen, fact_map[fid])
            for fid in candidate_id_list
            if fid in fact_map
        }
        slot_ranked = sorted(
            candidate_id_list, key=lambda fid: slot_scores_rrf.get(fid, 0.0), reverse=True
        )

        # Ranker 3: Char-trigram Jaccard (max across content / canonical / evidence).
        def _trig_score(fid: str) -> float:
            ct = index.content_trigrams.get(fid, frozenset())
            kt = index.canonical_trigrams.get(fid, frozenset())
            et = index.evidence_trigrams.get(fid, frozenset())
            if not query_trigrams:
                return 0.0
            s_c = len(query_trigrams & ct) / len(query_trigrams | ct) if ct else 0.0
            s_k = len(query_trigrams & kt) / len(query_trigrams | kt) if kt else 0.0
            s_e = len(query_trigrams & et) / len(query_trigrams | et) if et else 0.0
            return max(s_c, s_k, s_e)

        trigram_ranked = sorted(candidate_id_list, key=_trig_score, reverse=True)

        # Ranker 4–6: importance / retention / recency.
        importance_ranked = sorted(
            candidate_id_list,
            key=lambda fid: fact_map[fid].importance if fid in fact_map else 0.0,
            reverse=True,
        )
        retention_ranked = sorted(
            candidate_id_list,
            key=lambda fid: fact_map[fid].retention_score if fid in fact_map else 0.0,
            reverse=True,
        )
        recency_ranked = sorted(
            candidate_id_list,
            key=lambda fid: (
                fact_map[fid].created_at if fid in fact_map else datetime.min.replace(tzinfo=UTC)
            ),
            reverse=True,
        )

        fused_scores = _rrf_fuse(
            [
                bm25_ranked,
                slot_ranked,
                trigram_ranked,
                importance_ranked,
                retention_ranked,
                recency_ranked,
            ],
            weights=list(config.rrf_weights),
        )

        selected_ids: list[str] = sorted(
            (fid for fid in candidate_id_list if fid in fact_map),
            key=lambda fid: fused_scores.get(fid, 0.0),
            reverse=True,
        )[:top_k]

        if trace is not None:
            trace["stage3_rrf"] = {
                "intent": intent.value,
                "rrf_weights": list(config.rrf_weights),
                "selected_ids": list(selected_ids),
            }

        # Dense guarantee: the top-1 semantic result always makes the output.
        # Vocabulary-gap facts (BM25=0) are captured via dense search but may
        # score low in BM25-dominant RRF — guarantee the best dense match is included.
        _dense_guarantee_applied = False
        _dense_guarantee_fid: str | None = None
        _dense_guarantee_displaced: str | None = None
        if dense_scores:
            best_dense_fid = max(dense_scores, key=dense_scores.__getitem__)
            if best_dense_fid in fact_map and best_dense_fid not in selected_ids:
                _dense_guarantee_fid = best_dense_fid
                _dense_guarantee_applied = True
                if len(selected_ids) >= top_k:
                    _dense_guarantee_displaced = selected_ids[top_k - 1]
                    selected_ids = selected_ids[: top_k - 1] + [best_dense_fid]
                else:
                    selected_ids.append(best_dense_fid)
        if trace is not None:
            trace["stage3b_dense_guarantee"] = {
                "applied": _dense_guarantee_applied,
                "fid": _dense_guarantee_fid,
                "displaced": _dense_guarantee_displaced,
            }

        # Build (Fact, fused_score) pairs; sort best-first for MMR input.
        pairs: list[tuple[Fact, float]] = [
            (fact_map[fid], fused_scores.get(fid, 0.0)) for fid in selected_ids if fid in fact_map
        ]
        if not pairs:
            return []

        pairs.sort(key=lambda x: x[1], reverse=True)

        # --- STAGE 4a: DDSA — expand candidates via tags/entity/slot/time ---
        # Take the top third of candidates as activation seeds so that activated
        # facts have room in the final top_k.  With seeds == top_k and decay < 1,
        # all activated facts would be truncated (no-op), so we cap the seed set.
        # E1.5: gated by config.use_ddsa in addition to the global DDSA_ENABLED env flag.
        _trace_pre_ddsa_ids = [f.id for f, _ in pairs] if trace is not None else None
        if _ddsa.DDSA_ENABLED and config.use_ddsa and pairs:
            ddsa_seed_n = max(top_k // 3, min(10, len(pairs)))
            ddsa_seeds = pairs[:ddsa_seed_n]

            # Inject the dense-guaranteed fact into ddsa_seeds when it isn't already
            # there.  After pairs.sort(by BM25), a vocabulary-gap dense_fact (BM25=0)
            # sits at the tail and would be lost once tail-merge is removed.  Injecting
            # it as a seed preserves it at its original score (no decay applied to seeds).
            if dense_scores:
                best_dense_fid = max(dense_scores, key=dense_scores.__getitem__)
                if best_dense_fid in fact_map and not any(
                    f.id == best_dense_fid for f, _ in ddsa_seeds
                ):
                    dense_pair = next(((f, s) for f, s in pairs if f.id == best_dense_fid), None)
                    if dense_pair is not None:
                        # Replace the weakest seed (last position) to keep ddsa_seed_n stable.
                        ddsa_seeds = (
                            (ddsa_seeds[:-1] + [dense_pair]) if ddsa_seeds else [dense_pair]
                        )

            ddsa_extra_k = max(0, top_k - len(ddsa_seeds))
            pairs = spreading_activation(
                ddsa_seeds,
                index,
                topk=len(ddsa_seeds) + ddsa_extra_k,
                decay=0.6,
                temporal_window_sec=60,
                activation_budget=ddsa_extra_k,
            )

        if trace is not None:
            trace["stage4a_ddsa"] = {
                "enabled": _ddsa.DDSA_ENABLED,
                "pre_ddsa_ids": _trace_pre_ddsa_ids or [],
                "post_ddsa_ids": [f.id for f, _ in pairs],
            }

        # --- STAGE 4b: MMR diversity dedup (Jaccard-only, intent-aware lambda) ---
        _trace_pre_mmr_ids = [f.id for f, _ in pairs] if trace is not None else None
        pairs = self._mmr_select(pairs, top_k=top_k, lambda_=config.mmr_lambda)
        if trace is not None:
            _post_mmr = {f.id for f, _ in pairs}
            trace["stage4b_mmr"] = {
                "pre_mmr_ids": _trace_pre_mmr_ids or [],
                "post_mmr_ids": [f.id for f, _ in pairs],
                "dropped_ids": [fid for fid in (_trace_pre_mmr_ids or []) if fid not in _post_mmr],
            }

        # Update access metadata on the *original* (unfiltered) fact objects so
        # that the subsequent save persists ALL facts, not just the filtered set.
        returned_ids = {f.id for f, _ in pairs}
        access_time = datetime.now(UTC)
        for fact in facts:
            if fact.id in returned_ids:
                interval = (access_time - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = access_time
        self._storage.save(self._agent_id, all_facts)
        return pairs

    _MMR_DATE_PREFIX = re.compile(r"^\[\d.*?\]\s*")

    @staticmethod
    def _mmr_select(
        pairs: list[tuple[Fact, float]],
        *,
        top_k: int,
        lambda_: float = 0.5,
    ) -> list[tuple[Fact, float]]:
        """Select top_k facts via MMR (Carbonell & Goldstein 1998).

        Greedily picks facts that are both high-scoring and distinct from
        already-selected facts.  Similarity uses Jaccard only (intersection /
        union), so that short extracted facts are NOT penalized against the
        longer raw turns they originated from.

        Slot-aware protection (Phase E): facts with the same ``slot_key`` but
        different ``value_text`` are treated as distinct list items and are NOT
        penalized for lexical overlap — each item in a list is independently
        selected regardless of surface similarity to other items in that slot.

        Args:
            pairs: Candidates sorted by score (best first).
            top_k: Number of facts to return.
            lambda_: Relevance weight (0 = full diversity, 1 = no dedup).
                Default 0.5 balances relevance and diversity.

        Returns:
            Up to top_k (Fact, original_score) pairs in selection order.
        """
        if len(pairs) <= top_k:
            return pairs

        # Strip date prefixes before comparing content so that
        # "[8 May] Melanie went swimming" ≡ "Melanie went swimming".
        def _content_key(f: Fact) -> str:
            return KnowledgeBase._MMR_DATE_PREFIX.sub("", f.content).strip().lower()

        # Precompute token sets once — O(n × avg_tokens).
        token_sets: dict[str, frozenset[str]] = {
            f.id: frozenset(_tokenize(_content_key(f))) for f, _ in pairs
        }

        # Normalize RRF scores to [0, 1] for MMR arithmetic.
        raw_scores = [s for _, s in pairs]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        norm: dict[str, float] = {f.id: (s - min_s) / score_range for f, s in pairs}

        # Greedy selection — O(top_k × remaining × set_ops).
        # sel_facts tracks the selected Fact objects for slot-aware comparison.
        selected: list[tuple[Fact, float]] = [pairs[0]]
        sel_tokens: list[frozenset[str]] = [token_sets[pairs[0][0].id]]
        sel_facts: list[Fact] = [pairs[0][0]]
        remaining = list(pairs[1:])

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float("inf")

            for i, (f, _) in enumerate(remaining):
                ft = token_sets[f.id]
                # Max Jaccard similarity to any already-selected fact.
                # Slot-aware: different value_text under the same slot_key are
                # treated as list items — they do not compete with each other.
                max_sim = 0.0
                for sel_f, st in zip(sel_facts, sel_tokens, strict=False):
                    if (
                        f.slot_key
                        and f.slot_key == sel_f.slot_key
                        and f.value_text != sel_f.value_text
                    ):
                        # Different list items: no diversity penalty between them.
                        sim = 0.0
                    elif not ft or not st:
                        continue
                    else:
                        inter = len(ft & st)
                        union = len(ft | st)
                        sim = inter / union if union else 0.0
                    if sim > max_sim:
                        max_sim = sim

                mmr = lambda_ * norm[f.id] - (1.0 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            fact, score = remaining.pop(best_idx)
            selected.append((fact, score))
            sel_tokens.append(token_sets[fact.id])
            sel_facts.append(fact)

        return selected

    @staticmethod
    def _select_topk(
        scores: dict[str, dict[str, float]],
        token_sets: dict[str, frozenset[str]],
        idf_map: dict[str, float],
        specificity: float,
        top_k: int,
    ) -> list[str]:
        """Greedy selection: BM25 relevance + IDF-weighted token coverage.

        ``specificity`` (corpus-derived via IDF, no vocabulary lists) drives
        the coverage/relevance tradeoff:
          low (0.2)  → 80% coverage, 20% relevance  — broad/aggregation
          high (0.8) → 20% coverage, 80% relevance  — narrow/point

        Rarity is implicit: IDF-weighted coverage values rare tokens more
        than common ones — no separate rarity parameter.

        Args:
            scores: {fid: {'bm25': raw_score}} for candidate facts.
            token_sets: {fid: frozenset(stemmed_tokens)}.
            idf_map: {token: idf_value} precomputed from the corpus.
            specificity: Query specificity in [0.0, 1.0].
            top_k: Number of facts to return.

        Returns:
            List of selected fact IDs in greedy selection order.
        """
        if not scores:
            return []

        coverage_weight = max(0.1, 1.0 - specificity)
        relevance_weight = max(0.1, specificity)

        max_bm25 = max((s["bm25"] for s in scores.values()), default=1.0) or 1.0
        pool_scores: dict[str, float] = {
            fid: relevance_weight * (s["bm25"] / max_bm25) for fid, s in scores.items()
        }

        selected: list[str] = []
        covered_tokens: set[str] = set()
        remaining = sorted(pool_scores.keys(), key=lambda fid: pool_scores[fid], reverse=True)

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -float("inf")

            for i, fid in enumerate(remaining):
                fact_tokens = token_sets.get(fid, frozenset())
                new_tokens = fact_tokens - covered_tokens

                # IDF-weighted coverage: rare tokens contribute more value.
                total_idf = sum(idf_map.get(t, 0.0) for t in fact_tokens) or 1e-9
                new_idf = sum(idf_map.get(t, 0.0) for t in new_tokens)
                new_value = new_idf / total_idf

                composite = pool_scores[fid] + coverage_weight * new_value
                if composite > best_score:
                    best_score = composite
                    best_idx = i

            fid = remaining.pop(best_idx)
            selected.append(fid)
            covered_tokens |= token_sets.get(fid, frozenset())

        return selected

    @staticmethod
    def _sandwich_reorder(
        pairs: list[tuple[Fact, float]],
    ) -> list[tuple[Fact, float]]:
        """Reorder for LLM positional attention (Liu et al. NeurIPS 2024).

        LLMs attend best to beginning and end of context.  Place top-scoring
        facts at positions 1 and N, lower-scoring facts in the middle.
        """
        if len(pairs) <= 10:
            return pairs
        top1 = [pairs[0]]
        tail = pairs[1:5]
        middle = pairs[5:]
        return top1 + middle + tail

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> str:
        """Retrieve relevant facts as a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        pairs = self._execute_recall(
            query, top_k=top_k, now=now, include_unsupported=include_unsupported
        )
        if not pairs:
            return ""
        config = get_pipeline_config(classify_recall_intent(query))
        if config.sort_strategy == "sandwich":
            pairs = self._sandwich_reorder(pairs)
        elif config.sort_strategy == "chronological":
            head = list(pairs[:15])
            head.sort(key=lambda x: x[0].created_at)
            pairs = head + list(pairs[15:])
        seen: set[str] = set()
        lines: list[str] = []
        for f, _ in pairs:
            text = f.prompt_surface or f.source_verbatim or f.content
            if f.entity and f.attribute:
                text = f"[{f.entity}: {f.attribute}={f.value_text}] {text}"
            if text not in seen:
                seen.add(text)
                lines.append(f"[{len(lines) + 1}] {text}")
        return "\n".join(lines)

    def list_facts(self) -> list[Fact]:
        """Return all stored facts for this agent.

        Returns:
            List of all Facts, in storage order.
        """
        return self._storage.load(self._agent_id)

    def recall_facts(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[Fact]:
        """Structured alternative to recall() — returns Fact objects.

        Use when you need IDs, types, importance scores, or other metadata.
        Use recall() when you only need a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        return [
            f
            for f, _ in self._execute_recall(
                query,
                top_k=top_k,
                now=now,
                excluded_ids=excluded_ids,
                include_unsupported=include_unsupported,
            )
        ]

    def recall_facts_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[tuple[Fact, float]]:
        """Like recall_facts() but also returns the relevance score for each fact.

        The score is a hybrid value (TF-IDF + retention + importance). Use it
        to rank or filter results in integration adapters.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Empty list if no facts stored or none match.
        """
        return self._execute_recall(
            query,
            top_k=top_k,
            now=now,
            excluded_ids=excluded_ids,
            include_unsupported=include_unsupported,
        )

    def recall_facts_with_trace(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> tuple[list[tuple[Fact, float]], dict[str, Any]]:
        """Diagnostic variant — returns (results, per-stage trace dict).

        The trace dict has keys: stage1_candidates, stage3_rrf,
        stage3b_dense_guarantee, stage4a_ddsa, stage4b_mmr.
        Intended for debugging scripts and tests only; not exposed via MCP.
        """
        trace: dict[str, Any] = {}
        pairs = self._execute_recall(
            query,
            top_k=top_k,
            now=now,
            include_unsupported=include_unsupported,
            trace=trace,
        )
        return pairs, trace

    def recall_by_tag(self, tag: str, *, include_unsupported: bool = False) -> list[Fact]:
        """Return all facts that carry the given tag.

        Tags are assigned at add() time via the ``tags=`` parameter.

        Args:
            tag: The tag string to filter by.
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of Facts whose tags include ``tag`` (may be empty).
        """
        return [
            f
            for f in self._storage.load(self._agent_id)
            if tag in f.tags and (include_unsupported or f.supported is not False)
        ]

    def replace_facts(self, facts: list[Fact]) -> None:
        """Replace all stored facts with the given list (used for import).

        Args:
            facts: New facts to store; replaces any existing facts.
        """
        self._storage.save(self._agent_id, facts)
        logger.info("Replaced facts for agent '%s' (%d total)", self._agent_id, len(facts))

    def forget(self, fact_id: str) -> None:
        """Remove a specific fact by its ID.

        Args:
            fact_id: The 8-char hex ID of the fact to remove.
        """
        self._storage.delete(self._agent_id, fact_id)
        logger.info("Forgot fact '%s' from agent '%s'", fact_id, self._agent_id)

    def decay(self, *, now: datetime | None = None) -> None:
        """Apply Ebbinghaus forgetting curve to all stored facts.

        Args:
            now: Point-in-time for decay calculation (default: current UTC).
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return
        apply_decay(facts, type_exponents=self._decay_config, now=now)
        self._storage.save(self._agent_id, facts)
        logger.debug("Applied decay to %d facts for agent '%s'", len(facts), self._agent_id)

    def clear_all(self) -> None:
        """Remove all facts for this agent."""
        self._storage.save(self._agent_id, [])
        logger.info("Cleared all facts for agent '%s'", self._agent_id)

    def snapshot(self, name: str) -> None:
        """Save the current state of the knowledge base as a named snapshot.

        Args:
            name: Identifier for this snapshot (e.g. "before_v2_campaign").

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts = self._storage.load(self._agent_id)
        self._storage.save_snapshot(self._agent_id, name, facts)
        logger.info("Snapshot '%s' saved for agent '%s'", name, self._agent_id)

    def list_snapshots(self) -> list[str]:
        """Return names of all saved snapshots for this agent.

        Returns:
            List of snapshot names sorted by creation time (oldest first).

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        return self._storage.list_snapshots(self._agent_id)

    def restore(self, name: str) -> None:
        """Replace current facts with the contents of a named snapshot.

        Args:
            name: The snapshot to restore.

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
            KeyError: If the snapshot does not exist.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts = self._storage.load_snapshot(self._agent_id, name)
        self._storage.save(self._agent_id, facts)
        logger.info("Restored snapshot '%s' for agent '%s'", name, self._agent_id)

    def diff(self, a: str, b: str) -> SnapshotDiff:
        """Compute the difference between two snapshots.

        Use the special name ``"current"`` to refer to the live facts in storage.

        Args:
            a: Name of the first snapshot (or "current").
            b: Name of the second snapshot (or "current").

        Returns:
            A :class:`SnapshotDiff` with ``added`` and ``removed`` fact lists.

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
            KeyError: If a named snapshot does not exist.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts_a = (
            self._storage.load(self._agent_id)
            if a == "current"
            else self._storage.load_snapshot(self._agent_id, a)
        )
        facts_b = (
            self._storage.load(self._agent_id)
            if b == "current"
            else self._storage.load_snapshot(self._agent_id, b)
        )
        ids_a = {f.id for f in facts_a}
        ids_b = {f.id for f in facts_b}
        removed = [f for f in facts_a if f.id not in ids_b]
        added = [f for f in facts_b if f.id not in ids_a]
        return SnapshotDiff(snapshot_a=a, snapshot_b=b, added=added, removed=removed)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the knowledge base.

        Returns:
            Dict with total_facts, by_type counts, avg_importance, avg_retention.
        """
        _zero: dict[str, Any] = {
            "total_facts": 0,
            "by_type": {"semantic": 0, "procedural": 0, "episodic": 0},
            "avg_importance": 0.0,
            "avg_retention": 0.0,
        }
        facts = self._storage.load(self._agent_id)
        now_dt = datetime.now(UTC)
        active_facts = [f for f in facts if f.is_active(now_dt)]
        if not active_facts:
            return _zero

        type_counts = Counter(f.type.value for f in active_facts)
        return {
            "total_facts": len(active_facts),
            "by_type": {
                "semantic": type_counts.get("semantic", 0),
                "procedural": type_counts.get("procedural", 0),
                "episodic": type_counts.get("episodic", 0),
            },
            "avg_importance": sum(f.importance for f in active_facts) / len(active_facts),
            "avg_retention": sum(f.retention_score for f in active_facts) / len(active_facts),
        }


# Backward-compat re-export: callers that do
#   from ai_knot.knowledge import SharedMemoryPool
# continue to work without changes.
from ai_knot.pool import SharedMemoryPool as SharedMemoryPool  # noqa: E402, F401
