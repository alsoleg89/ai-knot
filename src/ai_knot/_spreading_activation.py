"""Data-driven spreading activation — expands retrieval candidates via tags/slot/time."""

from __future__ import annotations

import os
from collections import Counter
from datetime import timedelta

from ai_knot._inverted_index import InvertedIndex
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Module-level flag for ablation benchmarks.  DDSA is opt-in: two benchmark
# iterations showed net-negative effect vs ddsa-off baseline.  Retained for
# experiments — enable explicitly via AIKNOT_DDSA_ENABLED=true (or 1/yes).
DDSA_ENABLED: bool = os.environ.get("AIKNOT_DDSA_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)


def spreading_activation(
    seeds: list[tuple[Fact, float]],
    index: InvertedIndex,
    *,
    topk: int,
    decay: float = 0.6,
    temporal_window_sec: int = 300,
    activation_budget: int | None = None,
) -> list[tuple[Fact, float]]:
    """Expand a seed set with associated facts via tags, entity, slot, and time.

    Seeds are top-ranked facts from BM25 selection.  DDSA finds additional
    facts connected to seeds through shared structure already present in
    *index* — no extra index is built.

    Three activation axes:

    * **Tag-hop (Axis 1)**: facts sharing any tag with a seed.
    * **Entity-hop (Axis 1b)**: facts mentioning a seed's entity / value_text
      in their content (learn-ON mode only; no-op when entity is empty).
    * **Slot-cluster (Axis 2)**: facts with the same ``slot_key`` as seeds,
      but only when that slot appears in ≥2 seeds (prevents single-seed noise).
    * **Temporal-window (Axis 3)**: facts created within *temporal_window_sec*
      of any seed's ``created_at``.

    Activated facts receive a decayed score (< seed scores), so seeds always
    rank first in the combined output.

    Args:
        seeds: ``(Fact, score)`` pairs — activation anchors.
        index: Pre-built :class:`InvertedIndex` for the candidate pool.
        topk: Cap on returned results.  Seeds are preserved; activated facts
            fill the remaining ``topk - len(seeds)`` slots.
        decay: Score decay for tag/entity hops.  Temporal hop uses
            ``decay * 0.4``.  Default ``0.6``.
        temporal_window_sec: Seconds; facts within this window of any seed
            are considered temporally adjacent.  Set to ``0`` to disable.
        activation_budget: Maximum number of *extra* (non-seed) facts to add
            before the final ``topk`` cap.  Default is unconstrained within
            ``topk``.

    Returns:
        Combined list of seed + activated facts, sorted by score descending,
        capped at ``topk``.
    """
    if not seeds:
        return seeds

    seed_ids: set[str] = {f.id for f, _ in seeds}
    activated: dict[str, float] = {}  # fact_id → best activation score

    # ------------------------------------------------------------------
    # Axis 1: Tag-hop via tags_postings
    # Collect the maximum seed score per tag token, then propagate to all
    # facts that share that tag.
    # ------------------------------------------------------------------
    seed_tag_scores: dict[str, float] = {}
    for fact, score in seeds:
        for tag in fact.tags:
            tok = tag.lower().strip()
            if tok:
                seed_tag_scores[tok] = max(seed_tag_scores.get(tok, 0.0), score)
    for tag_tok, base in seed_tag_scores.items():
        for fid in index.tags_postings.get(tag_tok, {}):
            if fid not in seed_ids:
                activated[fid] = max(activated.get(fid, 0.0), base * decay)

    # ------------------------------------------------------------------
    # Axis 1b: Entity-hop via content_postings
    # Only fires for learn-ON facts (entity / value_text are populated by LLM
    # extraction).  Propagates to any fact whose content mentions the entity.
    # ------------------------------------------------------------------
    for fact, score in seeds:
        ent_toks: list[str] = []
        if fact.entity:
            ent_toks.extend(_tokenize(fact.entity))
        if fact.value_text and len(fact.value_text) > 2:
            ent_toks.extend(_tokenize(fact.value_text))
        for tok in ent_toks:
            for fid in index.content_postings.get(tok, {}):
                if fid not in seed_ids:
                    activated[fid] = max(activated.get(fid, 0.0), score * decay * 0.8)

    # ------------------------------------------------------------------
    # Axis 2: Slot-cluster via slot_tokens
    # Only activates when a slot_key appears in >=2 seeds (hot slot), to
    # avoid polluting results when a single extracted fact has a slot_key.
    # ------------------------------------------------------------------
    slot_counts = Counter(f.slot_key for f, _ in seeds if f.slot_key)
    hot_slots = {s for s, c in slot_counts.items() if c >= 2}
    if hot_slots:
        seed_slot_score: dict[str, float] = {
            f.slot_key: sc for f, sc in seeds if f.slot_key in hot_slots
        }
        for fid, fact in index.facts.items():
            if fid in seed_ids:
                continue
            if fact.slot_key and fact.slot_key in hot_slots:
                base = seed_slot_score.get(fact.slot_key, 0.3)
                activated[fid] = max(activated.get(fid, 0.0), base * decay)

    # ------------------------------------------------------------------
    # Axis 3: Temporal-window via created_at
    # Facts created within temporal_window_sec of any seed are weakly activated.
    # ------------------------------------------------------------------
    if temporal_window_sec > 0:
        seed_times = [(f.created_at, sc) for f, sc in seeds if f.created_at is not None]
        if seed_times:
            window = timedelta(seconds=temporal_window_sec)
            for fid, fact in index.facts.items():
                if fid in seed_ids or fid in activated or fact.created_at is None:
                    continue
                for t, score in seed_times:
                    if abs(fact.created_at - t) <= window:
                        activated[fid] = max(activated.get(fid, 0.0), score * decay * 0.4)
                        break

    if not activated:
        return seeds[:topk]

    # Sort activated facts by score, apply optional budget cap.
    budget = activation_budget if activation_budget is not None else len(activated)
    extra_sorted = sorted(activated.items(), key=lambda kv: kv[1], reverse=True)[:budget]
    extra = [(index.facts[fid], sc) for fid, sc in extra_sorted if fid in index.facts]

    # Seeds rank above activated (decay ensures this), but sort to be safe.
    combined = list(seeds) + extra
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:topk]
