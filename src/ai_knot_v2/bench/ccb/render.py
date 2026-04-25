"""ESWP render operator — sorts atoms by regret_charge × credence for extraction.

Two render modes:
- render_pack_eswp(atoms): sparse triple format (S1 default)
- render_pack_eswp_with_raw(atoms, episodes_lookup): atoms + raw episode text

bench/ only — must not be imported from core/ ops/ store/ api/.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode

_FACTUAL_PREDS: frozenset[str] = frozenset(
    {"is", "has", "works_at", "lives_in", "prefers", "dislikes", "moved_to"}
)


def render_pack_eswp(atoms: list[MemoryAtom], query: str) -> str:  # noqa: ARG001
    """Render atoms sorted by regret_charge × credence descending.

    Factual predicates rendered as assertions; event predicates with temporal tag.
    """
    ordered = sorted(atoms, key=lambda a: a.regret_charge * a.credence, reverse=True)
    lines: list[str] = []
    for atom in ordered:
        obj = atom.object_value or ""
        pred_display = atom.predicate.replace("_", " ")
        if atom.predicate in _FACTUAL_PREDS:
            line = f"[fact] {atom.subject} {pred_display} {obj}"
        else:
            temp = ""
            if atom.valid_from is not None and atom.valid_until is not None:
                temp = f" (valid {atom.valid_from}–{atom.valid_until})"
            line = f"[event] {atom.subject} {pred_display} {obj}{temp}"
        if atom.polarity == "neg":
            line = line.replace("[fact]", "[neg-fact]", 1).replace("[event]", "[neg-event]", 1)
        lines.append(line)
    return "\n".join(lines)


def render_pack_eswp_with_raw(
    atoms: list[MemoryAtom],
    episodes_lookup: dict[str, RawEpisode],
    query: str,  # noqa: ARG001
    top_k: int = 12,
    char_budget: int = 22_000,
    per_turn_max: int = 1200,
) -> str:
    """Render raw episode turns + atom triples, mirroring v1 evidence_text format.

    Episodes are deduplicated and ordered by descending atom utility
    (regret_charge × credence). Each line: ``[N] [DATE] Speaker: text``.
    Atom triples appended at the end as a small bullet list.
    """
    ordered = sorted(atoms, key=lambda a: a.regret_charge * a.credence, reverse=True)

    seen: set[str] = set()
    ep_ids_ordered: list[str] = []
    for atom in ordered:
        for ep_id in atom.evidence_episodes:
            if ep_id not in seen and ep_id in episodes_lookup:
                ep_ids_ordered.append(ep_id)
                seen.add(ep_id)
        if len(ep_ids_ordered) >= top_k:
            break

    parts: list[str] = []
    total_chars = 0
    for i, ep_id in enumerate(ep_ids_ordered[:top_k], start=1):
        ep = episodes_lookup[ep_id]
        try:
            d = datetime.fromtimestamp(ep.timestamp, tz=UTC).date().isoformat()
            date_part = f"[{d}] "
        except (OSError, OverflowError, ValueError):
            date_part = ""
        speaker_label = ep.metadata.get("speaker_name") or ep.user_id or ep.speaker
        raw = ep.text[:per_turn_max]
        line = f"[{i}] {date_part}{speaker_label}: {raw}"
        if total_chars + len(line) > char_budget:
            break
        parts.append(line)
        total_chars += len(line) + 1

    return "\n".join(parts)
