"""CWP lineage render — derivation-tree format for LLM extraction.

Each atom is rendered as a small witness chain:
    [claim]  Caroline allergic_to penicillin
        ← supporting observations:
            [2023-01-15] Caroline: "I'm allergic to penicillin"
            [2024-03-20] Caroline: "Doctor confirmed my penicillin allergy"

This replaces both the sparse triple format (`render_pack_eswp`) and the flat
raw-episode dump. Hypothesis H_CWP_2: presenting derivation context closes
the A9 Reader-Extraction Blindness gap (atoms retrieved but LLM cannot extract).

bench/ only.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot_v2.bench.cwp.persistence import PCTSignature, cwp_priority
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode

_FACTUAL_PREDS: frozenset[str] = frozenset(
    {"is", "has", "works_at", "lives_in", "prefers", "dislikes", "moved_to"}
)


def _format_claim(atom: MemoryAtom) -> str:
    pred = atom.predicate.replace("_", " ")
    obj = atom.object_value or ""
    polarity_marker = "" if atom.polarity == "pos" else "NOT "
    tag = "fact" if atom.predicate in _FACTUAL_PREDS else "event"
    if atom.valid_from is not None and atom.valid_until is not None:
        try:
            vf = datetime.fromtimestamp(atom.valid_from, tz=UTC).date().isoformat()
            vu = datetime.fromtimestamp(atom.valid_until, tz=UTC).date().isoformat()
            window = f" (valid {vf}–{vu})"
        except (OSError, OverflowError, ValueError):
            window = ""
    else:
        window = ""
    return f"[{tag}] {polarity_marker}{atom.subject} {pred} {obj}{window}".rstrip()


def _format_observation(ep: RawEpisode, max_chars: int = 400) -> str:
    try:
        d = datetime.fromtimestamp(ep.timestamp, tz=UTC).date().isoformat()
        date_part = f"[{d}] "
    except (OSError, OverflowError, ValueError):
        date_part = ""
    speaker = ep.metadata.get("speaker_name") or ep.user_id or ep.speaker
    text = ep.text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return f"{date_part}{speaker}: {text}"


def render_pack_cwp(
    atoms: list[MemoryAtom],
    episodes_lookup: dict[str, RawEpisode],
    query: str,  # noqa: ARG001
    pct_signatures: dict[str, PCTSignature] | None = None,
    *,
    max_atoms: int = 18,
    max_observations_per_atom: int = 3,
    char_budget: int = 22_000,
) -> str:
    """Render atoms as derivation trees: claim → supporting raw observations.

    Atoms ranked by cwp_priority(atom, sig) when pct_signatures given,
    else by atom.regret_charge × atom.credence.
    """
    if not atoms:
        return ""

    if pct_signatures:
        scored = [
            (
                cwp_priority(a, pct_signatures.get(a.atom_id, _zero_sig(a.atom_id))),
                a,
            )
            for a in atoms
        ]
    else:
        scored = [(a.regret_charge * a.credence, a) for a in atoms]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [a for _, a in scored[:max_atoms]]

    blocks: list[str] = []
    total_chars = 0

    for i, atom in enumerate(top, start=1):
        claim_line = f"[{i}] {_format_claim(atom)}"
        obs_lines: list[str] = []
        for ep_id in list(atom.evidence_episodes)[:max_observations_per_atom]:
            ep = episodes_lookup.get(ep_id)
            if ep is None:
                continue
            obs_lines.append("    ← " + _format_observation(ep))
        block = claim_line + "\n" + "\n".join(obs_lines) if obs_lines else claim_line
        if total_chars + len(block) + 1 > char_budget:
            break
        blocks.append(block)
        total_chars += len(block) + 1

    return "\n".join(blocks)


def _zero_sig(atom_id: str) -> PCTSignature:
    return PCTSignature(atom_id=atom_id, persistence_0=0.0, betweenness=0.0, cycle_membership=0.0)


def render_pack_raw_annotated(
    ep_ids: list[str],
    episodes_lookup: dict[str, RawEpisode],
    atoms: list[MemoryAtom],
    *,
    char_budget: int = 22_000,
    per_turn_max: int = 1200,
    annotations_per_episode: int = 2,
) -> str:
    """Inverted CWP render: raw episodes primary, atoms as small annotations.

    For each episode in order, we render:
        [N] [DATE] Speaker: <raw turn text>
            facts: <up to K terse atom claims derived from this episode>

    This keeps the v1-style raw-text retrieval power (which scored 60.1%) while
    surfacing the structured atom-claims as a low-weight extra signal. Falsifies
    the strict CWP-as-primitive hypothesis if it doesn't beat plain raw render.
    """
    # Build episode → list[atom] index (only atoms whose evidence cites the episode)
    by_ep: dict[str, list[MemoryAtom]] = {}
    for atom in atoms:
        for eid in atom.evidence_episodes:
            by_ep.setdefault(eid, []).append(atom)

    parts: list[str] = []
    total_chars = 0
    for i, ep_id in enumerate(ep_ids, start=1):
        ep = episodes_lookup.get(ep_id)
        if ep is None:
            continue
        try:
            d = datetime.fromtimestamp(ep.timestamp, tz=UTC).date().isoformat()
            date_part = f"[{d}] "
        except (OSError, OverflowError, ValueError):
            date_part = ""
        speaker = ep.metadata.get("speaker_name") or ep.user_id or ep.speaker
        text = ep.text[:per_turn_max]
        head = f"[{i}] {date_part}{speaker}: {text}"

        ann_lines: list[str] = []
        for atom in (by_ep.get(ep_id) or [])[:annotations_per_episode]:
            ann_lines.append(f"      facts: {_format_claim(atom)[len('[fact] ') :]}")
        block = head + ("\n" + "\n".join(ann_lines) if ann_lines else "")

        if total_chars + len(block) + 1 > char_budget:
            break
        parts.append(block)
        total_chars += len(block) + 1

    return "\n".join(parts)
