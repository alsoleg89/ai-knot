"""Sprint 6 — v2 LOCOMO 2-conv benchmark runner.

Runs MemoryAPI (pure Python, no MCP) on N conversations from locomo10.json.
Computes 8-metric scorecard per question, aggregates by category.

Usage:
    .venv/bin/python -m ai_knot_v2.bench.v2_locomo_runner
    .venv/bin/python -m ai_knot_v2.bench.v2_locomo_runner --convs 2 --data /path/to/locomo10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_knot_v2.api.product import MemoryAPI
from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest, RecallRequest
from ai_knot_v2.bench.scorecard import compute_scorecard
from ai_knot_v2.core.types import ReaderBudget

_DEFAULT_DATA = Path(__file__).parents[5] / "aiknotbench" / "data" / "locomo10.json"

_DEFAULT_BUDGET = ReaderBudget(
    max_atoms=60,
    max_tokens=8000,
    require_dependency_closure=True,
)


# ---------------------------------------------------------------------------
# LOCOMO JSON parsing
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    session_num: int
    timestamp: int


@dataclass
class QAPair:
    question: str
    answer: str
    evidence: list[str]
    category: int


@dataclass
class LocomoConvData:
    conv_idx: int
    speaker_a: str
    speaker_b: str
    turns: list[Turn]
    qa_pairs: list[QAPair]


def parse_locomo_json(data_path: Path, limit: int = 2) -> list[LocomoConvData]:
    """Parse locomo10.json into structured conversations."""
    with open(data_path) as f:
        raw: list[dict[str, Any]] = json.load(f)

    convs: list[LocomoConvData] = []
    base_ts = int(time.time()) - 365 * 24 * 3600  # ~1 year ago

    for idx, item in enumerate(raw[:limit]):
        conv = item["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        turns: list[Turn] = []
        # Collect turns from all session_N keys
        for snum in range(1, 40):
            key = f"session_{snum}"
            if key not in conv:
                continue
            session_ts = base_ts + (idx * 30 + snum) * 86400
            for tidx, t in enumerate(conv[key]):
                turns.append(
                    Turn(
                        dia_id=t["dia_id"],
                        speaker=t.get("speaker", speaker_a),
                        text=t.get("text", ""),
                        session_num=snum,
                        timestamp=session_ts + tidx * 60,
                    )
                )

        qa_pairs = [
            QAPair(
                question=q["question"],
                answer=q.get("answer") or q.get("adversarial_answer", ""),
                evidence=q.get("evidence", []),
                category=q.get("category", 1),
            )
            for q in item.get("qa", [])
        ]

        convs.append(
            LocomoConvData(
                conv_idx=idx,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                turns=turns,
                qa_pairs=qa_pairs,
            )
        )

    return convs


# ---------------------------------------------------------------------------
# Per-conversation benchmark
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    category: int
    gold_evidence_coverage: float
    required_atom_recall: float
    context_dilution_rate: float
    atoms_retrieved: int


@dataclass
class ConvResult:
    conv_idx: int
    total_atoms: int
    total_turns: int
    question_results: list[QuestionResult] = field(default_factory=list)


def run_conversation(conv: LocomoConvData) -> ConvResult:
    """Ingest conversation turns, score all QA pairs."""
    api = MemoryAPI(db_path=":memory:")

    # Build episodes preserving order; track dia_id → episode index
    episodes: list[EpisodeIn] = []
    dia_id_order: list[str] = []

    for turn in conv.turns:
        if not turn.text.strip():
            continue
        episodes.append(
            EpisodeIn(
                text=turn.text,
                speaker="user",
                user_id=turn.speaker,  # carry real speaker name for first-person resolution
                session_id=f"conv-{conv.conv_idx}-s{turn.session_num}",
                timestamp=turn.timestamp,
            )
        )
        dia_id_order.append(turn.dia_id)

    learn_resp = api.learn(LearnRequest(episodes=episodes))

    # Map dia_id → episode_id (positional alignment)
    dia_to_ep: dict[str, str] = {
        did: eid for did, eid in zip(dia_id_order, learn_resp.episode_ids, strict=False)
    }

    all_atoms = api._library.all_atoms()  # noqa: SLF001
    total_atoms = len(all_atoms)

    question_results: list[QuestionResult] = []

    for qa in conv.qa_pairs:
        # Gold episode IDs from evidence pointers (DX:Y)
        gold_ep_ids: set[str] = set()
        for ev in qa.evidence:
            ep_id = dia_to_ep.get(ev)
            if ep_id:
                gold_ep_ids.add(ep_id)

        # Gold atom IDs: atoms whose evidence_episodes overlap with gold episodes
        gold_atom_ids: set[str] = {
            a.atom_id for a in all_atoms if gold_ep_ids & set(a.evidence_episodes)
        }

        # Recall
        recall_resp = api.recall(RecallRequest(query=qa.question, max_atoms=60, max_tokens=8000))
        result_atom_ids = {a.atom_id for a in recall_resp.atoms}
        result_atoms = [a for a in all_atoms if a.atom_id in result_atom_ids]

        sc = compute_scorecard(
            result_atoms=result_atoms,
            all_atoms=all_atoms,
            query=qa.question,
            budget=_DEFAULT_BUDGET,
            gold_atom_ids=gold_atom_ids if gold_atom_ids else None,
            gold_episode_ids=gold_ep_ids if gold_ep_ids else None,
        )

        question_results.append(
            QuestionResult(
                question=qa.question[:80],
                category=qa.category,
                gold_evidence_coverage=sc.gold_evidence_coverage,
                required_atom_recall=sc.required_atom_recall,
                context_dilution_rate=sc.context_dilution_rate,
                atoms_retrieved=len(result_atoms),
            )
        )

    return ConvResult(
        conv_idx=conv.conv_idx,
        total_atoms=total_atoms,
        total_turns=len(episodes),
        question_results=question_results,
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def aggregate(results: list[ConvResult]) -> dict[str, Any]:
    by_cat: dict[int, list[float]] = defaultdict(list)
    all_gec: list[float] = []
    all_recall: list[float] = []

    for cr in results:
        for qr in cr.question_results:
            by_cat[qr.category].append(qr.gold_evidence_coverage)
            all_gec.append(qr.gold_evidence_coverage)
            all_recall.append(qr.required_atom_recall)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "total_questions": len(all_gec),
        "overall_gec": avg(all_gec),
        "overall_recall": avg(all_recall),
        "by_category": {
            cat: {"n": len(scores), "gec": avg(scores)} for cat, scores in sorted(by_cat.items())
        },
        "cat1_gec": avg(by_cat.get(1, [])),
    }


def report(results: list[ConvResult], agg: dict[str, Any]) -> None:
    print("\n=== ai-knot v2 LOCOMO Sprint-6 BG-run ===")
    for cr in results:
        print(f"\nConv {cr.conv_idx}: {cr.total_atoms} atoms from {cr.total_turns} turns")

    print(f"\nTotal questions: {agg['total_questions']}")
    print(f"Overall GoldEvidenceCoverage: {agg['overall_gec']:.3f}")
    print(f"Overall RequiredAtomRecall:   {agg['overall_recall']:.3f}")
    print("\nBy category (GoldEvidenceCoverage):")
    for cat, info in agg["by_category"].items():
        label = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "adversarial"}.get(
            cat, f"cat{cat}"
        )
        print(f"  cat{cat} ({label}): {info['gec']:.3f}  (n={info['n']})")

    cat1_gec = agg["cat1_gec"]
    gate_pass = cat1_gec >= 0.35
    gate_label = "PASS" if gate_pass else "FAIL (diagnostic needed)"
    print(f"\nSprint-6 gate: cat1 GEC ≥ 0.35 → {cat1_gec:.3f}  [{gate_label}]")

    if not gate_pass:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ai-knot v2 LOCOMO Sprint-6 runner")
    parser.add_argument(
        "--convs", type=int, default=2, help="Number of conversations to run (default 2)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help="Path to locomo10.json",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.convs} conversation(s) from {args.data} ...")
    convs = parse_locomo_json(args.data, limit=args.convs)

    results: list[ConvResult] = []
    for conv in convs:
        n_turns = len(conv.turns)
        n_qa = len(conv.qa_pairs)
        print(f"Running conv {conv.conv_idx} ({n_turns} turns, {n_qa} QA) ...")
        cr = run_conversation(conv)
        results.append(cr)

    agg = aggregate(results)
    report(results, agg)


if __name__ == "__main__":
    main()
