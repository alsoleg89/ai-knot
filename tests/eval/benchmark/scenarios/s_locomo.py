"""S-LoCoMo — Long-Context Memory QA benchmark (LoCoMo10 subset).

Evaluates memory recall quality against the LoCoMo10 dataset
(snap-research/locomo, CC BY-NC 4.0).  10 conversations, ~199 QA pairs each.

Protocol:
  1. For each conversation, insert every session turn via ``backend.insert()``.
  2. Reset per-session state, then for each QA pair retrieve top-5 facts.
  3. Compute best-doc token F1: max over retrieved texts of F1(text, gold_answer).
  4. Aggregate by question category:
       1 → single_hop_f1
       2 → multi_hop_f1
       3 → temporal_f1
       4 → open_ended_f1
       5 → adversarial_f1
     overall_f1 = mean over all pairs.

Dataset is downloaded on first run and cached in the system temp directory.
Pass ``--locomo-file /path/to/locomo10.json`` to use a local copy.

Metrics:
  overall_f1          — mean best-doc token F1 over all 10 conversations
  single_hop_f1       — category 1 questions
  multi_hop_f1        — category 2 questions
  temporal_f1         — category 3 questions
  open_ended_f1       — category 4 questions
  adversarial_f1      — category 5 questions (uses adversarial_answer as gold)
  evidence_recall_at5 — fraction of evidence turns found in top-5 retrieved texts
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_knot.tokenizer import tokenize
from tests.eval.benchmark._eval_utils import hit_rank_lexical
from tests.eval.benchmark.base import MemoryBackend, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s_locomo"

_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
_CACHE_PATH = Path(tempfile.gettempdir()) / "ai_knot_locomo10.json"

# Category integer → metric name mapping (LoCoMo paper convention).
_CAT_NAME: dict[int, str] = {
    1: "single_hop_f1",
    2: "multi_hop_f1",
    3: "temporal_f1",
    4: "open_ended_f1",
    5: "adversarial_f1",
}

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _load_locomo(local_path: str | None = None) -> list[dict[str, Any]]:
    """Load LoCoMo10 JSON, downloading and caching if needed."""
    if local_path:
        with open(local_path, encoding="utf-8") as fh:
            result: list[dict[str, Any]] = json.load(fh)
            return result

    env_path = os.environ.get("LOCOMO_FILE")
    if env_path and Path(env_path).is_file():
        with open(env_path, encoding="utf-8") as fh:
            result = json.load(fh)
            return result

    if _CACHE_PATH.is_file():
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            result = json.load(fh)
            return result

    # Download to cache.
    urllib.request.urlretrieve(_LOCOMO_URL, _CACHE_PATH)  # noqa: S310
    with open(_CACHE_PATH, encoding="utf-8") as fh:
        result = json.load(fh)
        return result


def _iter_turns(sample: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Flatten all sessions in a LoCoMo sample into turn strings and a dia_id map.

    The real LoCoMo schema stores sessions inside ``sample["conversation"]``
    as ``session_1``, ``session_2``, …  alongside ``session_N_date_time`` keys
    and ``speaker_a`` / ``speaker_b``.  We extract only ``session_N`` lists,
    sort them by *N*, and yield each turn.

    Returns:
        Tuple of (turn_texts, dia_map) where dia_map maps ``dia_id`` to the
        corresponding ``"speaker: text"`` string for evidence-based evaluation.
    """
    conversation = sample.get("conversation", sample)
    # Collect (session_number, key) pairs and sort by number.
    numbered: list[tuple[int, str]] = []
    for key in conversation:
        m = _SESSION_KEY_RE.match(key)
        if m:
            numbered.append((int(m.group(1)), key))
    numbered.sort()

    turns: list[str] = []
    dia_map: dict[str, str] = {}
    for _n, key in numbered:
        session = conversation[key]
        if not isinstance(session, list):
            continue
        for turn in session:
            if isinstance(turn, dict) and "text" in turn:
                speaker = turn.get("speaker", "speaker")
                turn_text = f"{speaker}: {turn['text']}"
                turns.append(turn_text)
                dia_id = turn.get("dia_id")
                if isinstance(dia_id, str):
                    dia_map[dia_id] = turn_text
    return turns, dia_map


def _best_f1_against(retrieved_texts: list[str], gold: str) -> float:
    """Max token F1 between any retrieved text and the gold answer.

    Gold tokens are computed once and reused across all retrieved texts.
    """
    if not retrieved_texts:
        return 0.0
    gold_tokens = tokenize(gold)
    if not gold_tokens:
        return 0.0
    gold_set = set(gold_tokens)
    best = 0.0
    for text in retrieved_texts:
        pred_tokens = tokenize(text)
        if not pred_tokens:
            continue
        pred_set = set(pred_tokens)
        common = pred_set & gold_set
        if not common:
            continue
        precision = len(common) / len(pred_set)
        recall = len(common) / len(gold_set)
        f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def _evidence_recall_at_k(
    retrieved_texts: list[str],
    evidence_ids: list[str],
    dia_map: dict[str, str],
    *,
    threshold: float = 0.5,
) -> float:
    """Fraction of evidence turns found in the retrieved texts.

    Each ``evidence_id`` is resolved to its original turn text via *dia_map*.
    A hit is counted when ``hit_rank_lexical`` (ATC-based, deterministic)
    finds the evidence text among *retrieved_texts* at any rank.

    Returns 0.0 when no evidence IDs can be resolved.
    """
    resolved = [dia_map[eid] for eid in evidence_ids if eid in dia_map]
    if not resolved:
        return 0.0
    hits = sum(
        1
        for ev_text in resolved
        if hit_rank_lexical(ev_text, retrieved_texts, threshold=threshold) is not None
    )
    return hits / len(resolved)


async def run(
    backend: MemoryBackend,
    judge: BaseJudge,
    *,
    locomo_file: str | None = None,
    max_conversations: int | None = None,
    max_qa_per_conv: int | None = None,
) -> ScenarioResult:
    """Run the LoCoMo QA evaluation.

    Args:
        backend: Memory backend to evaluate.
        judge: Unused (deterministic metric); kept for interface compatibility.
        locomo_file: Optional local path to ``locomo10.json``. Falls back to
            the ``LOCOMO_FILE`` env var, then downloads from GitHub.
        max_conversations: Limit number of conversations (for fast CI runs).
        max_qa_per_conv: Limit QA pairs per conversation (for fast CI runs).
    """
    try:
        dataset = _load_locomo(locomo_file)
    except Exception as exc:
        return ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={},
            insert_result=None,
            retrieval_result=None,
            notes=f"locomo10.json unavailable: {exc}",
        )

    if max_conversations is not None:
        dataset = dataset[:max_conversations]

    # f1_by_cat[category] → list of per-pair F1 scores
    f1_by_cat: dict[int, list[float]] = defaultdict(list)
    all_f1: list[float] = []
    all_evidence_recall: list[float] = []
    total_turns_ingested = 0
    total_qa = 0
    total_qa_scored = 0
    total_empty_retrievals = 0

    for conv in dataset:
        await backend.reset()

        # Phase 1: ingest all turns.
        turn_texts, dia_map = _iter_turns(conv)
        for turn_text in turn_texts:
            await backend.insert(turn_text)
        total_turns_ingested += len(turn_texts)

        # Phase 2: answer QA pairs.
        qa_pairs: list[dict[str, Any]] = conv.get("qa", [])
        if max_qa_per_conv is not None:
            qa_pairs = qa_pairs[:max_qa_per_conv]
        total_qa += len(qa_pairs)

        for qa in qa_pairs:
            question = str(qa.get("question", ""))
            category = int(qa.get("category", 0))

            # Category 5 (adversarial): gold is adversarial_answer.
            if category == 5:
                gold = str(qa.get("adversarial_answer", qa.get("answer", "")))
            else:
                gold = str(qa.get("answer", ""))

            if not question or not gold:
                continue

            await backend.reset_session()
            result = await backend.retrieve(question, top_k=5)

            if not result.texts:
                total_empty_retrievals += 1

            f1 = _best_f1_against(result.texts, gold)

            all_f1.append(f1)
            total_qa_scored += 1
            if category in _CAT_NAME:
                f1_by_cat[category].append(f1)

            # Evidence-based retrieval recall (use enriched texts when available).
            evidence_ids: list[str] = qa.get("evidence", [])
            if evidence_ids and dia_map:
                ev_texts = result.evidence_texts or result.texts
                ev_recall = _evidence_recall_at_k(ev_texts, evidence_ids, dia_map)
                all_evidence_recall.append(ev_recall)

    if not all_f1:
        return ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={},
            insert_result=None,
            retrieval_result=None,
            notes="no QA pairs evaluated",
        )

    overall = sum(all_f1) / len(all_f1)
    scores: dict[str, list[float]] = {"overall_f1": [overall]}
    for cat_id, name in _CAT_NAME.items():
        if cat_id in f1_by_cat:
            cat_scores = f1_by_cat[cat_id]
            scores[name] = [sum(cat_scores) / len(cat_scores)]

    ev_recall_mean = 0.0
    if all_evidence_recall:
        ev_recall_mean = sum(all_evidence_recall) / len(all_evidence_recall)
        scores["evidence_recall_at5"] = [ev_recall_mean]

    n_convs = len(dataset)
    notes = (
        f"conversations={n_convs}, turns_ingested={total_turns_ingested}, "
        f"qa_total={total_qa}, qa_scored={total_qa_scored}, "
        f"empty_retrievals={total_empty_retrievals}, overall_f1={overall:.3f}, "
        f"evidence_recall@5={ev_recall_mean:.3f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
