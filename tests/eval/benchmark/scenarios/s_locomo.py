"""S-LoCoMo — Long-Context Memory QA benchmark (LoCoMo10 subset).

Evaluates memory recall quality against the LoCoMo10 dataset
(snap-research/locomo, CC BY-NC 4.0).  10 conversations, ~199 QA pairs each.

Protocol:
  1. For each conversation, insert every session turn via ``backend.insert()``.
  2. For each QA pair, retrieve top-5 facts via ``backend.retrieve()``.
  3. Compute best-doc token F1: max over retrieved texts of F1(text, gold_answer).
  4. Aggregate by question category:
       1 → single_hop_f1
       2 → multi_hop_f1
       3 → temporal_f1
       4 → adversarial_f1   (if present in dataset)
     overall_f1 = mean over all pairs.

Dataset is downloaded on first run and cached in the system temp directory.
Pass ``--locomo-file /path/to/locomo10.json`` to use a local copy.

Metrics:
  overall_f1        — mean best-doc token F1 over all 10 conversations
  single_hop_f1     — category 1 questions
  multi_hop_f1      — category 2 questions
  temporal_f1       — category 3 questions
  adversarial_f1    — category 4 questions (if present; else omitted)
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

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
    4: "adversarial_f1",
}


def _load_locomo(local_path: str | None = None) -> list[dict]:
    """Load LoCoMo10 JSON, downloading and caching if needed."""
    if local_path:
        with open(local_path, encoding="utf-8") as fh:
            return json.load(fh)

    env_path = os.environ.get("LOCOMO_FILE")
    if env_path and Path(env_path).is_file():
        with open(env_path, encoding="utf-8") as fh:
            return json.load(fh)

    if _CACHE_PATH.is_file():
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    # Download to cache.
    urllib.request.urlretrieve(_LOCOMO_URL, _CACHE_PATH)  # noqa: S310
    with open(_CACHE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _iter_turns(conv: dict) -> list[str]:
    """Flatten all sessions in a conversation into a list of speaker: text strings."""
    turns: list[str] = []
    for key, val in conv.items():
        if not key.startswith("session_") or "_date" in key or not isinstance(val, list):
            continue
        for turn in val:
            if isinstance(turn, dict) and "text" in turn:
                speaker = turn.get("speaker", "speaker")
                turns.append(f"{speaker}: {turn['text']}")
    return turns


def _best_f1_against(retrieved_texts: list[str], gold: str) -> float:
    """Max token F1 between any retrieved text and the gold answer.

    Gold tokens are computed once and reused across all retrieved texts.
    """
    if not retrieved_texts:
        return 0.0
    gold_tokens = gold.lower().split()
    if not gold_tokens:
        return 0.0
    gold_set = set(gold_tokens)
    best = 0.0
    for text in retrieved_texts:
        pred_tokens = text.lower().split()
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

    for conv in dataset:
        await backend.reset()

        # Phase 1: ingest all turns.
        for turn_text in _iter_turns(conv):
            await backend.insert(turn_text)

        # Phase 2: answer QA pairs.
        qa_pairs: list[dict] = conv.get("qa", [])
        if max_qa_per_conv is not None:
            qa_pairs = qa_pairs[:max_qa_per_conv]

        for qa in qa_pairs:
            question = str(qa.get("question", ""))
            gold = str(qa.get("answer", ""))
            category = int(qa.get("category", 0))

            if not question or not gold:
                continue

            result = await backend.retrieve(question, top_k=5)
            f1 = _best_f1_against(result.texts, gold)

            all_f1.append(f1)
            if category in _CAT_NAME:
                f1_by_cat[category].append(f1)

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

    n_convs = len(dataset)
    n_qa = len(all_f1)
    notes = f"conversations={n_convs}, qa_pairs={n_qa}, overall_f1={overall:.3f}"

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
