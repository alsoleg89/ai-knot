"""S16 — Explicit Update Semantics (MemoryOp DELETE / NOOP / UPDATE).

Verifies that the MemoryOp signals emitted by the extractor are correctly
honoured during slot resolution in KnowledgeBase.learn():

  DELETE — "I no longer work at Acme" → close the matched slot, no insert.
  NOOP   — "still the same salary" → skip entirely, no insert, no mutation.
  UPDATE — same value_text as existing but conversation corrects context →
            force supersede (override structural "reinforce").

Only runs against AiKnotBackend (single-agent) which exposes ``_kb``.
Returns a skipped result for all other backends.

Metrics (deterministic):
  delete_correctness — 1.0 if DELETE closes the slot without inserting
  noop_correctness   — 1.0 if NOOP neither inserts nor mutates existing
  update_correctness — 1.0 if UPDATE forces supersede over reinforce
"""

from __future__ import annotations

from unittest.mock import patch

from ai_knot.types import ConversationTurn, Fact, MemoryOp, MemoryType
from tests.eval.benchmark.base import MemoryBackend, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s16_update_correctness"

_TURN = [ConversationTurn(role="user", content="x")]


def _slotted(content: str, *, slot_key: str, value_text: str, op: MemoryOp) -> Fact:
    return Fact(
        content=content,
        type=MemoryType.SEMANTIC,
        importance=0.8,
        slot_key=slot_key,
        value_text=value_text,
        op=op,
    )


async def run(backend: MemoryBackend, judge: BaseJudge) -> ScenarioResult:
    await backend.reset()

    kb = getattr(backend, "_kb", None)
    if kb is None:
        return ScenarioResult(
            scenario_id=SCENARIO_ID,
            backend_name=backend.name,
            judge_scores={},
            insert_result=None,
            retrieval_result=None,
            notes="backend does not expose _kb; skipped",
        )

    # ------------------------------------------------------------------ DELETE
    kb._storage.save(kb._agent_id, [])
    old = kb.add("Alex works at Acme")
    old.slot_key = "Alex::employer"
    old.value_text = "Acme"
    kb._storage.save(kb._agent_id, [old])

    delete_fact = _slotted(
        "Alex no longer works at Acme",
        slot_key="Alex::employer",
        value_text="Acme",
        op=MemoryOp.DELETE,
    )
    with patch("ai_knot.knowledge.Extractor.extract", return_value=[delete_fact]):
        inserted = kb.learn(_TURN, api_key="fake")

    delete_ok = len(inserted) == 0 and all(
        f.valid_until is not None for f in kb.list_facts() if f.slot_key == "Alex::employer"
    )

    # ------------------------------------------------------------------ NOOP
    kb._storage.save(kb._agent_id, [])
    old2 = kb.add("Alex earns 95k")
    old2.slot_key = "Alex::salary"
    old2.value_text = "95000"
    old2.state_confidence = 0.9
    kb._storage.save(kb._agent_id, [old2])

    noop_fact = _slotted(
        "Alex salary still 95k",
        slot_key="Alex::salary",
        value_text="95000",
        op=MemoryOp.NOOP,
    )
    with patch("ai_knot.knowledge.Extractor.extract", return_value=[noop_fact]):
        inserted2 = kb.learn(_TURN, api_key="fake")

    stored_after_noop = kb.list_facts()
    noop_ok = (
        len(inserted2) == 0
        and len(stored_after_noop) == 1
        and abs(stored_after_noop[0].state_confidence - 0.9) < 1e-9
    )

    # ------------------------------------------------------------------ UPDATE
    kb._storage.save(kb._agent_id, [])
    old3 = kb.add("Alex earns 95k")
    old3.slot_key = "Alex::salary"
    old3.value_text = "95000"
    old3.version = 0
    kb._storage.save(kb._agent_id, [old3])

    update_fact = _slotted(
        "Alex salary confirmed 95000 after grade review",
        slot_key="Alex::salary",
        value_text="95000",
        op=MemoryOp.UPDATE,
    )
    with patch("ai_knot.knowledge.Extractor.extract", return_value=[update_fact]):
        inserted3 = kb.learn(_TURN, api_key="fake")

    update_ok = len(inserted3) == 1 and inserted3[0].version == 1

    scores = {
        "delete_correctness": [1.0 if delete_ok else 0.0],
        "noop_correctness": [1.0 if noop_ok else 0.0],
        "update_correctness": [1.0 if update_ok else 0.0],
    }

    notes = (
        f"delete={'pass' if delete_ok else 'FAIL'}, "
        f"noop={'pass' if noop_ok else 'FAIL'}, "
        f"update={'pass' if update_ok else 'FAIL'}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
