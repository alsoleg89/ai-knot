from __future__ import annotations

import time
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput

_SEVERITY_APPLICABILITY: dict[str, float] = {
    "fatal": 0.1,
    "severe": 0.25,
    "major": 0.4,
    "minor": 0.65,
    "trivial": 0.8,
}


def _parse_applicability(content: str) -> float:
    """Infer applicability penalty from the severity of critique."""
    lower = content.lower()
    for token, score in _SEVERITY_APPLICABILITY.items():
        if token in lower:
            return score
    return 0.5  # neutral default


class CriticRole(BaseRole):
    name = "critic"

    def run(self, ctx: RoleContext) -> RoleOutput:
        theory_so_far = ctx.corpus.read_theory()
        recent_proofs = ctx.corpus.read_proofs(last_n=2)
        proofs_block = ""
        if recent_proofs:
            snippets = [p.get("content", "")[:150] for p in recent_proofs]
            proofs_block = "Recent proof results:\n" + "\n---\n".join(snippets) + "\n\n"

        recalled = ctx.recall(theory_so_far[:200], k=3, stream="critique")
        recall_block = ""
        if recalled:
            snippets = [
                str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:150]
                for r in recalled
            ]
            recall_block = (
                "Prior critiques on related theories:\n" + "\n---\n".join(snippets) + "\n\n"
            )
        system = (
            "You are Critic, an adversarial reviewer of multi-agent memory theories. "
            "Attack theory drafts and proofs: find hidden assumptions, counterexamples, "
            "logical gaps, and inapplicability cases. Be specific and constructive. "
            "Format: ATTACK | TARGET_CLAIM | SEVERITY (fatal/severe/major/minor/trivial) "
            "| FAILURE_MODE | PROPOSED_FIX"
        )
        user = (
            f"Research focus: {ctx.focus!r}.\n{proofs_block}{recall_block}"
            f"Current theory:\n{theory_so_far[:1500]}\n\n"
            "Identify the 2 strongest attacks against this theory."
        )
        resp = self.llm.chat(system, user)

        applicability = _parse_applicability(resp.content)
        candidates = ctx.corpus.read_theory_candidates(last_n=1)
        candidate_id = candidates[0]["candidate_id"] if candidates else "unknown"

        entry: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "focus": ctx.focus,
            "candidate_id": candidate_id,
            "applicability_signal": applicability,
            "content": resp.content,
        }
        ctx.corpus.append_critique(entry)

        # Update fitness with applicability signal (provability from latest proof, or neutral)
        fitness_index = ctx.corpus.read_fitness_index()
        provability = 0.5
        if fitness_index:
            top = fitness_index[0]
            if str(top.get("candidate_id", "")) == candidate_id:
                provability = float(top.get("provability", 0.5))

        fitness_record: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "candidate_id": candidate_id,
            "provability": provability,
            "applicability": applicability,
            "novelty": 0.5,
            "fitness": provability * 0.6 + applicability * 0.2 + 0.5 * 0.2,
        }
        ctx.corpus.append_fitness_record(fitness_record)

        return RoleOutput(
            role_name=self.name,
            summary=f"Critic: {resp.content[:120]}",
            tokens_used=resp.total_tokens,
            data={"candidate_id": candidate_id, "applicability": applicability},
        )
