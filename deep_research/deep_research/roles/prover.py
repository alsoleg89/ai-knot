from __future__ import annotations

import time
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput

_PROVABILITY: dict[str, float] = {"proved": 1.0, "open": 0.5, "disproved": 0.0}


def _parse_result(content: str) -> str:
    """Extract proof result token (proved/disproved/open) from structured LLM output."""
    lower = content.lower()
    for token in ("disproved", "proved", "open"):
        if token in lower:
            return token
    return "open"


class ProverRole(BaseRole):
    name = "prover"

    def run(self, ctx: RoleContext) -> RoleOutput:
        theory_so_far = ctx.corpus.read_theory()
        recent_proofs = ctx.corpus.read_proofs(last_n=3)
        proofs_block = ""
        if recent_proofs:
            snippets = [p.get("content", "")[:200] for p in recent_proofs]
            proofs_block = "Recent proof attempts:\n" + "\n---\n".join(snippets) + "\n\n"

        recalled = ctx.recall(ctx.focus, k=3, stream="sources")
        recall_block = ""
        if recalled:
            snippets = [
                str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:150]
                for r in recalled
            ]
            recall_block = (
                "Relevant sources (proof techniques):\n" + "\n---\n".join(snippets) + "\n\n"
            )
        system = (
            "You are Prover, a mathematical proof verifier for multi-agent memory theories. "
            "Pick the most critical unresolved proposition from the current theory and attempt "
            "to prove it or find a counterexample. Mark unresolved claims as OPEN CONJECTURE. "
            "Format your response as: "
            "PROPOSITION | APPROACH | RESULT (proved/disproved/open) | PROOF_SKETCH"
        )
        user = (
            f"Research focus: {ctx.focus!r}.\n{proofs_block}{recall_block}"
            f"Current theory:\n{theory_so_far[:1500]}\n\n"
            "Select the most critical proposition and attempt to prove it."
        )
        resp = self.llm.chat(system, user)

        result_token = _parse_result(resp.content)
        provability = _PROVABILITY[result_token]

        # Attribute proof to the latest theory candidate if available
        candidates = ctx.corpus.read_theory_candidates(last_n=1)
        candidate_id = candidates[0]["candidate_id"] if candidates else "unknown"

        entry: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "focus": ctx.focus,
            "candidate_id": candidate_id,
            "result": result_token,
            "provability": provability,
            "content": resp.content,
        }
        ctx.corpus.append_proof(entry)

        # Emit fitness signal so Theorist can select fittest candidates
        fitness_record: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "candidate_id": candidate_id,
            "provability": provability,
            "result": result_token,
            # applicability and novelty scored by Experimenter/Critic later; default neutral
            "applicability": 0.5,
            "novelty": 0.5,
            "fitness": provability * 0.6 + 0.5 * 0.2 + 0.5 * 0.2,
        }
        ctx.corpus.append_fitness_record(fitness_record)

        return RoleOutput(
            role_name=self.name,
            summary=f"Prover [{result_token}] {resp.content[:100]}",
            tokens_used=resp.total_tokens,
            data={
                "candidate_id": candidate_id,
                "result": result_token,
                "provability": provability,
            },
        )
