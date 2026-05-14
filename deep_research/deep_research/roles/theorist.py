from __future__ import annotations

import time
import uuid
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput


class TheoristRole(BaseRole):
    name = "theorist"

    def run(self, ctx: RoleContext) -> RoleOutput:
        system = (
            "You are Theorist, a mathematical theory builder for multi-agent memory systems. "
            "Propose and refine candidate theories. Each theory must include: "
            "TITLE | DEFINITIONS | CORE_PROPOSITION | APPLICABILITY | FITNESS_SCORE (0.0-1.0). "
            "Maintain a population of competing theories; evolve the fittest."
        )
        theory_so_far = ctx.corpus.read_theory()
        if ctx.phase == "evolve":
            fitness_index = ctx.corpus.read_fitness_index()
            top_candidates = ctx.corpus.read_theory_candidates(last_n=3)
            fitness_block = ""
            if fitness_index:
                lines = [
                    f"  [{r['candidate_id']}] fitness={r.get('fitness', 0):.2f} "
                    f"prov={r.get('provability', 0):.2f} app={r.get('applicability', 0):.2f}"
                    for r in fitness_index[:3]
                ]
                fitness_block = "Fitness index (top 3):\n" + "\n".join(lines) + "\n"
            candidates_block = ""
            if top_candidates:
                snippets = [
                    f"[{c['candidate_id']}]: {c.get('content', '')[:300]}" for c in top_candidates
                ]
                candidates_block = "Recent candidates:\n" + "\n---\n".join(snippets) + "\n"
            recalled_critiques = ctx.recall(theory_so_far[:300], k=3, stream="critique")
            critique_recall_block = ""
            if recalled_critiques:
                snippets = [
                    str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:150]
                    for r in recalled_critiques
                ]
                critique_recall_block = (
                    "Relevant past critiques:\n" + "\n---\n".join(snippets) + "\n\n"
                )
            action = (
                "Refine the existing theory based on critique and proof results. "
                "Cross-pollinate the strongest elements from competing candidates. "
                "Produce a new evolved candidate with improved FITNESS_SCORE."
            )
            user = (
                f"Research focus: {ctx.focus!r}. Phase: evolve.\n"
                f"{fitness_block}{candidates_block}{critique_recall_block}"
                f"Current leading theory:\n{theory_so_far[:800]}\n\n{action}"
            )
        else:
            recalled = ctx.recall(ctx.focus, k=3)
            recall_block = ""
            if recalled:
                snippets = [
                    str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:150]
                    for r in recalled
                ]
                recall_block = "Related past corpus entries:\n" + "\n---\n".join(snippets) + "\n\n"
            action = "Propose a novel theory candidate for multi-agent memory."
            user = (
                f"Research focus: {ctx.focus!r}. Phase: {ctx.phase}. "
                f"Current theory:\n{theory_so_far[:1000]}\n\n{recall_block}{action}"
            )
        resp = self.llm.chat(system, user)
        candidate_id = str(uuid.uuid4())[:8]
        candidate: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "candidate_id": candidate_id,
            "focus": ctx.focus,
            "content": resp.content,
        }
        ctx.corpus.append_theory_candidate(candidate)
        ctx.corpus.write_theory(
            f"# Theory — tick {ctx.tick}\n\nFocus: {ctx.focus}\n\n{resp.content}"
        )
        return RoleOutput(
            role_name=self.name,
            summary=f"Theorist candidate {candidate_id}: {resp.content[:120]}",
            tokens_used=resp.total_tokens,
            data={"candidate_id": candidate_id},
        )
