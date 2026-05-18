from __future__ import annotations

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput

_PHASE_GUIDANCE: dict[str, str] = {
    "generate": "This is a GENERATE phase: expand the hypothesis space. Prioritize breadth.",
    "prove": "This is a PROVE phase: push for formalization. Prioritize proof attempts.",
    "critique": "This is a CRITIQUE phase: find weaknesses. Prioritize adversarial angles.",
    "evolve": "This is an EVOLVE phase: synthesize gains. Prioritize theory refinement.",
}


class DirectorRole(BaseRole):
    name = "director"

    def run(self, ctx: RoleContext) -> RoleOutput:
        phase_hint = _PHASE_GUIDANCE.get(ctx.phase, "")

        # Build maturity summary from fitness index
        fitness_index = ctx.corpus.read_fitness_index()
        maturity_line = ""
        if fitness_index:
            top = fitness_index[0]
            top_fitness = float(top.get("fitness", 0.0))
            top_result = str(top.get("result", "open"))
            maturity_line = (
                f"Leading theory fitness={top_fitness:.2f} (proof result: {top_result}). "
            )
            if top_fitness >= 0.7:
                maturity_line += "MATURITY THRESHOLD APPROACHING — consider finalizing theory. "

        # Build dead-ends block from disproved proofs
        disproved = ctx.corpus.read_disproved_hypotheses(last_n=15)
        dead_block = ""
        if disproved:
            snippets = [d[:120] for d in disproved]
            dead_block = (
                "DEAD ENDS — already formally disproved, do NOT revisit:\n"
                + "\n".join(f"  - {s}" for s in snippets)
                + "\n"
            )

        # Convergence nudge after many ticks
        convergence_hint = ""
        if ctx.tick >= 36:
            convergence_hint = (
                "CONVERGENCE MODE: enough exploration done. "
                "Push the surviving open hypothesis toward a concrete combined architecture, "
                "a precision-adjusted bound, and a runnable prototype on hard data "
                "(≥5000 utterances, baseline recall@60 < 70%). "
            )

        recalled = ctx.recall(ctx.focus, k=3)
        recall_line = ""
        if recalled:
            snippets = [
                str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:80].strip()
                for r in recalled[:2]
            ]
            recall_line = "Recent relevant work: " + " | ".join(snippets) + ". "
        system = (
            "You are the Director of a deep research campaign on long-dialogue memory, "
            "inference-time indexing, and retrieval. Keep the original campaign brief as "
            "the invariant objective; use the current focus only as the next local step. "
            "Synthesize the current state and set the research focus for the next tick. "
            f"{phase_hint} {convergence_hint}Be specific, actionable, and benchmarkable."
        )
        user = (
            f"{ctx.brief_block(max_chars=1400)}"
            f"Tick {ctx.tick} | Phase: {ctx.phase}. {maturity_line}{recall_line}"
            f"{dead_block}"
            f"Current focus: {ctx.focus!r}. "
            "What should the research focus be for the next tick? "
            "Reply with one concise sentence."
        )
        resp = self.llm.chat(system, user)
        new_focus = resp.content.strip()
        return RoleOutput(
            role_name=self.name,
            summary=new_focus,
            tokens_used=resp.total_tokens,
            data={"new_focus": new_focus},
        )
