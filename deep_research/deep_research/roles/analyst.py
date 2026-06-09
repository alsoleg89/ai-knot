from __future__ import annotations

import time
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput


class AnalystRole(BaseRole):
    name = "analyst"

    def run(self, ctx: RoleContext) -> RoleOutput:
        system = (
            "You are Analyst, a deep-reading researcher. "
            "Extract structured claims, constructions, and proof techniques from sources. "
            "Format: CLAIM | EVIDENCE | IMPLICATION_FOR_MULTI_AGENT_MEMORY"
        )
        theory_so_far = ctx.corpus.read_theory()
        recall_block = ctx.recall_block(
            ctx.focus, k=3, stream="sources", header="Semantically relevant past sources:"
        )
        user = (
            f"Research focus: {ctx.focus!r}. "
            f"Current theory (excerpt):\n{theory_so_far[:800]}\n\n"
            f"{recall_block}"
            "Extract 3 key claims from related literature relevant to this focus."
        )
        resp = self.llm.chat(system, user)
        entry: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "type": "analysis",
            "focus": ctx.focus,
            "content": resp.content,
        }
        ctx.corpus.append_source(entry)
        return RoleOutput(
            role_name=self.name,
            summary=f"Analyst: {resp.content[:120]}",
            tokens_used=resp.total_tokens,
            data={"claims": resp.content},
        )
