from __future__ import annotations

import time
from typing import TYPE_CHECKING

from deep_research.config import CampaignConfig
from deep_research.corpus import CampaignState, Corpus
from deep_research.llm import LLMClient
from deep_research.roles.analyst import AnalystRole
from deep_research.roles.base import BaseRole, RoleContext
from deep_research.roles.critic import CriticRole
from deep_research.roles.director import DirectorRole
from deep_research.roles.experimenter import ExperimenterRole
from deep_research.roles.prover import ProverRole
from deep_research.roles.scout import ScoutRole
from deep_research.roles.theorist import TheoristRole

if TYPE_CHECKING:
    from deep_research.memory import SemanticMemory

ROLE_CYCLE = ("scout", "analyst", "theorist", "prover", "critic", "experimenter")

# Maps position within ROLE_CYCLE to research phase.
# generate(0-2) → prove(3) → critique(4) → evolve(5)
_PHASE_MAP: dict[int, str] = {
    0: "generate",
    1: "generate",
    2: "generate",
    3: "prove",
    4: "critique",
    5: "evolve",
}


class Orchestrator:
    def __init__(
        self,
        config: CampaignConfig,
        corpus: Corpus,
        llm: LLMClient,
        semantic: SemanticMemory | None = None,
    ) -> None:
        self.config = config
        self.corpus = corpus
        self._semantic = semantic
        self._roles: dict[str, BaseRole] = {
            "director": DirectorRole(llm),
            "scout": ScoutRole(llm),
            "analyst": AnalystRole(llm),
            "theorist": TheoristRole(llm),
            "prover": ProverRole(llm),
            "critic": CriticRole(llm),
            "experimenter": ExperimenterRole(llm),
        }

    def run(self) -> None:
        """Run the tick loop until a budget is exhausted or status != 'running'."""
        if self._semantic is not None:
            self._semantic.sync()
        while True:
            state = self.corpus.load_state()
            if self._budget_exhausted(state):
                break
            self._tick()
            if self._semantic is not None:
                self._semantic.sync()
            if self.config.tick_sleep_seconds > 0:
                time.sleep(self.config.tick_sleep_seconds)

    def _budget_exhausted(self, state: CampaignState) -> bool:
        if state.status != "running":
            return True
        if state.tick >= self.config.tick_budget:
            state.status = "exhausted"
            self.corpus.save_state(state)
            return True
        if time.time() - state.wall_start >= self.config.wall_clock_seconds:
            state.status = "exhausted"
            self.corpus.save_state(state)
            return True
        if state.tokens_used >= self.config.token_budget:
            state.status = "exhausted"
            self.corpus.save_state(state)
            return True
        return False

    def _tick(self) -> None:
        state = self.corpus.load_state()
        tick = state.tick

        cycle_pos = tick % len(ROLE_CYCLE)
        phase = _PHASE_MAP[cycle_pos]
        role_name = ROLE_CYCLE[cycle_pos]
        tokens = 0
        new_focus = state.focus

        try:
            # Director sets focus for this tick
            dir_ctx = RoleContext(
                tick=tick,
                focus=state.focus,
                corpus=self.corpus,
                phase=phase,
                semantic=self._semantic,
            )
            dir_out = self._roles["director"].run(dir_ctx)
            new_focus = str(dir_out.data.get("new_focus", state.focus))
            tokens += dir_out.tokens_used

            # Worker role (round-robin through the 6 research roles)
            ctx = RoleContext(
                tick=tick,
                focus=new_focus,
                corpus=self.corpus,
                phase=phase,
                semantic=self._semantic,
            )
            out = self._roles[role_name].run(ctx)
            tokens += out.tokens_used

            self.corpus.append_journal(
                {
                    "tick": tick,
                    "ts": time.time(),
                    "role": role_name,
                    "focus": new_focus,
                    "summary": out.summary,
                    "tokens": tokens,
                }
            )
        except Exception as exc:
            self.corpus.append_journal(
                {
                    "tick": tick,
                    "ts": time.time(),
                    "role": f"error:{role_name}",
                    "focus": new_focus,
                    "summary": f"tick error: {exc!r}",
                    "tokens": 0,
                }
            )

        # Always advance tick — campaign never stalls on a single role failure
        state = self.corpus.load_state()
        state.tick = tick + 1
        state.focus = new_focus
        state.last_role = role_name
        state.tokens_used += tokens
        self.corpus.save_state(state)
