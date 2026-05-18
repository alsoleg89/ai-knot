from __future__ import annotations

import re
import subprocess
import sys
import time
import uuid
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput

_CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

_SYSTEM_GENERATE = (
    "You are Experimenter, an applied scientist for long-dialogue retrieval systems. "
    "Implement the core of the current leading theory as a self-contained Python prototype. "
    "The prototype MUST use hard synthetic data: (1) ≥5000 distractor utterances, "
    "(2) one-off facts with Jaccard overlap < 0.15 between question and gold utterance, "
    "(3) if baseline BM25-surrogate recall@60 ≥ 0.70 on the generated corpus, print "
    "'CORPUS TOO EASY — invalid' and exit — do not report results on easy corpora. "
    "(4) implement the proposed inference-time memory layer, "
    "(5) print: baseline recall@60, memory-layer recall@60, union recall@60, "
    "delta pp, and 3 examples where memory finds what baseline misses. "
    "Write the complete runnable code inside a ```python block. "
    "After the code, write one line: "
    "TYPE: free | HYPOTHESIS: <one sentence> | EXPECTED_OUTCOME: <one sentence>. "
    "If this experiment requires expensive benchmark infrastructure (external APIs, "
    "large GPU compute), skip the code block and write REQUIRES_APPROVAL at the end."
)

_SYSTEM_BENCH = (
    "You are Experimenter, an applied scientist for long-dialogue retrieval systems. "
    "Survey the landscape of existing long-term memory and retrieval benchmarks "
    "(e.g. LoCoMo, MemGPT eval, HippoRAG bench, MIRIX, personal-dialogue QA). "
    "Identify the 2 most critical deficiencies in these benchmarks for evaluating "
    "single-fact recall under lexical mismatch. Then propose a concrete better benchmark design: "
    "NAME | TASK_TYPE | METRIC | WHY_BETTER. "
    "If a live benchmark run is needed to validate the proposal, end with REQUIRES_APPROVAL."
)


def _extract_code(content: str) -> str:
    m = _CODE_RE.search(content)
    return m.group(1).strip() if m else ""


def _run_code(path: str, timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[:1000],
            "stderr": proc.stderr[:500],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "timeout"}
    except OSError as exc:
        return {"exit_code": -1, "stdout": "", "stderr": str(exc)}


class ExperimenterRole(BaseRole):
    name = "experimenter"

    def run(self, ctx: RoleContext) -> RoleOutput:
        theory_so_far = ctx.corpus.read_theory()
        fitness_index = ctx.corpus.read_fitness_index()
        top_fitness = float(fitness_index[0].get("fitness", 0.0)) if fitness_index else 0.0

        if ctx.phase == "critique":
            system = _SYSTEM_BENCH
            user = (
                f"{ctx.brief_block(max_chars=1600)}"
                f"Research focus: {ctx.focus!r}. "
                f"Leading theory fitness={top_fitness:.2f}.\n"
                f"Current theory excerpt:\n{theory_so_far[:600]}\n\n"
                "Survey, critique, and propose a better benchmark."
            )
        else:
            recall_block = ctx.recall_block(
                ctx.focus, k=3, stream="experiments", header="Related past experiments:"
            )
            system = _SYSTEM_GENERATE
            user = (
                f"{ctx.brief_block(max_chars=1600)}"
                f"Research focus: {ctx.focus!r}. "
                f"Leading theory fitness={top_fitness:.2f}.\n"
                f"{recall_block}"
                f"Current theory:\n{theory_so_far[:1000]}\n\n"
                "Implement a Python prototype for this theory."
            )

        resp = self.llm.chat(system, user)
        exp_id = str(uuid.uuid4())[:8]
        requires_approval = "REQUIRES_APPROVAL" in resp.content

        run_result: dict[str, Any] = {}
        proto_path: str | None = None
        novelty = 0.5

        if not requires_approval:
            code = _extract_code(resp.content)
            if code:
                written = ctx.corpus.write_prototype(exp_id, code)
                proto_path = str(written)
                run_result = _run_code(proto_path)
                novelty = 0.8 if run_result["exit_code"] == 0 else 0.3

        entry: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "exp_id": exp_id,
            "focus": ctx.focus,
            "phase": ctx.phase,
            "requires_approval": requires_approval,
            "proto_path": proto_path,
            "run_result": run_result,
            "content": resp.content,
        }
        ctx.corpus.append_experiment(entry)

        if requires_approval:
            ctx.corpus.add_to_approval_queue(exp_id=exp_id, description=resp.content)

        # Emit novelty fitness signal
        candidates = ctx.corpus.read_theory_candidates(last_n=1)
        candidate_id = candidates[0]["candidate_id"] if candidates else "unknown"
        provability = float(fitness_index[0].get("provability", 0.5)) if fitness_index else 0.5
        applicability = float(fitness_index[0].get("applicability", 0.5)) if fitness_index else 0.5
        fitness_record: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "candidate_id": candidate_id,
            "provability": provability,
            "applicability": applicability,
            "novelty": novelty,
            "fitness": provability * 0.6 + applicability * 0.2 + novelty * 0.2,
        }
        ctx.corpus.append_fitness_record(fitness_record)

        summary = (
            f"Experimenter exp={exp_id} approval={requires_approval} "
            f"proto={'yes' if proto_path else 'no'} "
            f"exit={run_result.get('exit_code', 'n/a')}"
        )
        return RoleOutput(
            role_name=self.name,
            summary=summary,
            tokens_used=resp.total_tokens,
            data={
                "exp_id": exp_id,
                "requires_approval": requires_approval,
                "proto_path": proto_path,
                "run_result": run_result,
                "novelty": novelty,
            },
        )
