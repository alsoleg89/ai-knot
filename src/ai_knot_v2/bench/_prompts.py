"""Ported prompts from aiknotbench/src/evaluator.ts.

SHAs verified against aiknotbench/config/canonical.json to ensure parity with v1.
bench/ only — never imported from core/ ops/ store/ api/.
"""

from __future__ import annotations

import hashlib

ANSWER_SYSTEM: str = "Answer the question based on the memory context below. Answer concisely."

JUDGE_SYSTEM: str = (
    "You are an evaluation judge."
    " Given a question, a candidate answer, and the gold answer,\n"
    "decide whether the candidate answer is correct.\n"
    "\n"
    'Return JSON exactly like: {"verdict": "CORRECT"} or {"verdict": "WRONG"}\n'
    "\n"
    "Rules:\n"
    "- CORRECT if the candidate answer conveys the same essential information as the gold answer.\n"
    "- Exact wording is not required; semantic equivalence is sufficient.\n"
    "- WRONG if the candidate answer is missing key facts,"
    " contradicts the gold, or is irrelevant.\n"
    "- Do not output anything other than the JSON object."
)

# SHAs from aiknotbench/config/canonical.json — must match for v1/v2 parity
_EXPECTED_ANSWER_SHA: str = "54102d3fcf457ff97bf30e6b1e074b1bb8689c48881fcf1ccc761f46a2660f95"
_EXPECTED_JUDGE_SHA: str = "2e225095dea82d2e2aea37ee75420ac6cb4fef9729e31ef8ecb6bff5406d8ae9"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_prompt_parity() -> tuple[bool, str]:
    """Return (ok, message). True iff prompts match canonical SHAs."""
    a_sha = _sha(ANSWER_SYSTEM)
    j_sha = _sha(JUDGE_SYSTEM)
    if a_sha != _EXPECTED_ANSWER_SHA:
        return False, f"ANSWER_SYSTEM SHA mismatch: got {a_sha}, want {_EXPECTED_ANSWER_SHA}"
    if j_sha != _EXPECTED_JUDGE_SHA:
        return False, f"JUDGE_SYSTEM SHA mismatch: got {j_sha}, want {_EXPECTED_JUDGE_SHA}"
    return True, "ok"
