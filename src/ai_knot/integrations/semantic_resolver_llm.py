"""Reference ``SemanticConflictResolver`` backed by a caller-supplied LLM.

ai-knot core ships no model and imports none.  The deterministic
:class:`~ai_knot.multi_agent.canonical.ClaimFamilyResolver` resolves slotted and
lexically-near conflicts; value conflicts whose rivals share a *subject* but
diverge lexically — "the REST endpoint supports both protocols" vs "the REST
endpoint is deprecated" — fall below the IDF-overlap floor and need semantic
judgement (their shared subject tokens are low-IDF, their conflicting values are
high-IDF, so no deterministic clustering can group them without a value lexicon).

This adapter implements the opt-in semantic-resolver seam
(``SharedMemoryPool(semantic_resolver=...)``) **without adding any dependency**:
it is parameterized by a ``complete(prompt: str) -> str`` callable, so wiring an
actual model (OpenAI, Claude, a local server) is the integrator's choice and the
core package never imports an LLM.  The default retrieval path stays
deterministic and dependency-free.

Usage::

    from openai import OpenAI  # the integrator's own dependency, not ai-knot's

    client = OpenAI()

    def complete(prompt: str) -> str:
        out = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return out.choices[0].message.content or ""

    pool = SharedMemoryPool(
        storage=storage,
        semantic_resolver=LLMSemanticConflictResolver(complete),
    )
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ai_knot.types import Fact

_PROMPT_HEADER = (
    "You are resolving competing memory claims from different agents. The numbered "
    "claims below all concern a related subject. Identify which claims are STALE — "
    "superseded or contradicted by a more current claim about the SAME subject "
    "(for example a later 'deprecated', 'tightened', or 'expanded' statement "
    "supersedes the earlier state it replaces). Do NOT mark complementary claims "
    "(a different aspect of the same subject) as stale.\n\n"
    "Reply with the numbers of the stale claims only, comma-separated, or 'none'.\n"
)


class LLMSemanticConflictResolver:
    """Opt-in semantic resolver that defers the supersede/keep call to an LLM.

    Conforms to the ``SemanticConflictResolver`` protocol: ``__call__`` takes the
    candidate facts the deterministic resolver left standing and returns the ids
    of the ones judged superseded (stale); the pool drops them before the final
    cut.
    """

    def __init__(self, complete: Callable[[str], str]) -> None:
        self._complete = complete

    def __call__(self, candidates: list[Fact]) -> set[str]:
        if len(candidates) < 2:
            return set()
        prompt = self._build_prompt(candidates)
        try:
            response = self._complete(prompt)
        except Exception:
            # A resolver failure must never break recall: keep everything, which
            # is exactly the deterministic-only behaviour.
            return set()
        return self._parse(response, candidates)

    @staticmethod
    def _build_prompt(candidates: list[Fact]) -> str:
        lines = [_PROMPT_HEADER]
        lines.extend(f"{i}. {f.content}" for i, f in enumerate(candidates, start=1))
        return "\n".join(lines)

    @staticmethod
    def _parse(response: str, candidates: list[Fact]) -> set[str]:
        lowered = response.strip().lower()
        if not lowered or lowered.startswith("none"):
            return set()
        superseded: set[str] = set()
        for num in re.findall(r"\d+", response):
            idx = int(num) - 1
            if 0 <= idx < len(candidates):
                superseded.add(candidates[idx].id)
        # Never supersede the entire cluster — that would erase the subject; treat
        # an all-stale verdict as no-confidence and keep everything.
        if len(superseded) >= len(candidates):
            return set()
        return superseded
