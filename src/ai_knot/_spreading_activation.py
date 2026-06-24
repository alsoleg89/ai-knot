"""DDSA stub for pf3-runtime reproduction.

Real DDSA was an experimental graph-walk recall extension. At pf3-runtime
(2026-04-13) ``AIKNOT_DDSA_ENABLED`` defaulted to ``"false"`` and the gate at
``knowledge.py::_execute_recall`` skipped the call entirely, so the function
body was never exercised. This stub keeps the same import/signature contract
without restoring the experimental code.
"""

from __future__ import annotations

import os
from typing import Any

DDSA_ENABLED: bool = os.environ.get("AIKNOT_DDSA_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)


def spreading_activation(
    seeds: list[tuple[Any, float]],
    index: Any | None = None,
    *,
    topk: int | None = None,
    decay: float = 0.6,
    temporal_window_sec: int = 60,
    activation_budget: int = 0,
) -> list[tuple[Any, float]]:
    """No-op pass-through. Returns the seed pairs unchanged.

    With ``DDSA_ENABLED=False`` (default) the caller's gate prevents this
    from running; the body exists only to satisfy the call signature when a
    user opts in via env. Re-enabling DDSA requires the original
    implementation, which is out of scope for the pf3 baseline restore.
    """
    if topk is not None:
        return list(seeds)[:topk]
    return list(seeds)
