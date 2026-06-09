from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignConfig:
    """Immutable campaign configuration. Hash uniquely identifies a run config."""

    brief_text: str
    model: str = "claude-opus-4-7"
    tick_budget: int = 10_000
    wall_clock_seconds: int = 259_200  # 72 hours
    token_budget: int = 50_000_000
    tick_sleep_seconds: float = 0.0  # inter-tick pause for API rate limiting
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    def hash(self) -> str:
        payload = json.dumps(
            {
                "brief_text": self.brief_text,
                "model": self.model,
                "tick_budget": self.tick_budget,
                "wall_clock_seconds": self.wall_clock_seconds,
                "token_budget": self.token_budget,
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
