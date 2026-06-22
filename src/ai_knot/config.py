"""Typed, validated configuration for ai-knot, loaded from environment variables.

A single source of truth for the env vars that configure a ``KnowledgeBase`` /
MCP server, with validation at load time so a misconfiguration fails fast with a
clear message instead of surfacing as an obscure error on the first recall.

Module-level experiment/debug flags (``AI_KNOT_RERANK*``, ``AI_KNOT_*_DEBUG``,
``AIKNOT_DDSA_ENABLED``, ``AI_KNOT_DENSE_WEIGHT_MULT``) are read at import time in
their own modules and are intentionally NOT centralized here — they are niche and
moving them would change evaluation timing for little operational gain.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

_TRUE = ("1", "true", "yes")
_BACKENDS = ("yaml", "sqlite", "postgres")


def _as_bool(value: str | None) -> bool:
    """Interpret an env string as a boolean (``1``/``true``/``yes`` → True)."""
    return (value or "").lower() in _TRUE


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated string of floats into a tuple."""
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class StorageConfig:
    """Storage backend selection and location."""

    backend: Literal["yaml", "sqlite", "postgres"]
    data_dir: str
    db_path: str | None
    dsn: str | None


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider used for ``learn()`` extraction and optional recall expansion."""

    provider: str | None
    api_key: str | None
    model: str | None
    llm_recall: bool


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding endpoint for the dense retrieval signal."""

    url: str
    model: str
    api_key: str | None


@dataclass(frozen=True)
class RecallConfig:
    """Tunables for the recall pipeline."""

    rrf_weights: tuple[float, ...] | None
    expansion_weight: float | None
    episodic_ttl_hours: float


@dataclass(frozen=True)
class AIKnotConfig:
    """The full validated configuration for a single KnowledgeBase / server."""

    agent_id: str
    storage: StorageConfig
    llm: LLMConfig
    embed: EmbedConfig
    recall: RecallConfig

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AIKnotConfig:
        """Build and validate a configuration from ``env`` (defaults to ``os.environ``).

        Raises:
            ValueError: If a value is invalid (unknown backend, missing postgres
                DSN, non-positive TTL, negative RRF weight, unparseable number).
        """
        e: Mapping[str, str] = os.environ if env is None else env

        backend = e.get("AI_KNOT_STORAGE", "sqlite")
        if backend not in _BACKENDS:
            raise ValueError(
                f"AI_KNOT_STORAGE must be one of {', '.join(_BACKENDS)}, got {backend!r}"
            )
        data_dir = e.get("AI_KNOT_DATA_DIR", ".ai_knot")
        db_path = e.get("AI_KNOT_DB_PATH")
        if backend == "sqlite":
            dsn: str | None = db_path or os.path.join(data_dir, "ai_knot.db")
        elif backend == "postgres":
            dsn = e.get("AI_KNOT_DSN")
            if not dsn:
                raise ValueError(
                    "postgres backend requires a DSN — set AI_KNOT_DSN to the connection string"
                )
        else:
            dsn = None

        raw_rrf = e.get("AI_KNOT_RRF_WEIGHTS")
        try:
            rrf_weights = _parse_float_tuple(raw_rrf) if raw_rrf else None
        except ValueError as exc:
            raise ValueError(f"AI_KNOT_RRF_WEIGHTS must be comma-separated floats: {exc}") from None
        if rrf_weights is not None and any(w < 0 for w in rrf_weights):
            raise ValueError("AI_KNOT_RRF_WEIGHTS must be non-negative")

        raw_exp = e.get("AI_KNOT_EXPANSION_WEIGHT")
        try:
            expansion_weight = float(raw_exp) if raw_exp else None
        except ValueError:
            raise ValueError(f"AI_KNOT_EXPANSION_WEIGHT must be a float, got {raw_exp!r}") from None

        raw_ttl = e.get("AI_KNOT_EPISODIC_TTL")
        try:
            episodic_ttl_hours = float(raw_ttl) if raw_ttl else 72.0
        except ValueError:
            raise ValueError(f"AI_KNOT_EPISODIC_TTL must be a float, got {raw_ttl!r}") from None
        if episodic_ttl_hours <= 0:
            raise ValueError(f"AI_KNOT_EPISODIC_TTL must be > 0, got {episodic_ttl_hours}")

        return cls(
            agent_id=e.get("AI_KNOT_AGENT_ID", "default"),
            storage=StorageConfig(
                backend=cast(Literal["yaml", "sqlite", "postgres"], backend),
                data_dir=data_dir,
                db_path=db_path,
                dsn=dsn,
            ),
            llm=LLMConfig(
                provider=e.get("AI_KNOT_PROVIDER"),
                api_key=e.get("AI_KNOT_API_KEY"),
                model=e.get("AI_KNOT_MODEL"),
                llm_recall=_as_bool(e.get("AI_KNOT_LLM_RECALL")),
            ),
            embed=EmbedConfig(
                url=e.get("AI_KNOT_EMBED_URL", "http://localhost:11434"),
                model=e.get("AI_KNOT_EMBED_MODEL", "nomic-embed-text"),
                api_key=e.get("AI_KNOT_EMBED_API_KEY") or e.get("OPENAI_API_KEY"),
            ),
            recall=RecallConfig(
                rrf_weights=rrf_weights,
                expansion_weight=expansion_weight,
                episodic_ttl_hours=episodic_ttl_hours,
            ),
        )
