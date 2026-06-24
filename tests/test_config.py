"""Tests for ai_knot.config.AIKnotConfig.from_env — typed, validated config."""

from __future__ import annotations

import pytest

from ai_knot.config import AIKnotConfig


def test_defaults() -> None:
    cfg = AIKnotConfig.from_env({})
    assert cfg.agent_id == "default"
    assert cfg.storage.backend == "sqlite"
    assert cfg.storage.data_dir == ".ai_knot"
    assert cfg.storage.dsn == ".ai_knot/ai_knot.db"  # sqlite path derived from data_dir
    assert cfg.llm.provider is None
    assert cfg.llm.llm_recall is False
    assert cfg.embed.url == "http://localhost:11434"
    assert cfg.embed.model == "nomic-embed-text"
    assert cfg.recall.rrf_weights is None
    assert cfg.recall.episodic_ttl_hours == 72.0


def test_full_env_is_parsed() -> None:
    cfg = AIKnotConfig.from_env(
        {
            "AI_KNOT_AGENT_ID": "agent-1",
            "AI_KNOT_STORAGE": "yaml",
            "AI_KNOT_DATA_DIR": "/data",
            "AI_KNOT_LLM_RECALL": "true",
            "AI_KNOT_PROVIDER": "openai",
            "AI_KNOT_API_KEY": "sk-x",
            "AI_KNOT_MODEL": "gpt-4.1-nano",
            "AI_KNOT_RRF_WEIGHTS": "1.0, 2.0, 3.0",
            "AI_KNOT_EXPANSION_WEIGHT": "0.5",
            "AI_KNOT_EPISODIC_TTL": "24",
            "AI_KNOT_EMBED_URL": "https://api.openai.com",
            "AI_KNOT_EMBED_MODEL": "text-embedding-3-small",
            "AI_KNOT_EMBED_API_KEY": "sk-embed",
        }
    )
    assert cfg.agent_id == "agent-1"
    assert cfg.storage.backend == "yaml"
    assert cfg.storage.dsn is None  # yaml has no dsn
    assert cfg.llm.llm_recall is True
    assert cfg.llm.provider == "openai"
    assert cfg.recall.rrf_weights == (1.0, 2.0, 3.0)
    assert cfg.recall.expansion_weight == 0.5
    assert cfg.recall.episodic_ttl_hours == 24.0
    assert cfg.embed.api_key == "sk-embed"


def test_db_path_overrides_sqlite_dsn() -> None:
    cfg = AIKnotConfig.from_env({"AI_KNOT_STORAGE": "sqlite", "AI_KNOT_DB_PATH": "/tmp/x.db"})
    assert cfg.storage.dsn == "/tmp/x.db"


def test_embed_api_key_falls_back_to_openai_key() -> None:
    cfg = AIKnotConfig.from_env({"OPENAI_API_KEY": "sk-openai"})
    assert cfg.embed.api_key == "sk-openai"
    # explicit embed key wins over the OpenAI fallback
    cfg2 = AIKnotConfig.from_env({"OPENAI_API_KEY": "sk-openai", "AI_KNOT_EMBED_API_KEY": "sk-e"})
    assert cfg2.embed.api_key == "sk-e"


def test_postgres_requires_dsn() -> None:
    with pytest.raises(ValueError, match="postgres backend requires a DSN"):
        AIKnotConfig.from_env({"AI_KNOT_STORAGE": "postgres"})


def test_postgres_dsn_from_env() -> None:
    cfg = AIKnotConfig.from_env(
        {"AI_KNOT_STORAGE": "postgres", "AI_KNOT_DSN": "postgresql://localhost/x"}
    )
    assert cfg.storage.dsn == "postgresql://localhost/x"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="AI_KNOT_STORAGE must be one of"):
        AIKnotConfig.from_env({"AI_KNOT_STORAGE": "redis"})


def test_non_positive_ttl_rejected() -> None:
    with pytest.raises(ValueError, match="AI_KNOT_EPISODIC_TTL must be > 0"):
        AIKnotConfig.from_env({"AI_KNOT_EPISODIC_TTL": "0"})


def test_bad_ttl_value_rejected() -> None:
    with pytest.raises(ValueError, match="AI_KNOT_EPISODIC_TTL must be a float"):
        AIKnotConfig.from_env({"AI_KNOT_EPISODIC_TTL": "soon"})


def test_negative_rrf_weight_rejected() -> None:
    with pytest.raises(ValueError, match="AI_KNOT_RRF_WEIGHTS must be non-negative"):
        AIKnotConfig.from_env({"AI_KNOT_RRF_WEIGHTS": "1.0,-2.0"})


def test_bad_rrf_weight_rejected() -> None:
    with pytest.raises(ValueError, match="AI_KNOT_RRF_WEIGHTS must be comma-separated floats"):
        AIKnotConfig.from_env({"AI_KNOT_RRF_WEIGHTS": "1.0,abc"})


def test_config_is_frozen() -> None:
    cfg = AIKnotConfig.from_env({})
    with pytest.raises(AttributeError):
        cfg.agent_id = "x"  # type: ignore[misc]
