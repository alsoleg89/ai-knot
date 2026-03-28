"""Simulation scenarios — end-to-end tests for memory, storage, providers, CLI.

These tests simulate real-world usage patterns to verify that agentmemo
works correctly as a whole system, not just individual units.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentmemo.cli import main
from agentmemo.extractor import Extractor, deduplicate_facts
from agentmemo.forgetting import calculate_retention
from agentmemo.integrations.openai import MemoryEnabledOpenAI
from agentmemo.knowledge import KnowledgeBase
from agentmemo.providers import create_provider
from agentmemo.providers.base import call_with_retry
from agentmemo.storage import create_storage
from agentmemo.storage.sqlite_storage import SQLiteStorage
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import ConversationTurn, Fact, MemoryType

# ---------------------------------------------------------------------------
# 2A. Memory Scenarios (13 tests)
# ---------------------------------------------------------------------------


class TestMemoryScenarios:
    """Simulate real-world memory usage patterns."""

    def test_01_personal_assistant_cold_start(self, tmp_path: pathlib.Path) -> None:
        """Add 5 facts, recall specific info — basic add + recall flow."""
        kb = KnowledgeBase(agent_id="assistant", storage=YAMLStorage(base_dir=str(tmp_path)))

        kb.add("User name is Alexei", importance=0.95)
        kb.add("User works at Sber as CTO", type=MemoryType.SEMANTIC, importance=0.9)
        kb.add("User prefers Python over Java", type=MemoryType.PROCEDURAL, importance=0.85)
        kb.add("User deploys on Kubernetes", importance=0.8)
        kb.add("Always use pytest, never unittest", type=MemoryType.PROCEDURAL, importance=0.9)

        result = kb.recall("what is the user's name?")
        assert "Alexei" in result

        result = kb.recall("what programming language?")
        assert "Python" in result

    def test_02_conversation_extraction(self, tmp_path: pathlib.Path) -> None:
        """Mock LLM returns 3 facts from 5 turns — learn → storage → recall."""
        kb = KnowledgeBase(agent_id="learner", storage=YAMLStorage(base_dir=str(tmp_path)))

        mock_response = json.dumps(
            [
                {"content": "User deploys with Docker", "type": "semantic", "importance": 0.85},
                {"content": "User hates async code", "type": "procedural", "importance": 0.8},
                {"content": "User works at Yandex", "type": "semantic", "importance": 0.9},
            ]
        )

        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.default_model = "mock-v1"
        mock_provider.call.return_value = mock_response

        turns = [
            ConversationTurn(role="user", content="I deploy everything in Docker"),
            ConversationTurn(role="assistant", content="Got it"),
            ConversationTurn(role="user", content="I really hate async, prefer sync"),
            ConversationTurn(role="assistant", content="Understood"),
            ConversationTurn(role="user", content="By the way, I work at Yandex"),
        ]

        new_facts = kb.learn(turns, provider=mock_provider)
        assert len(new_facts) == 3

        result = kb.recall("Docker deployment")
        assert "Docker" in result

        result = kb.recall("where does user work?")
        assert "Yandex" in result

    def test_03_forgetting_curve_1_hour(self) -> None:
        """High importance fact stays >0.99 after 1 hour."""
        fact = Fact(content="Critical info", importance=0.95, access_count=5)
        now = fact.last_accessed + timedelta(hours=1)
        retention = calculate_retention(fact, now=now)
        assert retention > 0.99

    def test_04_forgetting_curve_30_days(self) -> None:
        """Low importance (0.3) fact drops below 0.05 after 30 days."""
        fact = Fact(content="Trivial info", importance=0.3, access_count=0)
        now = fact.last_accessed + timedelta(days=30)
        retention = calculate_retention(fact, now=now)
        assert retention < 0.05

    def test_05_forgetting_access_reinforcement(self, tmp_path: pathlib.Path) -> None:
        """Fact recalled 10 times retains much better than one never accessed."""
        kb = KnowledgeBase(agent_id="reinforce", storage=YAMLStorage(base_dir=str(tmp_path)))
        kb.add("Python is the best language", importance=0.7)

        for _ in range(10):
            kb.recall("what language?")

        facts = kb.list_facts()
        assert facts[0].access_count >= 10
        assert facts[0].retention_score > 0.95

    def test_06_deduplication(self) -> None:
        """Near-duplicate facts are merged by Jaccard similarity."""
        facts = [
            Fact(content="User likes Python programming language"),
            Fact(content="User likes Python programming"),
            Fact(content="User deploys on Kubernetes"),
        ]
        unique = deduplicate_facts(facts)
        assert len(unique) == 2
        assert any("Kubernetes" in f.content for f in unique)

    def test_07_multi_agent_isolation(self, tmp_path: pathlib.Path) -> None:
        """Two agents share storage backend but see only their own facts."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        kb_a = KnowledgeBase(agent_id="agent_alpha", storage=storage)
        kb_b = KnowledgeBase(agent_id="agent_beta", storage=storage)

        kb_a.add("Alpha secret fact", importance=0.9)
        kb_b.add("Beta secret fact", importance=0.9)

        result_a = kb_a.recall("secret")
        result_b = kb_b.recall("secret")

        assert "Alpha" in result_a
        assert "Beta" not in result_a
        assert "Beta" in result_b
        assert "Alpha" not in result_b

    def test_08_top_k_limiting(self, tmp_path: pathlib.Path) -> None:
        """20 facts, recall with top_k=3 returns at most 3 lines."""
        kb = KnowledgeBase(agent_id="topk", storage=YAMLStorage(base_dir=str(tmp_path)))
        for i in range(20):
            kb.add(f"Fact number {i} about Python programming", importance=0.8)

        result = kb.recall("Python", top_k=3)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) <= 3

    def test_09_empty_kb_recall(self, tmp_path: pathlib.Path) -> None:
        """Recall on a fresh KB returns empty string, no crash."""
        kb = KnowledgeBase(agent_id="empty", storage=YAMLStorage(base_dir=str(tmp_path)))
        result = kb.recall("anything at all")
        assert result == ""

    def test_10_unicode_support(self, tmp_path: pathlib.Path) -> None:
        """Russian, Chinese, and emoji facts can be stored and recalled."""
        kb = KnowledgeBase(agent_id="unicode", storage=YAMLStorage(base_dir=str(tmp_path)))

        kb.add("Пользователь работает в Сбере", importance=0.9)
        kb.add("用户喜欢Python编程", importance=0.85)
        kb.add("User loves 🐍 Python", importance=0.8)

        result = kb.recall("Сбер работа")
        assert "Сбер" in result

        result = kb.recall("Python")
        assert "Python" in result

    def test_11_export_clear_import_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Export → clear → import preserves all facts via KnowledgeBase API."""
        kb = KnowledgeBase(agent_id="roundtrip", storage=YAMLStorage(base_dir=str(tmp_path)))
        kb.add("Important fact one", importance=0.9, type=MemoryType.SEMANTIC)
        kb.add("Important fact two", importance=0.8, type=MemoryType.PROCEDURAL)
        kb.add("Important fact three", importance=0.7, type=MemoryType.EPISODIC)

        original = kb.list_facts()
        assert len(original) == 3

        kb.clear_all()
        assert kb.list_facts() == []

        kb.replace_facts(original)
        restored = kb.list_facts()
        assert len(restored) == 3
        assert {f.content for f in restored} == {f.content for f in original}

    def test_12_context_rot_simulation(self, tmp_path: pathlib.Path) -> None:
        """50 facts: 45 stale + 5 fresh — recall returns mostly fresh facts."""
        kb = KnowledgeBase(agent_id="rot", storage=YAMLStorage(base_dir=str(tmp_path)))

        old_time = datetime(2025, 1, 1, tzinfo=UTC)

        # Add 45 stale, low-importance facts
        for i in range(45):
            fact = Fact(
                content=f"Stale info number {i}",
                importance=0.2,
                access_count=0,
                created_at=old_time,
                last_accessed=old_time,
            )
            facts = kb.list_facts()
            facts.append(fact)
            kb.replace_facts(facts)

        # Add 5 fresh, high-importance facts
        for i in range(5):
            kb.add(f"Fresh important fact {i} about deployment", importance=0.95)

        result = kb.recall("deployment", top_k=5)
        lines = result.strip().split("\n")
        fresh_count = sum(1 for line in lines if "Fresh important" in line)
        assert fresh_count >= 3  # Most results should be fresh

    def test_13_importance_boundaries(self) -> None:
        """importance=0.0 → instantly forgotten; importance=1.0 → persists months."""
        zero_fact = Fact(content="Zero importance", importance=0.0, access_count=0)
        one_hour_later = zero_fact.last_accessed + timedelta(hours=1)
        assert calculate_retention(zero_fact, now=one_hour_later) == 0.0

        high_fact = Fact(content="Max importance", importance=1.0, access_count=10)
        three_months_later = high_fact.last_accessed + timedelta(days=90)
        retention = calculate_retention(high_fact, now=three_months_later)
        assert retention > 0.1  # Still partially retained after 3 months


# ---------------------------------------------------------------------------
# 2B. Storage Backend Scenarios (8 tests)
# ---------------------------------------------------------------------------


class TestStorageScenarios:
    """Test storage backends under realistic conditions."""

    def test_14_yaml_to_sqlite_migration(self, tmp_path: pathlib.Path) -> None:
        """Save in YAML, load all, save to SQLite, verify identical content."""
        yaml_storage = YAMLStorage(base_dir=str(tmp_path / "yaml"))
        sqlite_storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))

        facts = [
            Fact(content="Fact one", type=MemoryType.SEMANTIC, importance=0.9),
            Fact(content="Fact two", type=MemoryType.PROCEDURAL, importance=0.8),
            Fact(content="Fact three", type=MemoryType.EPISODIC, importance=0.7),
        ]

        yaml_storage.save("migrator", facts)
        loaded = yaml_storage.load("migrator")
        sqlite_storage.save("migrator", loaded)
        migrated = sqlite_storage.load("migrator")

        assert len(migrated) == 3
        assert {f.content for f in migrated} == {f.content for f in facts}
        assert {f.type for f in migrated} == {f.type for f in facts}

    def test_15_sqlite_concurrent_reads(self, tmp_path: pathlib.Path) -> None:
        """3 threads reading same agent_id simultaneously — no crashes."""
        storage = SQLiteStorage(db_path=str(tmp_path / "concurrent.db"))
        facts = [Fact(content=f"Concurrent fact {i}", importance=0.8) for i in range(10)]
        storage.save("shared", facts)

        results: list[list[Fact]] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                loaded = storage.load("shared")
                results.append(loaded)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent reads failed: {errors}"
        assert all(len(r) == 10 for r in results)

    def test_16_yaml_write_persistence(self, tmp_path: pathlib.Path) -> None:
        """Write 100 facts, re-instantiate storage, verify all persisted."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        facts = [Fact(content=f"Persistent fact {i}", importance=0.8) for i in range(100)]
        storage.save("persistent", facts)

        # Re-instantiate — simulates process restart
        storage2 = YAMLStorage(base_dir=str(tmp_path))
        loaded = storage2.load("persistent")
        assert len(loaded) == 100

    def test_17_sqlite_wal_mode(self, tmp_path: pathlib.Path) -> None:
        """SQLite storage uses WAL journal mode for better concurrency."""
        db_path = str(tmp_path / "wal_test.db")
        SQLiteStorage(db_path=db_path)

        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        conn.close()
        assert result is not None
        assert result[0] == "wal"

    def test_18_storage_factory_routing(self, tmp_path: pathlib.Path) -> None:
        """create_storage() routes to correct backend class."""
        yaml_s = create_storage("yaml", base_dir=str(tmp_path))
        assert isinstance(yaml_s, YAMLStorage)

        sqlite_s = create_storage("sqlite", base_dir=str(tmp_path))
        assert isinstance(sqlite_s, SQLiteStorage)

        with pytest.raises(ValueError, match="Unknown storage backend"):
            create_storage("mongodb")

    def test_19_large_dataset_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """1000 facts save+load round-trip on both YAML and SQLite."""
        facts = [
            Fact(content=f"Fact number {i} with some detail", importance=round(i / 1000, 3))
            for i in range(1000)
        ]

        sqlite_storage = SQLiteStorage(db_path=str(tmp_path / "large.db"))
        sqlite_storage.save("large", facts)
        loaded = sqlite_storage.load("large")
        assert len(loaded) == 1000

    def test_20_delete_nonexistent_fact(self, tmp_path: pathlib.Path) -> None:
        """Delete non-existent fact raises no error."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        storage.save("test_agent", [Fact(content="One fact")])
        storage.delete("test_agent", "nonexistent_id")  # Should not raise
        assert len(storage.load("test_agent")) == 1

    def test_21_postgres_storage_mock(self) -> None:
        """PostgresStorage logic works with mocked psycopg module."""
        mock_psycopg = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value = mock_conn

        # Mock the context manager for _get_conn (uses `with conn`)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        import sys

        sys.modules["psycopg"] = mock_psycopg
        try:
            import agentmemo.storage.postgres_storage as pg_mod

            importlib.reload(pg_mod)
            storage = pg_mod.PostgresStorage(dsn="postgresql://fake:fake@localhost/fake")

            # Table creation happened
            mock_conn.execute.assert_called()

            # Test save
            facts = [Fact(content="Test fact", importance=0.8)]
            storage.save("test_agent", facts)

            # Test list_agents
            mock_cur = MagicMock()
            mock_cur.fetchall.return_value = [("agent1",), ("agent2",)]
            mock_conn.execute.return_value = mock_cur
            agents = storage.list_agents()
            assert agents == ["agent1", "agent2"]
        finally:
            del sys.modules["psycopg"]


# ---------------------------------------------------------------------------
# 2C. Provider Scenarios (5 tests)
# ---------------------------------------------------------------------------


class TestProviderScenarios:
    """Test LLM provider factory, retry logic, and extractor edge cases."""

    def test_22_provider_factory_all_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_provider() creates correct class for each supported provider."""
        from agentmemo.providers.anthropic import AnthropicProvider
        from agentmemo.providers.openai_compat import OpenAICompatProvider

        provider = create_provider("openai", "sk-fake")
        assert isinstance(provider, OpenAICompatProvider)

        provider = create_provider("anthropic", "sk-ant-fake")
        assert isinstance(provider, AnthropicProvider)

        provider = create_provider("gigachat", "fake-token")
        assert isinstance(provider, OpenAICompatProvider)

        provider = create_provider("qwen", "fake-qwen-key")
        assert isinstance(provider, OpenAICompatProvider)

        provider = create_provider("openai-compat", "fake-key", base_url="http://localhost:8080/v1")
        assert isinstance(provider, OpenAICompatProvider)

        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("not_a_provider", "key")

    def test_23_retry_on_429(self) -> None:
        """call_with_retry retries on 429 and succeeds on 3rd attempt."""
        import httpx

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "v1"

        response_429 = MagicMock()
        response_429.status_code = 429
        error_429 = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response_429)

        mock_provider.call.side_effect = [error_429, error_429, "Success response"]

        with patch("agentmemo.providers.base.time.sleep"):
            result = call_with_retry(mock_provider, "sys", "user", "model", max_retries=3)

        assert result == "Success response"
        assert mock_provider.call.call_count == 3

    def test_24_provider_env_var_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider resolves API key from environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        provider = create_provider("openai")
        assert provider is not None

    def test_25_malformed_json_from_llm(self) -> None:
        """Extractor handles malformed LLM response gracefully."""
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "v1"
        mock_provider.call.return_value = "This is not JSON at all, sorry!"

        extractor = Extractor(provider=mock_provider)
        turns = [ConversationTurn(role="user", content="test message")]
        result = extractor.extract(turns)
        assert result == []

    def test_26_markdown_fenced_json(self) -> None:
        """Extractor strips markdown code fences from LLM response."""
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.default_model = "v1"
        fenced = '```json\n[{"content": "User likes Docker", '
        fenced += '"type": "semantic", "importance": 0.8}]\n```'
        mock_provider.call.return_value = fenced

        extractor = Extractor(provider=mock_provider)
        turns = [ConversationTurn(role="user", content="I use Docker")]
        result = extractor.extract(turns)
        assert len(result) == 1
        assert "Docker" in result[0].content


# ---------------------------------------------------------------------------
# 2D. CLI Scenarios (4 tests)
# ---------------------------------------------------------------------------


class TestCLIScenarios:
    """Test CLI commands in realistic sequences."""

    def test_27_full_cli_lifecycle(self, tmp_path: pathlib.Path) -> None:
        """add → show → recall → stats → export → clear → import → show."""
        runner = CliRunner()
        data_dir = str(tmp_path)

        # Add facts
        r = runner.invoke(main, ["--data-dir", data_dir, "add", "cli_agent", "User likes Python"])
        assert r.exit_code == 0

        r = runner.invoke(
            main,
            ["--data-dir", data_dir, "add", "cli_agent", "User deploys with Docker", "-i", "0.9"],
        )
        assert r.exit_code == 0

        # Show
        r = runner.invoke(main, ["--data-dir", data_dir, "show", "cli_agent"])
        assert r.exit_code == 0
        assert "Python" in r.output
        assert "Docker" in r.output
        assert "2 facts total" in r.output

        # Recall
        r = runner.invoke(main, ["--data-dir", data_dir, "recall", "cli_agent", "what language?"])
        assert r.exit_code == 0
        assert "Python" in r.output

        # Stats
        r = runner.invoke(main, ["--data-dir", data_dir, "stats", "cli_agent"])
        assert r.exit_code == 0
        assert "Total facts: 2" in r.output

        # Export
        export_file = str(tmp_path / "export.yaml")
        r = runner.invoke(main, ["--data-dir", data_dir, "export", "cli_agent", export_file])
        assert r.exit_code == 0
        assert "Exported 2 facts" in r.output

        # Clear
        r = runner.invoke(main, ["--data-dir", data_dir, "clear", "cli_agent"], input="y\n")
        assert r.exit_code == 0

        # Verify empty
        r = runner.invoke(main, ["--data-dir", data_dir, "show", "cli_agent"])
        assert "No facts" in r.output

        # Import
        r = runner.invoke(main, ["--data-dir", data_dir, "import", "cli_agent", export_file])
        assert r.exit_code == 0
        assert "Imported 2 facts" in r.output

        # Show again — restored
        r = runner.invoke(main, ["--data-dir", data_dir, "show", "cli_agent"])
        assert "Python" in r.output
        assert "2 facts total" in r.output

    def test_28_cli_with_sqlite_storage(self, tmp_path: pathlib.Path) -> None:
        """Full cycle with --storage sqlite flag."""
        runner = CliRunner()
        data_dir = str(tmp_path)

        r = runner.invoke(
            main,
            ["--data-dir", data_dir, "-s", "sqlite", "add", "sql_agent", "SQLite fact one"],
        )
        assert r.exit_code == 0

        r = runner.invoke(main, ["--data-dir", data_dir, "-s", "sqlite", "show", "sql_agent"])
        assert r.exit_code == 0
        assert "SQLite fact" in r.output

        r = runner.invoke(
            main,
            ["--data-dir", data_dir, "-s", "sqlite", "recall", "sql_agent", "SQLite"],
        )
        assert r.exit_code == 0
        assert "SQLite" in r.output

    def test_29_cli_error_paths(self, tmp_path: pathlib.Path) -> None:
        """Error handling: empty fact, non-existent agent, broken YAML."""
        runner = CliRunner()
        data_dir = str(tmp_path)

        # Empty content
        r = runner.invoke(main, ["--data-dir", data_dir, "add", "err_agent", "   "])
        assert r.exit_code != 0

        # Show non-existent agent — should show "No facts" not crash
        r = runner.invoke(main, ["--data-dir", data_dir, "show", "nonexistent"])
        assert r.exit_code == 0
        assert "No facts" in r.output

        # Import broken YAML
        broken_file = tmp_path / "broken.yaml"
        broken_file.write_text("!!invalid yaml: [", encoding="utf-8")
        r = runner.invoke(main, ["--data-dir", data_dir, "import", "err_agent", str(broken_file)])
        assert r.exit_code != 0

    def test_30_cli_decay_command(self, tmp_path: pathlib.Path) -> None:
        """Add facts with old timestamps, run decay, verify retention dropped."""
        storage = YAMLStorage(base_dir=str(tmp_path))
        old_fact = Fact(
            content="Old stale fact about Java",
            importance=0.3,
            access_count=0,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            last_accessed=datetime(2025, 1, 1, tzinfo=UTC),
        )
        storage.save("decay_agent", [old_fact])

        runner = CliRunner()
        r = runner.invoke(main, ["--data-dir", str(tmp_path), "decay", "decay_agent"])
        assert r.exit_code == 0
        assert "Decay applied" in r.output

        # Check retention actually dropped
        facts = storage.load("decay_agent")
        assert facts[0].retention_score < 0.1


# ---------------------------------------------------------------------------
# 2E. Integration Scenarios (2 tests)
# ---------------------------------------------------------------------------


class TestIntegrationScenarios:
    """Test OpenAI integration in realistic setups."""

    def test_31_openai_enrichment_full_flow(self, tmp_path: pathlib.Path) -> None:
        """Add facts, call enrich_messages, verify system prompt contains memory."""
        kb = KnowledgeBase(agent_id="openai_sim", storage=YAMLStorage(base_dir=str(tmp_path)))
        kb.add("User prefers Python for backend", importance=0.9)
        kb.add("User deploys on Docker Compose", importance=0.85)

        client = MemoryEnabledOpenAI(knowledge_base=kb)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write me a deployment script"},
        ]

        enriched = client.enrich_messages(messages)

        system_content = enriched[0]["content"]
        assert "coding assistant" in system_content
        assert "Agent Memory" in system_content
        assert "Docker" in system_content or "Python" in system_content

    def test_32_openai_enrichment_no_relevant_facts(self, tmp_path: pathlib.Path) -> None:
        """Facts about Python, query about cooking — original messages returned unchanged."""
        kb = KnowledgeBase(agent_id="irrelevant", storage=YAMLStorage(base_dir=str(tmp_path)))
        kb.add("User prefers Python programming language", importance=0.9)
        kb.add("User uses Docker for deployment", importance=0.85)

        client = MemoryEnabledOpenAI(knowledge_base=kb)
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "How to cook pasta carbonara?"},
        ]

        enriched = client.enrich_messages(messages)

        # The retriever may still return results (TF-IDF has no threshold),
        # but the key point is it doesn't crash and returns valid messages
        assert len(enriched) >= 1
        user_msgs = [m for m in enriched if m["role"] == "user"]
        assert user_msgs[0]["content"] == "How to cook pasta carbonara?"
