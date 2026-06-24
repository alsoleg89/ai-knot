"""Tests for YAML storage backend."""

from __future__ import annotations

import pathlib

from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MemoryType


class TestYAMLStorageSaveLoad:
    """Basic save/load round-trip."""

    def test_save_and_load(self, yaml_storage: YAMLStorage, sample_facts: list[Fact]) -> None:
        yaml_storage.save("agent1", sample_facts)
        loaded = yaml_storage.load("agent1")
        assert len(loaded) == len(sample_facts)
        assert loaded[0].content == sample_facts[0].content

    def test_load_nonexistent_agent(self, yaml_storage: YAMLStorage) -> None:
        loaded = yaml_storage.load("nonexistent")
        assert loaded == []

    def test_overwrite_replaces(self, yaml_storage: YAMLStorage) -> None:
        facts_v1 = [Fact(content="version 1")]
        facts_v2 = [Fact(content="version 2"), Fact(content="version 2b")]

        yaml_storage.save("agent1", facts_v1)
        yaml_storage.save("agent1", facts_v2)

        loaded = yaml_storage.load("agent1")
        assert len(loaded) == 2
        assert loaded[0].content == "version 2"

    def test_preserves_all_fields(self, yaml_storage: YAMLStorage) -> None:
        fact = Fact(
            content="Full field test",
            type=MemoryType.PROCEDURAL,
            importance=0.42,
            retention_score=0.77,
            access_count=13,
            tags=["a", "b", "c"],
        )
        yaml_storage.save("agent1", [fact])
        loaded = yaml_storage.load("agent1")[0]

        assert loaded.content == fact.content
        assert loaded.type == fact.type
        assert loaded.importance == fact.importance
        assert loaded.retention_score == fact.retention_score
        assert loaded.access_count == fact.access_count
        assert loaded.tags == fact.tags
        assert loaded.id == fact.id


class TestYAMLStorageMultiAgent:
    """Multiple agents stored independently."""

    def test_agents_isolated(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save("alice", [Fact(content="Alice fact")])
        yaml_storage.save("bob", [Fact(content="Bob fact")])

        alice_facts = yaml_storage.load("alice")
        bob_facts = yaml_storage.load("bob")

        assert len(alice_facts) == 1
        assert len(bob_facts) == 1
        assert alice_facts[0].content == "Alice fact"
        assert bob_facts[0].content == "Bob fact"

    def test_list_agents(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save("agent_a", [Fact(content="a")])
        yaml_storage.save("agent_b", [Fact(content="b")])

        agents = yaml_storage.list_agents()
        assert set(agents) == {"agent_a", "agent_b"}


class TestYAMLStorageDelete:
    """Deleting individual facts."""

    def test_delete_fact(self, yaml_storage: YAMLStorage) -> None:
        facts = [Fact(content="keep"), Fact(content="delete me")]
        yaml_storage.save("agent1", facts)

        yaml_storage.delete("agent1", facts[1].id)

        loaded = yaml_storage.load("agent1")
        assert len(loaded) == 1
        assert loaded[0].content == "keep"

    def test_delete_nonexistent_fact(self, yaml_storage: YAMLStorage) -> None:
        yaml_storage.save("agent1", [Fact(content="only")])
        # Should not raise
        yaml_storage.delete("agent1", "nonexistent_id")
        loaded = yaml_storage.load("agent1")
        assert len(loaded) == 1


class TestYAMLStorageFileFormat:
    """Verify files are human-readable YAML."""

    def test_creates_yaml_file(self, yaml_storage: YAMLStorage, tmp_dir: pathlib.Path) -> None:
        yaml_storage.save("myagent", [Fact(content="readable")])

        yaml_path = tmp_dir / "myagent" / "knowledge.yaml"
        assert yaml_path.exists()

        content = yaml_path.read_text()
        assert "readable" in content
        assert "content:" in content or "readable" in content
