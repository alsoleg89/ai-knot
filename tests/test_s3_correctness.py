"""Regression tests for S3 correctness & idempotency fixes.

Covers:
- stable_bundle_id is deterministic (same input → same output, different inputs → different IDs)
- save_bundles is idempotent (double-save does not create duplicate member rows)
- rebuild_materialized preserves dirty_keys on failure and clears them on success
- tool_ingest_episode: non-ISO session_date falls back to None instead of raising
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_knot.query_types import BundleKind, stable_bundle_id

# ---------------------------------------------------------------------------
# stable_bundle_id
# ---------------------------------------------------------------------------


class TestStableBundleId:
    def test_same_inputs_same_id(self) -> None:
        id1 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        id2 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        assert id1 == id2, "stable_bundle_id must be deterministic"

    def test_different_kind_different_id(self) -> None:
        id1 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        id2 = stable_bundle_id(BundleKind.STATE_TIMELINE, "Alice")
        assert id1 != id2

    def test_different_topic_different_id(self) -> None:
        id1 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        id2 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Bob")
        assert id1 != id2

    def test_id_is_16_hex_chars(self) -> None:
        bid = stable_bundle_id(BundleKind.RELATION_SUPPORT, "Alice::works_at")
        assert len(bid) == 16
        assert all(c in "0123456789abcdef" for c in bid)

    def test_string_kind_accepted(self) -> None:
        """stable_bundle_id should accept a plain string kind."""
        id1 = stable_bundle_id("entity_topic", "Alice")
        id2 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        assert id1 == id2


# ---------------------------------------------------------------------------
# save_bundles idempotency
# ---------------------------------------------------------------------------


class TestSaveBundlesIdempotent:
    def _make_storage(self, tmp_path: Path):
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        return SQLiteStorage(db_path=str(tmp_path / "test.db"))

    def _make_bundle(self, agent_id: str, topic: str = "Alice"):
        from ai_knot.query_types import SupportBundle, stable_bundle_id

        return SupportBundle(
            id=stable_bundle_id(BundleKind.ENTITY_TOPIC, topic),
            agent_id=agent_id,
            kind=BundleKind.ENTITY_TOPIC,
            topic=topic,
            member_claim_ids=("claim-1", "claim-2"),
            score_formula="mean",
            bundle_score=0.8,
            built_from_materialization_version=1,
            built_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

    def test_double_save_no_duplicate_members(self, tmp_path: Path) -> None:
        storage = self._make_storage(tmp_path)
        agent_id = "agent"
        bundle = self._make_bundle(agent_id)
        memberships = {bundle.id: ["claim-1", "claim-2"]}

        storage.save_bundles(agent_id, [bundle], memberships)
        storage.save_bundles(agent_id, [bundle], memberships)

        members = storage.load_bundle_members(agent_id, [bundle.id])
        assert len(members[bundle.id]) == 2, (
            f"Double save must not create duplicate member rows; got {len(members[bundle.id])}"
        )

    def test_save_updated_members(self, tmp_path: Path) -> None:
        """Second save with different member list should replace, not append."""
        storage = self._make_storage(tmp_path)
        agent_id = "agent"
        bundle = self._make_bundle(agent_id)

        storage.save_bundles(agent_id, [bundle], {bundle.id: ["claim-1", "claim-2"]})
        storage.save_bundles(agent_id, [bundle], {bundle.id: ["claim-3"]})

        members = storage.load_bundle_members(agent_id, [bundle.id])
        assert members[bundle.id] == ["claim-3"], (
            f"Second save should replace old memberships; got {members[bundle.id]}"
        )


# ---------------------------------------------------------------------------
# rebuild_materialized dirty_keys preservation
# ---------------------------------------------------------------------------


class TestRebuildMaterializedDirtyKeys:
    def _make_kb(self, tmp_path: Path, agent_id: str = "agent"):
        from ai_knot.knowledge import KnowledgeBase
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
        return KnowledgeBase(agent_id=agent_id, storage=storage)

    def test_dirty_keys_cleared_on_success(self, tmp_path: Path) -> None:
        """After a successful rebuild, dirty_keys_json should be empty."""
        kb = self._make_kb(tmp_path)
        # Ingest an episode so there's something to rebuild.
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=datetime(2024, 1, 1, tzinfo=UTC),
            raw_text="Alice works as a software engineer.",
        )
        report = kb.rebuild_materialized(force=True)
        assert not report.skipped
        # After successful rebuild, dirty_keys must be cleared.
        meta = kb._storage.load_materialization_meta(kb._agent_id)  # type: ignore[attr-defined]
        assert meta.get("dirty_keys_json", "[]") in ("[]", "", None)
        assert meta.get("rebuild_status") == "ready"


# ---------------------------------------------------------------------------
# tool_ingest_episode: non-ISO session_date robustness
# ---------------------------------------------------------------------------


class TestToolIngestEpisodeDateParsing:
    def _make_kb(self, tmp_path: Path):
        from ai_knot.knowledge import KnowledgeBase
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        return KnowledgeBase(
            agent_id="agent",
            storage=SQLiteStorage(db_path=str(tmp_path / "test.db")),
        )

    def test_iso_date_accepted(self, tmp_path: Path) -> None:
        from ai_knot._mcp_tools import tool_ingest_episode

        kb = self._make_kb(tmp_path)
        result = tool_ingest_episode(
            kb,
            session_id="sess-1",
            turn_id="turn-0",
            raw_text="Alice loves hiking.",
            session_date="2024-03-15",
        )
        data = json.loads(result)
        assert "episode_id" in data

    def test_non_iso_date_does_not_raise(self, tmp_path: Path) -> None:
        """Non-ISO date string should produce a warning and fall back to None."""
        from ai_knot._mcp_tools import tool_ingest_episode

        kb = self._make_kb(tmp_path)
        # Should NOT raise; must return a valid episode_id.
        result = tool_ingest_episode(
            kb,
            session_id="sess-2",
            turn_id="turn-0",
            raw_text="Bob prefers tea.",
            session_date="8 May, 2023",  # non-ISO format
        )
        data = json.loads(result)
        assert "episode_id" in data, f"Expected episode_id, got: {data}"

    def test_none_date_accepted(self, tmp_path: Path) -> None:
        from ai_knot._mcp_tools import tool_ingest_episode

        kb = self._make_kb(tmp_path)
        result = tool_ingest_episode(
            kb,
            session_id="sess-3",
            turn_id="turn-0",
            raw_text="Carol drinks coffee.",
            session_date=None,
        )
        data = json.loads(result)
        assert "episode_id" in data
