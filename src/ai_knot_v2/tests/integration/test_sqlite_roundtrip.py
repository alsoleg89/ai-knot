"""Sprint 2 — SQLite roundtrip integration tests."""

from __future__ import annotations

import dataclasses
import time
from typing import Literal, cast

import pytest

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.provenance import AuditEvent, AuditTrail
from ai_knot_v2.core.types import EvidencePack, EvidenceSpan
from ai_knot_v2.store.sqlite import SqliteStore


@pytest.fixture
def store() -> SqliteStore:
    return SqliteStore(":memory:")


def _make_atom(atom_id: str | None = None) -> MemoryAtom:
    return MemoryAtom(
        atom_id=atom_id or new_ulid(),
        agent_id="agent-1",
        user_id="user-42",
        variables=("salary",),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={"value": 120_000},
        intervention_domain=("salary",),
        predicate="has_salary",
        subject="user-42",
        object_value="120000",
        polarity="pos",
        valid_from=1_700_000_000,
        valid_until=None,
        observation_time=1_700_000_100,
        belief_time=1_700_000_100,
        granularity="day",
        entity_orbit_id="entity:user-42",
        transport_provenance=("session-1",),
        depends_on=(),
        depended_by=(),
        risk_class="finance",
        risk_severity=0.3,
        regret_charge=0.1,
        irreducibility_score=0.8,
        protection_energy=0.5,
        action_affect_mask=1,
        credence=0.95,
        evidence_episodes=("ep-001",),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


def _make_episode(episode_id: str | None = None) -> RawEpisode:
    return RawEpisode(
        episode_id=episode_id or new_ulid(),
        agent_id="agent-1",
        user_id="user-42",
        session_id="session-1",
        turn_index=0,
        speaker="user",
        text="My salary is 120k.",
        timestamp=1_700_000_000,
    )


class TestAtomRoundtrip:
    def test_save_and_get_atom(self, store: SqliteStore) -> None:
        atom = _make_atom()
        store.save_atom(atom)
        retrieved = store.get_atom(atom.atom_id)
        assert retrieved == atom

    def test_get_nonexistent_atom_returns_none(self, store: SqliteStore) -> None:
        assert store.get_atom("NONEXISTENT") is None

    def test_save_atom_is_idempotent(self, store: SqliteStore) -> None:
        atom = _make_atom()
        store.save_atom(atom)
        store.save_atom(atom)  # should not raise
        retrieved = store.get_atom(atom.atom_id)
        assert retrieved == atom

    def test_atom_with_none_user_id(self, store: SqliteStore) -> None:
        atom = _make_atom()
        atom_no_user = MemoryAtom(**{**dataclasses.asdict(atom), "user_id": None})
        store.save_atom(atom_no_user)
        retrieved = store.get_atom(atom_no_user.atom_id)
        assert retrieved == atom_no_user
        assert retrieved is not None and retrieved.user_id is None

    def test_query_by_entity(self, store: SqliteStore) -> None:
        a1 = _make_atom()
        a2 = _make_atom()
        store.save_atom(a1)
        store.save_atom(a2)
        results = store.get_atoms_by_entity("entity:user-42")
        assert len(results) == 2

    def test_atom_tuple_fields_preserved(self, store: SqliteStore) -> None:
        atom = _make_atom()
        store.save_atom(atom)
        retrieved = store.get_atom(atom.atom_id)
        assert retrieved is not None
        assert isinstance(retrieved.variables, tuple)
        assert isinstance(retrieved.depends_on, tuple)
        assert isinstance(retrieved.evidence_episodes, tuple)
        assert isinstance(retrieved.causal_graph, tuple)


class TestEpisodeRoundtrip:
    def test_save_and_get_episode(self, store: SqliteStore) -> None:
        ep = _make_episode()
        store.save_episode(ep)
        retrieved = store.get_episode(ep.episode_id)
        assert retrieved == ep

    def test_get_nonexistent_episode_returns_none(self, store: SqliteStore) -> None:
        assert store.get_episode("NONEXISTENT") is None


class TestEvidencePackRoundtrip:
    def test_save_and_get_empty_pack(self, store: SqliteStore) -> None:
        pack = EvidencePack(pack_id=new_ulid(), atoms=("atom-1",), spans=())
        store.save_evidence_pack(pack)
        retrieved = store.get_evidence_pack(pack.pack_id)
        assert retrieved is not None
        assert retrieved.pack_id == pack.pack_id
        assert retrieved.atoms == pack.atoms

    def test_get_nonexistent_pack_returns_none(self, store: SqliteStore) -> None:
        assert store.get_evidence_pack("NONEXISTENT") is None

    def test_pack_with_spans(self, store: SqliteStore) -> None:
        span = EvidenceSpan(
            episode_id="ep-001",
            start_char=0,
            end_char=18,
            text="My salary is 120k.",
            relevance_score=0.9,
        )
        pack = EvidencePack(pack_id=new_ulid(), atoms=("atom-1",), spans=(span,))
        store.save_evidence_pack(pack)
        retrieved = store.get_evidence_pack(pack.pack_id)
        assert retrieved is not None
        assert len(retrieved.spans) == 1
        assert retrieved.spans[0].text == "My salary is 120k."


class TestAuditTrail:
    def test_append_and_trace(self, store: SqliteStore) -> None:
        atom_id = new_ulid()
        event = AuditEvent(
            event_id=new_ulid(),
            operation="write",
            atom_id=atom_id,
            agent_id="agent-1",
            timestamp=int(time.time()),
            details={"source": "test"},
        )
        store.append_audit_event(event)
        trail = store.trace(atom_id)
        assert isinstance(trail, AuditTrail)
        assert trail.atom_id == atom_id
        assert len(trail.events) == 1
        assert trail.events[0].operation == "write"

    def test_trace_empty_returns_empty_trail(self, store: SqliteStore) -> None:
        trail = store.trace("NONEXISTENT")
        assert trail.atom_id == "NONEXISTENT"
        assert len(trail.events) == 0

    def test_multiple_events_ordered_by_timestamp(self, store: SqliteStore) -> None:
        atom_id = new_ulid()
        for i, op in enumerate(["write", "read", "forget"]):
            store.append_audit_event(
                AuditEvent(
                    event_id=new_ulid(),
                    operation=cast(Literal["write", "forget", "consolidate", "read"], op),
                    atom_id=atom_id,
                    agent_id="agent-1",
                    timestamp=1_700_000_000 + i,
                )
            )
        trail = store.trace(atom_id)
        assert len(trail.events) == 3
        assert [e.operation for e in trail.events] == ["write", "read", "forget"]
