"""MemoryAPI — the public product surface for ai-knot v2.

This class composes the full pipeline: learn, recall, explain, trace, inspect.
All persistence is via SqliteStore + AtomLibrary.
No LLM calls anywhere in this class.
"""

from __future__ import annotations

import time

from ai_knot_v2.api.sdk import (
    AtomDTO,
    ExplainResponse,
    InspectResponse,
    LearnRequest,
    LearnResponse,
    RecallRequest,
    RecallResponse,
    TraceResponse,
)
from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.groupoid import EntityGroupoid
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.types import ReaderBudget
from ai_knot_v2.ops.read import recall as _recall
from ai_knot_v2.ops.write import write_episodes
from ai_knot_v2.store.sqlite import SqliteStore


def _atom_to_dto(atom: object) -> AtomDTO:
    from ai_knot_v2.core.atom import MemoryAtom

    a: MemoryAtom = atom  # type: ignore[assignment]
    return AtomDTO(
        atom_id=a.atom_id,
        predicate=a.predicate,
        subject=a.subject,
        object_value=a.object_value,
        polarity=a.polarity,
        risk_class=a.risk_class,
        risk_severity=a.risk_severity,
        credence=a.credence,
        valid_from=a.valid_from,
        valid_until=a.valid_until,
        entity_orbit_id=a.entity_orbit_id,
        synthesis_method=a.synthesis_method,
    )


class MemoryAPI:
    """Top-level product API.

    Usage::

        api = MemoryAPI(db_path=":memory:")
        api.learn(LearnRequest(episodes=[EpisodeIn(text="Alice is a doctor.")]))
        result = api.recall(RecallRequest(query="What is Alice's job?"))
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        agent_id: str = "agent-1",
    ) -> None:
        self._store = SqliteStore(db_path)
        self._library = AtomLibrary()
        self._groupoid = EntityGroupoid()
        self._agent_id = agent_id

    def learn(self, request: LearnRequest) -> LearnResponse:
        """Ingest episodes, extract atoms, persist to store + library."""
        episodes: list[RawEpisode] = []
        for ep in request.episodes:
            episodes.append(
                RawEpisode(
                    episode_id=new_ulid(),
                    agent_id=ep.agent_id or request.agent_id or self._agent_id,
                    user_id=ep.user_id if ep.user_id is not None else request.user_id,
                    session_id=ep.session_id,
                    turn_index=0,
                    speaker=ep.speaker,  # type: ignore[arg-type]
                    text=ep.text,
                    timestamp=ep.timestamp or int(time.time()),
                    metadata=ep.metadata,
                )
            )

        result = write_episodes(episodes, self._store, self._library, self._groupoid)

        return LearnResponse(
            episode_ids=list(result.episode_ids),
            atom_ids=list(result.atom_ids),
            skipped_duplicate=result.skipped_duplicate,
            skipped_dominated=result.skipped_dominated,
        )

    def recall(self, request: RecallRequest) -> RecallResponse:
        """Retrieve relevant atoms for a query."""
        budget = ReaderBudget(
            max_atoms=request.max_atoms,
            max_tokens=request.max_tokens,
            require_dependency_closure=request.require_dependency_closure,
        )
        result = _recall(request.query, self._library, budget)

        return RecallResponse(
            query=request.query,
            atoms=[_atom_to_dto(a) for a in result.atoms],
            evidence_pack_id=result.evidence_pack_id,
            intervention_variable=(
                result.intervention.variable if result.intervention else "general"
            ),
        )

    def explain(self, atom_id: str) -> ExplainResponse:
        """Return provenance explanation for a specific atom."""
        atom = self._library.get(atom_id)
        if atom is None:
            raise KeyError(f"Atom {atom_id!r} not found in library")

        return ExplainResponse(
            atom_id=atom.atom_id,
            predicate=atom.predicate,
            subject=atom.subject,
            object_value=atom.object_value,
            evidence_episodes=list(atom.evidence_episodes),
            risk_class=atom.risk_class,
            synthesis_method=atom.synthesis_method,
        )

    def trace(self, atom_id: str) -> TraceResponse:
        """Return audit trail for a specific atom."""
        trail = self._store.trace(atom_id)
        events = [
            {
                "event_id": event.event_id,
                "operation": event.operation,
                "atom_id": event.atom_id,
                "agent_id": event.agent_id,
                "timestamp": event.timestamp,
                "details": event.details,
            }
            for event in trail.events
        ]
        return TraceResponse(atom_id=atom_id, events=events)

    def inspect_memory(
        self,
        filters: dict[str, str] | None = None,
    ) -> InspectResponse:
        """Return atoms from library, optionally filtered.

        Supported filter keys: risk_class, predicate, entity_orbit_id.
        """
        filters = filters or {}
        atoms = self._library.all_atoms()

        if "risk_class" in filters:
            atoms = [a for a in atoms if a.risk_class == filters["risk_class"]]
        if "predicate" in filters:
            atoms = [a for a in atoms if a.predicate == filters["predicate"]]
        if "entity_orbit_id" in filters:
            atoms = [a for a in atoms if a.entity_orbit_id == filters["entity_orbit_id"]]

        return InspectResponse(
            atoms=[_atom_to_dto(a) for a in atoms],
            total=len(atoms),
        )
