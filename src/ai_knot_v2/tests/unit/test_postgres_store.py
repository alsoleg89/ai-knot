"""Sprint 21 — PostgresStore unit tests (import-only; psycopg3 not installed in CI)."""

from __future__ import annotations

import importlib
import sys


class TestPostgresStoreImport:
    def test_module_importable(self) -> None:
        mod = importlib.import_module("ai_knot_v2.store.postgres")
        assert hasattr(mod, "PostgresStore")

    def test_postgres_store_raises_without_psycopg(self, monkeypatch: object) -> None:
        """If psycopg3 is missing, __init__ raises a descriptive ImportError."""
        import builtins

        real_import = builtins.__import__

        def _block_psycopg(name: str, *args: object, **kwargs: object) -> object:
            if name == "psycopg":
                raise ImportError("No module named 'psycopg'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        import ai_knot_v2.store.postgres as pg_mod

        # remove cached module so lazy import runs again
        sys.modules.pop("psycopg", None)
        monkeypatch.setattr(builtins, "__import__", _block_psycopg)

        import pytest

        with pytest.raises(ImportError, match="psycopg3 is required"):
            pg_mod.PostgresStore.__new__(pg_mod.PostgresStore).__init__("postgresql://x/y")

    def test_ddl_contains_all_tables(self) -> None:
        from ai_knot_v2.store.postgres import _PG_DDL

        for table in ("episodes", "atoms", "evidence_packs", "audit_trail"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in _PG_DDL

    def test_atom_to_row_round_trip(self) -> None:
        import time

        from ai_knot_v2.core.atom import MemoryAtom
        from ai_knot_v2.store.postgres import _atom_to_row, _row_to_atom

        now = int(time.time())
        atom = MemoryAtom(
            atom_id="pg-test-1",
            agent_id="a1",
            user_id="u1",
            variables=("x",),
            causal_graph=(("x", "y"),),
            kernel_kind="fact",
            kernel_payload={"k": "v"},
            intervention_domain=("x",),
            predicate="has_allergy",
            subject="Alice",
            object_value="penicillin",
            polarity="positive",
            valid_from=None,
            valid_until=None,
            observation_time=now,
            belief_time=now,
            granularity="day",
            entity_orbit_id="orbit-alice",
            transport_provenance=("ep-1",),
            depends_on=(),
            depended_by=(),
            risk_class="critical",
            risk_severity=0.9,
            regret_charge=0.8,
            irreducibility_score=0.7,
            protection_energy=0.6,
            action_affect_mask=0,
            credence=0.95,
            evidence_episodes=("ep-1",),
            synthesis_method="direct",
            validation_tests=(),
            contradiction_events=(),
        )
        row = _atom_to_row(atom)
        restored = _row_to_atom(row)

        assert restored.atom_id == atom.atom_id
        assert restored.predicate == atom.predicate
        assert restored.object_value == atom.object_value
        assert restored.risk_class == atom.risk_class
        assert abs(restored.credence - atom.credence) < 1e-6
