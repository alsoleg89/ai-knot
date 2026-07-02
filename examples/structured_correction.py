"""Repo-native proof of structured updates, deletes, and lineage.

Run::

    python examples/structured_correction.py
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

from ai_knot import Fact, KnowledgeBase, MemoryOp
from ai_knot.storage import YAMLStorage


@dataclass
class HistoryRow:
    content: str
    slot_key: str
    version: int
    active: bool


@dataclass
class StructuredCorrectionResult:
    search_output: str
    active_contents: list[str]
    history_rows: list[HistoryRow]
    employer_lineage: list[str]


def build_demo_result() -> StructuredCorrectionResult:
    with TemporaryDirectory(prefix="ai-knot-structured-demo-") as data_dir:
        kb = KnowledgeBase(
            agent_id="structured-demo",
            storage=YAMLStorage(base_dir=data_dir),
            embed_url="",
        )

        kb.add_resolved(
            [
                Fact(
                    content="User works at Acme",
                    entity="user",
                    attribute="employer",
                    value_text="Acme",
                ),
                Fact(
                    content="User works from Berlin",
                    entity="user",
                    attribute="office",
                    value_text="Berlin",
                ),
            ]
        )

        current_employer = kb.add_resolved(
            [
                Fact(
                    content="User now works at Globex",
                    entity="user",
                    attribute="employer",
                    value_text="Globex",
                    op=MemoryOp.UPDATE,
                )
            ]
        )[0]

        kb.add_resolved(
            [
                Fact(
                    content="User no longer works from Berlin",
                    entity="user",
                    attribute="office",
                    slot_key="user::office",
                    op=MemoryOp.DELETE,
                )
            ]
        )

        all_facts = [
            fact
            for fact in kb.list_facts()
            if fact.slot_key and not fact.slot_key.endswith("::_agg")
        ]
        active_facts = [fact for fact in all_facts if fact.is_active()]
        history_rows = sorted(
            [
                HistoryRow(
                    content=fact.content,
                    slot_key=fact.slot_key,
                    version=fact.version,
                    active=fact.is_active(),
                )
                for fact in all_facts
            ],
            key=lambda row: (0 if row.active else 1, row.slot_key, -row.version, row.content),
        )
        recalled = [
            fact
            for fact in kb.recall_facts("where does the user work now?")
            if not fact.slot_key.endswith("::_agg")
        ]

        return StructuredCorrectionResult(
            search_output="\n".join(
                f"[{idx}] {fact.content}" for idx, fact in enumerate(recalled, start=1)
            ),
            active_contents=[fact.content for fact in active_facts],
            history_rows=history_rows,
            employer_lineage=[fact.content for fact in kb.lineage(current_employer.id)],
        )


def main() -> None:
    result = build_demo_result()

    print("=== Search ===")
    print(result.search_output)
    print()

    print("=== Active memories ===")
    for content in result.active_contents:
        print(f"  - {content}")
    print()

    print("=== Full history ===")
    for row in result.history_rows:
        status = "active" if row.active else "inactive"
        print(f"  - [{status}] {row.slot_key} v{row.version}: {row.content}")
    print()

    print("=== Employer lineage ===")
    for idx, content in enumerate(result.employer_lineage, start=1):
        print(f"  {idx}. {content}")


if __name__ == "__main__":
    main()
