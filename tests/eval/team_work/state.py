"""Study-only source deltas; publishers never reset executor state."""

from __future__ import annotations

from datetime import timedelta

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory
from ai_knot.storage.base import AtomicUpdateCapable
from ai_knot.types import Fact, MemoryType
from tests.eval.team_repos.runner import OWNER, identity, moment


def source_facts(case, index: int) -> tuple[Fact, ...]:
    if type(index) is not int or not 0 <= index < len(case.stages):
        raise ValueError("invalid stage index")
    previous = {}
    for number, stage in enumerate(case.stages[: index + 1]):
        current = {}
        for rank, doc in enumerate(sorted(stage.documents, key=lambda d: d.path)):
            key = identity(case.id, doc.path)
            text = doc.path + "\n" + doc.text
            qualifiers = {"task_key": doc.task_key} if doc.task_key else {}
            old = previous.get(key)
            if old is not None and old.content == text and old.qualifiers == qualifiers:
                current[key] = old
                continue
            observed = moment(number) + timedelta(microseconds=rank)
            current[key] = Fact(
                text,
                id=key,
                type=MemoryType.SEMANTIC,
                qualifiers=qualifiers,
                created_at=observed,
                valid_from=observed,
                event_time=observed,
                last_accessed=observed,
                version=1 if old is None else old.version + 1,
                slot_key=case.id + "::" + doc.path,
                value_text=text,
                origin_agent_id="repo-observer",
                visibility_scope="global",
            )
        previous = current
    return tuple(previous.values())


def publish_stage(storage: AtomicUpdateCapable, case, index: int) -> None:
    facts = source_facts(case, index)
    prefix = case.id + "::"
    storage.atomic_update(
        "__shared__",
        lambda old: (
            [
                fact
                for fact in old
                if not (
                    fact.origin_agent_id == "repo-observer" and fact.slot_key.startswith(prefix)
                )
            ]
            + list(facts)
        ),
    )


def ordered_entries(rows) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (row["ref"]["namespace"], row["ref"]["fact_id"], row["ref"]["revision"]),
    )
    return [{**row, "index": i} for i, row in enumerate(ordered, 1)]


def memory_for(storage, index: int) -> AlwaysOnMemory:
    # Sources use distinct microseconds; all are active before the reader starts.
    def clock():
        return moment(index) + timedelta(seconds=1)

    return AlwaysOnMemory(
        KnowledgeBase(OWNER, storage=storage, embed_url=""),
        pool=SharedMemoryPool(storage, persist_stats=True, embed_url="", clock=clock),
        clock=clock,
    )
