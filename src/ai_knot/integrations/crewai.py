"""CrewAI memory adapter.

ai-knot complements CrewAI's orchestration/runtime with deterministic,
self-hosted long-term memory. The adapter is designed for CrewAI's native
``Crew(memory=...)`` / ``Agent(memory=...)`` surfaces: it stores facts in an
``ai_knot.KnowledgeBase`` and returns CrewAI-shaped memory records and matches.

There is **no hard dependency** on ``crewai``. Importing this module is safe
without CrewAI installed; the class falls back to lightweight shims for direct
method use and tests. When CrewAI *is* installed before import, the adapter is a
real subclass of CrewAI's ``Memory`` model, so you can pass it straight into
``Crew(memory=...)`` or scope it for an agent with ``memory.scope(...)``.

Example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.crewai import AiKnotCrewAIMemory

    kb = KnowledgeBase("assistant", provider="openai")
    kb.add("User prefers Python over Java")
    kb.add("User deploys APIs with Docker and Kubernetes")

    memory = AiKnotCrewAIMemory(kb, top_k=5)
    # Pass ``memory`` into Crew(memory=...) or Agent(memory=memory.scope(...))
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ai_knot.extractor import Extractor
from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import ConversationTurn, Fact, MemoryType

try:  # pragma: no cover - exercised via reload tests
    from crewai.memory.unified_memory import Memory as _CrewAIMemoryBase
except ImportError:  # pragma: no cover - no CrewAI installed in normal test env
    _CrewAIMemoryBase = object


_CREWAI_IMPORT_ERROR = (
    "CrewAI memory integration requires CrewAI. "
    'Install with: pip install "ai-knot[crewai]" (or pip install crewai)'
)
_SCOPE_KEY = "crewai_scope"
_CATEGORIES_KEY = "crewai_categories_json"
_METADATA_KEY = "crewai_metadata_json"
_PRIVATE_KEY = "crewai_private"
_SOURCE_KEY = "crewai_source"


def _normalise_scope(path: str | None) -> str:
    if not path or path == "/":
        return "/"
    scope = path.rstrip("/")
    if not scope.startswith("/"):
        scope = "/" + scope
    return scope or "/"


def _join_scope(root: str | None, path: str | None) -> str:
    root_scope = _normalise_scope(root)
    path_scope = _normalise_scope(path)
    if root_scope == "/":
        return path_scope
    if path_scope == "/":
        return root_scope
    return f"{root_scope.rstrip('/')}{path_scope}"


def _child_scope(parent: str, scope: str) -> str | None:
    root = _normalise_scope(parent)
    child = _normalise_scope(scope)
    if root == "/":
        relative = child.lstrip("/")
    else:
        if child != root and not child.startswith(root.rstrip("/") + "/"):
            return None
        relative = child[len(root.rstrip("/")) :].lstrip("/")
    if not relative:
        return None
    return _join_scope(root, relative.split("/", 1)[0])


def _scope_matches(scope: str, prefix: str | None) -> bool:
    if prefix is None:
        return True
    norm_scope = _normalise_scope(scope)
    norm_prefix = _normalise_scope(prefix)
    return (
        norm_prefix == "/"
        or norm_scope == norm_prefix
        or norm_scope.startswith(norm_prefix.rstrip("/") + "/")
    )


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)


def _clean_categories(categories: Iterable[str] | None) -> list[str]:
    if not categories:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in categories:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _fallback_extract_memories(content: str) -> list[str]:
    text = content.strip()
    if not text:
        return []

    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+\.)\s*", "", raw_line).strip()
        if line:
            lines.append(line)
    if len(lines) > 1:
        return lines

    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if part.strip()
    ]
    if 1 < len(sentences) <= 6:
        return sentences
    return [text]


def _memory_type(value: object, *, default: MemoryType = MemoryType.SEMANTIC) -> MemoryType:
    if isinstance(value, MemoryType):
        return value
    if isinstance(value, str):
        try:
            return MemoryType(value)
        except ValueError:
            return default
    return default


def _importance(value: object, *, default: float = 0.8) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _crewai_memory_scope_cls() -> type[Any] | None:
    try:
        from crewai.memory.memory_scope import MemoryScope
    except ImportError:
        return None
    return MemoryScope


def _crewai_types() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo
    except ImportError:
        return _MemoryRecordShim, _MemoryMatchShim, _ScopeInfoShim
    return MemoryRecord, MemoryMatch, ScopeInfo


class _NoOpStorage:
    """Placeholder storage passed into CrewAI's base ``Memory`` model.

    The adapter overrides persistence and retrieval with ai-knot, so this
    storage should never be used in practice. It exists purely to avoid pulling
    in CrewAI's default LanceDB backend during adapter construction.
    """

    def save(self, records: list[Any]) -> None:
        return None

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        return []

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return 0

    def update(self, record: Any) -> None:
        return None

    def get_record(self, record_id: str) -> Any | None:
        return None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        return []

    def get_scope_info(self, scope: str) -> Any:
        _, _, ScopeInfo = _crewai_types()
        return ScopeInfo(
            path=scope,
            record_count=0,
            categories=[],
            oldest_record=None,
            newest_record=None,
            child_scopes=[],
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        return []

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        return {}

    def count(self, scope_prefix: str | None = None) -> int:
        return 0

    def reset(self, scope_prefix: str | None = None) -> None:
        return None

    async def asave(self, records: list[Any]) -> None:
        return None

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        return []

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return 0


@dataclass
class _MemoryRecordShim:
    id: str
    content: str
    scope: str = "/"
    categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    embedding: list[float] | None = None
    source: str | None = None
    private: bool = False


@dataclass
class _MemoryMatchShim:
    record: _MemoryRecordShim
    score: float
    match_reasons: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [f"- (score={self.score:.2f}) {self.record.content}"]
        if self.record.categories:
            lines.append(f"  categories: {', '.join(self.record.categories)}")
        for key, value in self.record.metadata.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class _ScopeInfoShim:
    path: str
    record_count: int = 0
    categories: list[str] = field(default_factory=list)
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    child_scopes: list[str] = field(default_factory=list)


class _FallbackMemoryScope:
    """CrewAI-like scope view used when CrewAI is not installed."""

    def __init__(self, memory: AiKnotCrewAIMemory, root_path: str) -> None:
        self._memory = memory
        self.root_path = _normalise_scope(root_path)

    @property
    def read_only(self) -> bool:
        return bool(getattr(self._memory, "read_only", False))

    def _scope_path(self, scope: str | None) -> str:
        return _join_scope(self.root_path, scope)

    def remember(
        self,
        content: str,
        scope: str | None = "/",
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
    ) -> Any:
        return self._memory.remember(
            content,
            scope=self._scope_path(scope),
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
        )

    def remember_many(
        self,
        contents: list[str],
        scope: str | None = "/",
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
    ) -> list[Any]:
        return self._memory.remember_many(
            contents,
            scope=self._scope_path(scope),
            categories=categories,
            metadata=metadata,
            importance=importance,
            source=source,
            private=private,
            agent_role=agent_role,
        )

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: Literal["shallow", "deep"] = "deep",
        source: str | None = None,
        include_private: bool = False,
    ) -> list[Any]:
        return self._memory.recall(
            query,
            scope=self._scope_path(scope),
            categories=categories,
            limit=limit,
            depth=depth,
            source=source,
            include_private=include_private,
        )

    def extract_memories(self, content: str) -> list[str]:
        return self._memory.extract_memories(content)

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        return self._memory.forget(
            scope=self._scope_path(scope),
            categories=categories,
            older_than=older_than,
            metadata_filter=metadata_filter,
            record_ids=record_ids,
        )

    def list_scopes(self, path: str = "/") -> list[str]:
        return self._memory.list_scopes(self._scope_path(path))

    def info(self, path: str = "/") -> Any:
        return self._memory.info(self._scope_path(path))

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        return self._memory.tree(self._scope_path(path), max_depth=max_depth)

    def list_categories(self, path: str | None = None) -> dict[str, int]:
        search_scope = self.root_path if path is None else self._scope_path(path)
        return self._memory.list_categories(search_scope)

    def reset(self, scope: str | None = None) -> None:
        self._memory.reset(self._scope_path(scope))

    def subscope(self, path: str) -> _FallbackMemoryScope:
        return _FallbackMemoryScope(self._memory, self._scope_path(path))


class AiKnotCrewAIMemory(_CrewAIMemoryBase):
    """CrewAI-compatible memory backed by ``KnowledgeBase``.

    The adapter keeps CrewAI's orchestration and memory-tool ergonomics while
    routing persistence and recall through ai-knot's fact store.

    Args:
        knowledge_base: The KnowledgeBase to use for long-term memory.
        top_k: Default number of facts to return for recall.
        default_type: Memory type used when CrewAI saves raw content without an
            explicit type.
        default_importance: Importance used when none is provided.
        read_only: When True, ``remember`` and ``remember_many`` are silent
            no-ops, mirroring CrewAI's read-only memory slices.
        root_scope: Optional root scope prefix used for all remember/recall
            operations. This mirrors CrewAI's hierarchical scope model.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        top_k: int = 5,
        default_type: MemoryType | str = MemoryType.SEMANTIC,
        default_importance: float = 0.8,
        read_only: bool = False,
        root_scope: str | None = None,
        **memory_kwargs: Any,
    ) -> None:
        if _CrewAIMemoryBase is object:
            self.memory_kind = "memory"
            self.read_only = read_only
            self.root_scope = root_scope
        else:
            super().__init__(
                storage=_NoOpStorage(),
                embedder=memory_kwargs.pop("embedder", lambda texts: [[0.0] for _ in texts]),
                read_only=read_only,
                root_scope=root_scope,
                **memory_kwargs,
            )
        object.__setattr__(self, "_kb", knowledge_base)
        object.__setattr__(self, "_top_k", max(1, top_k))
        object.__setattr__(self, "_default_type", _memory_type(default_type))
        object.__setattr__(self, "_default_importance", _importance(default_importance))

    def _overlay_fact(
        self,
        fact_id: str,
        *,
        scope: str,
        categories: list[str],
        metadata: dict[str, Any],
        source: str | None,
        private: bool,
    ) -> Fact:
        facts = self._kb.list_facts()
        for fact in facts:
            if fact.id != fact_id:
                continue
            fact.qualifiers[_SCOPE_KEY] = scope
            fact.qualifiers[_CATEGORIES_KEY] = _json_dump(categories)
            fact.qualifiers[_METADATA_KEY] = _json_dump(metadata)
            fact.qualifiers[_PRIVATE_KEY] = "1" if private else "0"
            if source is not None:
                fact.qualifiers[_SOURCE_KEY] = source
            elif _SOURCE_KEY in fact.qualifiers:
                fact.qualifiers.pop(_SOURCE_KEY)

            crewai_tags = [tag for tag in fact.tags if not tag.startswith("crewai:")]
            crewai_tags.extend(categories)
            crewai_tags.append(f"crewai:scope:{scope}")
            crewai_tags.extend(f"crewai:category:{item}" for item in categories)
            fact.tags = list(dict.fromkeys(tag for tag in crewai_tags if tag))
            self._kb.replace_facts(facts)
            return fact
        raise KeyError(fact_id)

    def _effective_scope(self, scope: str | None, root_scope: str | None = None) -> str:
        root = root_scope if root_scope is not None else getattr(self, "root_scope", None)
        return _join_scope(root, scope)

    def _fact_scope(self, fact: Fact) -> str:
        return _normalise_scope(fact.qualifiers.get(_SCOPE_KEY))

    def _fact_categories(self, fact: Fact) -> list[str]:
        stored = _parse_json_list(fact.qualifiers.get(_CATEGORIES_KEY))
        if stored:
            return stored
        return [
            tag
            for tag in fact.tags
            if tag and not tag.startswith("crewai:") and not tag.startswith("date:")
        ]

    def _fact_metadata(self, fact: Fact, *, score: float | None = None) -> dict[str, Any]:
        metadata = _parse_json_dict(fact.qualifiers.get(_METADATA_KEY))
        metadata.setdefault("type", str(fact.type))
        metadata.setdefault("created_at", fact.created_at.isoformat())
        if fact.tags:
            metadata.setdefault("tags", list(fact.tags))
        if fact.event_time is not None:
            metadata.setdefault("event_time", fact.event_time.isoformat())
        if score is not None:
            metadata["score"] = round(score, 4)
        return metadata

    def _fact_source(self, fact: Fact) -> str | None:
        value = fact.qualifiers.get(_SOURCE_KEY)
        return value or None

    def _fact_private(self, fact: Fact) -> bool:
        return fact.qualifiers.get(_PRIVATE_KEY, "0") == "1"

    def _fact_matches(
        self,
        fact: Fact,
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        source: str | None = None,
        include_private: bool = False,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> bool:
        if record_ids is not None and fact.id not in set(record_ids):
            return False
        if older_than is not None and not fact.created_at < older_than:
            return False
        if not _scope_matches(self._fact_scope(fact), scope_prefix):
            return False
        requested_categories = _clean_categories(categories)
        if requested_categories and not set(requested_categories).issubset(
            set(self._fact_categories(fact))
        ):
            return False
        if metadata_filter:
            fact_metadata = self._fact_metadata(fact)
            for key, value in metadata_filter.items():
                if fact_metadata.get(key) != value:
                    return False
        return not (
            self._fact_private(fact) and not include_private and self._fact_source(fact) != source
        )

    def _record_from_fact(self, fact: Fact, *, score: float | None = None) -> Any:
        MemoryRecord, _, _ = _crewai_types()
        return MemoryRecord(
            id=fact.id,
            content=fact.answer_surface,
            scope=self._fact_scope(fact),
            categories=self._fact_categories(fact),
            metadata=self._fact_metadata(fact, score=score),
            importance=fact.importance,
            created_at=fact.created_at,
            last_accessed=fact.last_accessed,
            embedding=None,
            source=self._fact_source(fact),
            private=self._fact_private(fact),
        )

    def _match_from_fact(self, fact: Fact, score: float) -> Any:
        _, MemoryMatch, _ = _crewai_types()
        return MemoryMatch(
            record=self._record_from_fact(fact, score=score),
            score=score,
            match_reasons=["ai-knot hybrid recall"],
            evidence_gaps=[],
        )

    def drain_writes(self) -> None:
        """CrewAI compatibility hook — ai-knot writes are synchronous here."""
        return None

    def remember(
        self,
        content: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,  # noqa: ARG002 - CrewAI API compat
        root_scope: str | None = None,
    ) -> Any:
        if getattr(self, "read_only", False):
            return None
        body = content.strip()
        if not body:
            return None

        metadata_dict = dict(metadata or {})
        clean_categories = _clean_categories(categories or metadata_dict.get("categories"))
        fact = self._kb.add(
            body,
            type=_memory_type(metadata_dict.get("type"), default=self._default_type),
            importance=_importance(
                importance if importance is not None else metadata_dict.get("importance"),
                default=self._default_importance,
            ),
            tags=clean_categories,
        )
        fact = self._overlay_fact(
            fact.id,
            scope=self._effective_scope(scope, root_scope),
            categories=clean_categories,
            metadata=metadata_dict,
            source=source,
            private=private,
        )
        return self._record_from_fact(fact)

    def remember_many(
        self,
        contents: list[str],
        scope: str | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        source: str | None = None,
        private: bool = False,
        agent_role: str | None = None,
        root_scope: str | None = None,
    ) -> list[Any]:
        records = []
        for item in contents:
            record = self.remember(
                item,
                scope=scope,
                categories=categories,
                metadata=metadata,
                importance=importance,
                source=source,
                private=private,
                agent_role=agent_role,
                root_scope=root_scope,
            )
            if record is not None:
                records.append(record)
        return records

    def extract_memories(self, content: str) -> list[str]:
        text = content.strip()
        if not text:
            return []

        provider = getattr(self._kb, "_default_provider", None)
        if provider is None:
            return _fallback_extract_memories(text)

        try:
            extractor = Extractor(
                provider=provider,
                api_key=getattr(self._kb, "_default_api_key", None),
                model=getattr(self._kb, "_default_model", None),
                **getattr(self._kb, "_default_provider_kwargs", {}),
            )
            facts = extractor.extract([ConversationTurn(role="user", content=text)])
        except Exception:
            return _fallback_extract_memories(text)

        extracted = [fact.content.strip() for fact in facts if fact.content.strip()]
        return extracted or _fallback_extract_memories(text)

    def recall(
        self,
        query: str,
        scope: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        depth: Literal["shallow", "deep"] = "deep",  # noqa: ARG002 - API compat
        source: str | None = None,
        include_private: bool = False,
    ) -> list[Any]:
        if not query.strip():
            return []

        raw_limit = max(limit * 10, self._top_k, 25)
        pairs = self._kb.recall_facts_with_scores(query, top_k=raw_limit)
        matches = [
            self._match_from_fact(fact, score)
            for fact, score in pairs
            if self._fact_matches(
                fact,
                scope_prefix=self._effective_scope(scope),
                categories=categories,
                source=source,
                include_private=include_private,
            )
        ]
        return matches[:limit]

    def forget(
        self,
        scope: str | None = None,
        categories: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        facts = self._kb.list_facts()
        keep: list[Fact] = []
        removed = 0
        for fact in facts:
            if self._fact_matches(
                fact,
                scope_prefix=self._effective_scope(scope),
                categories=categories,
                older_than=older_than,
                metadata_filter=metadata_filter,
                record_ids=record_ids,
                include_private=True,
            ):
                removed += 1
            else:
                keep.append(fact)
        if removed:
            self._kb.replace_facts(keep)
        return removed

    def list_scopes(self, parent: str = "/") -> list[str]:
        search_parent = self._effective_scope(parent)
        children = {
            child
            for fact in self._kb.list_facts()
            for child in [_child_scope(search_parent, self._fact_scope(fact))]
            if child is not None
        }
        return sorted(children)

    def info(self, scope: str = "/") -> Any:
        _, _, ScopeInfo = _crewai_types()
        target_scope = self._effective_scope(scope)
        facts = [
            fact
            for fact in self._kb.list_facts()
            if _scope_matches(self._fact_scope(fact), target_scope)
        ]
        categories = sorted({cat for fact in facts for cat in self._fact_categories(fact)})
        child_scopes = self.list_scopes(target_scope)
        oldest = min((fact.created_at for fact in facts), default=None)
        newest = max((fact.created_at for fact in facts), default=None)
        return ScopeInfo(
            path=target_scope,
            record_count=len(facts),
            categories=categories,
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=child_scopes,
        )

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        target_scope = self._effective_scope(scope_prefix)
        counts: Counter[str] = Counter()
        for fact in self._kb.list_facts():
            if not _scope_matches(self._fact_scope(fact), target_scope):
                continue
            counts.update(self._fact_categories(fact))
        return dict(sorted(counts.items()))

    def count(self, scope_prefix: str | None = None) -> int:
        target_scope = self._effective_scope(scope_prefix)
        return sum(
            1
            for fact in self._kb.list_facts()
            if _scope_matches(self._fact_scope(fact), target_scope)
        )

    def tree(self, path: str = "/", max_depth: int = 3) -> str:
        root = self._effective_scope(path)
        lines = [f"{root} ({self.count(root)} records)"]

        def walk(parent: str, depth: int) -> None:
            if depth >= max_depth:
                return
            for child in self.list_scopes(parent):
                indent = "  " * (depth + 1)
                lines.append(f"{indent}{child} ({self.count(child)} records)")
                walk(child, depth + 1)

        walk(root, 0)
        return "\n".join(lines)

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is None or self._effective_scope(scope_prefix) == "/":
            self._kb.clear_all()
            return
        self.forget(scope=scope_prefix)

    def reset_all(self) -> None:
        self._kb.clear_all()

    def scope(self, path: str) -> Any:
        scope_cls = _crewai_memory_scope_cls()
        normalised = _normalise_scope(path)
        if scope_cls is None:
            return _FallbackMemoryScope(self, normalised)
        return scope_cls(memory=self, root_path=normalised)
