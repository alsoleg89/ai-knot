# ai-knot Raw-First V2 — Project Impact Audit and Copy-Paste Skeleton

**Date:** 2026-04-13  
**Purpose:** separate implementation-facing audit of everything else in the project that must change once the raw-first V2 core lands  
**Companion docs:**
- [raw_first_plan_v2_20260413.md](/Users/alsoleg/Documents/github/ai-knot/research/raw_first_plan_v2_20260413.md)
- [raw_first_memory_architecture_20260413.md](/Users/alsoleg/Documents/github/ai-knot/research/raw_first_memory_architecture_20260413.md)

---

## 1. Why This Separate File Exists

The main V2 plan covers the core memory architecture:

- raw episodes
- atomic claims
- helper plane
- deterministic raw operators

But if we implement only the core, the rest of the project will drift out of sync.

This file covers the **real blast radius**:

- public Python API
- CLI
- MCP tools
- npm/TypeScript wrapper
- integrations
- docs/examples
- snapshots/export/import
- benchmark backends
- tests
- release/migration concerns

This is the list of project surfaces that engineers usually forget until the last week.

---

## 2. High-Level Blast Radius

| Area | Must change now | Can wait one release | Why |
|---|---|---|---|
| Core schema and storage | Yes | No | V2 cannot exist without it |
| `KnowledgeBase` public API | Yes | No | raw-first mode must be first-class |
| `stats()` semantics | Yes | No | Wave 1 otherwise reports misleading memory totals |
| CLI | Yes | No | restore/rebuild/export semantics change |
| MCP tools | Yes | No | external consumers need structured raw access |
| npm wrapper | Yes | No | TS clients currently see only old `Fact` model |
| OpenAI/OpenClaw integrations | Yes | Maybe | they should expose raw-first value, not only old recall |
| README/examples | Yes | No | current docs still describe facts-only memory |
| snapshots/export/import | Yes | No | now bundle-based, not facts-only |
| existing unit tests | Yes | No | many assertions assume facts-only world |
| benchmark regression guard | Yes | No | storage/query-surface changes can silently regress current recall |
| benchmark backend raw-mode path | Should | Yes | only needed if we want to measure raw mode directly |
| multi-agent pool | No | Yes | should be treated as follow-up track |
| npm versioning strategy | No | Yes | additive API can ship later, but rollout must be explicit |

---

## 3. What Else in the Project Will Need to Change

### 3.1 Public Python API

#### Files

- [src/ai_knot/__init__.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/__init__.py)
- [src/ai_knot/knowledge.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/knowledge.py)

#### What changes

- export new dataclasses:
  - `Episode`
  - `RawAnswer`
  - `RawAnswerItem`
  - `RawTrace`
- add new `KnowledgeBase` methods:
  - `raw_query()`
  - `raw_query_with_trace()`
  - `list_episodes()`
  - `rebuild_materialized()`
  - `export_bundle()`
  - `import_bundle()`
- keep existing methods stable:
  - `recall()`
  - `recall_facts()`
  - `recall_facts_with_scores()`

#### Compatibility strategy

- do not break `recall()`
- do not rename `Fact`
- do not remove existing `snapshot()/restore()`
- add new bundle-aware capabilities behind compatible methods

#### Copy-paste skeleton

```python
from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.raw_query import RawAnswer, RawAnswerItem, RawTrace
from ai_knot.types import Episode, Fact, MemoryType, MemoryOp, ConversationTurn

__all__ = [
    "ConversationTurn",
    "Episode",
    "Fact",
    "KnowledgeBase",
    "MemoryOp",
    "MemoryType",
    "RawAnswer",
    "RawAnswerItem",
    "RawTrace",
    "SharedMemoryPool",
]
```

---

### 3.2 `KnowledgeBase` Surface

#### File

[src/ai_knot/knowledge.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/knowledge.py)

#### What changes beyond the core memory logic

- `list_facts()` semantics must be clarified:
  - return claims only
  - or optionally include helper claims
- add `list_episodes()`
- add bundle-aware `replace_bundle()` or separate `replace_facts()` plus `replace_episodes()`
- `stats()` must define whether helper claims count toward totals
- `snapshot()` and `restore()` become bundle-aware under the hood
- `clear_all()` must clear facts and episodes

#### Critical semantic conflict

`KnowledgeBase._execute_recall()` currently filters out `MemoryType.EPISODIC` by default.

Under V2 this is no longer a minor hidden dependency. It becomes a semantic conflict:

- `raw_query()` must be able to reason over dated/raw memory
- `recall()` should still remain conservative for prompt injection
- both must read from one bundle-backed substrate without splitting memory visibility into two incompatible modes

#### Explicit V2 resolution

Recommended rule:

- keep `recall()` conservative by default
- make `_execute_recall()` accept explicit visibility knobs:
  - `include_episodic`
  - `record_kinds`
  - `claim_kinds`
- let `raw_query()` use deliberate raw-aware retrieval, not hidden side effects

A lot of code and tests currently assume:

- `list_facts()` is the full memory
- snapshots contain only facts
- `replace_facts()` is enough for import/restore

That assumption becomes false under V2.

#### Copy-paste skeleton

```python
class KnowledgeBase(_LearningMixin):
    def raw_query(
        self,
        query: str,
        *,
        now: datetime | None = None,
        include_evidence: bool = True,
    ) -> RawAnswer:
        ...

    def raw_query_with_trace(
        self,
        query: str,
        *,
        now: datetime | None = None,
    ) -> tuple[RawAnswer, RawTrace]:
        ...

    def list_episodes(self) -> list[Episode]:
        bundle = self._storage.load_bundle(self._agent_id)
        return list(bundle["episodes"])

    def export_bundle(self) -> dict[str, object]:
        return self._storage.load_bundle(self._agent_id)

    def import_bundle(self, bundle: dict[str, object]) -> None:
        self._storage.save_bundle(
            self._agent_id,
            episodes=bundle["episodes"],
            facts=bundle["facts"],
            schema_version=bundle["schema_version"],
            materializer_version=bundle["materializer_version"],
        )
```

#### Copy-paste skeleton for the recall conflict

```python
def _execute_recall(
    self,
    query: str,
    *,
    top_k: int,
    now: datetime | None = None,
    include_episodic: bool = False,
    record_kinds: set[RecordKind] | None = None,
    claim_kinds: set[ClaimKind] | None = None,
    include_unsupported: bool = False,
) -> list[tuple[Fact, float]]:
    ...


def raw_query(self, query: str, *, now: datetime | None = None) -> RawAnswer:
    claims = self._execute_recall(
        query,
        top_k=50,
        now=now,
        include_episodic=True,
        record_kinds={RecordKind.ATOMIC, RecordKind.HELPER},
    )
    ...
```

---

### 3.3 CLI

#### File

[src/ai_knot/cli.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/cli.py)

#### What changes

Current CLI is facts-only. That becomes misleading.

Need new commands:

- `raw-query`
- `show-episodes`
- `rebuild`
- `export-bundle`
- `import-bundle`

Current commands that must change internally:

- `show`
- `stats`
- `export`
- `import`
- `clear`
- `decay`

#### Minimum CLI policy

- keep old `recall` command as-is
- add `raw-query` instead of changing `recall`
- keep old `export` only for legacy facts export
- add `export-bundle` as the new recommended path

#### Copy-paste skeleton

```python
@main.command("raw-query")
@click.argument("agent_id")
@click.argument("query")
@click.option("--json-output", is_flag=True, default=False)
@click.pass_context
def raw_query_cmd(ctx: click.Context, agent_id: str, query: str, json_output: bool) -> None:
    kb = _make_kb(ctx, agent_id)
    answer = kb.raw_query(query)
    if json_output:
        click.echo(json.dumps(dataclasses.asdict(answer), ensure_ascii=False, indent=2))
        return
    for item in answer.items:
        click.echo(f"{item.label}: {item.value}")


@main.command("show-episodes")
@click.argument("agent_id")
@click.pass_context
def show_episodes(ctx: click.Context, agent_id: str) -> None:
    kb = _make_kb(ctx, agent_id)
    for ep in kb.list_episodes():
        click.echo(f"[{ep.observed_at.isoformat()}] ({ep.role}) {ep.raw_text}")


@main.command("rebuild")
@click.argument("agent_id")
@click.pass_context
def rebuild(ctx: click.Context, agent_id: str) -> None:
    kb = _make_kb(ctx, agent_id)
    rebuilt = kb.rebuild_materialized()
    click.echo(f"Rebuilt {len(rebuilt)} helper claims.")
```

---

### 3.4 MCP Tools

#### Files

- [src/ai_knot/_mcp_tools.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/_mcp_tools.py)
- [src/ai_knot/mcp_server.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/mcp_server.py)

#### What changes

Current MCP surface exposes:

- `recall`
- `recall_json`
- `list_facts`
- `snapshot`
- `restore`

That is not enough for raw-first V2.

Need additional tools:

- `raw_query_json`
- `list_episodes`
- `rebuild_materialized`
- maybe `export_bundle_json`

#### Compatibility policy

- keep `recall` and `recall_json` unchanged
- do not overload `recall_json` with raw-answer semantics
- expose raw mode as a new tool, not as a breaking change

#### Copy-paste skeleton: `_mcp_tools.py`

```python
def tool_raw_query_json(kb: KnowledgeBase, query: str) -> str:
    answer = kb.raw_query(query)
    return json.dumps(dataclasses.asdict(answer), ensure_ascii=False)


def tool_list_episodes(kb: KnowledgeBase) -> str:
    episodes = kb.list_episodes()
    data = [
        {
            "episode_id": ep.episode_id,
            "turn_id": ep.turn_id,
            "role": ep.role,
            "observed_at": ep.observed_at.isoformat(),
            "raw_text": ep.raw_text,
        }
        for ep in episodes
    ]
    return json.dumps(data, ensure_ascii=False)


def tool_rebuild_materialized(kb: KnowledgeBase) -> str:
    rebuilt = kb.rebuild_materialized()
    return json.dumps({"rebuilt": len(rebuilt)}, ensure_ascii=False)
```

#### Copy-paste skeleton: `mcp_server.py`

```python
from ai_knot._mcp_tools import (
    tool_raw_query_json,
    tool_list_episodes,
    tool_rebuild_materialized,
)


@app.tool()
def raw_query_json(query: str) -> str:
    """Run deterministic raw-mode query and return structured JSON."""
    return tool_raw_query_json(kb, query)


@app.tool()
def list_episodes() -> str:
    """List preserved raw episodes as JSON."""
    return tool_list_episodes(kb)


@app.tool()
def rebuild_materialized() -> str:
    """Recompute helper/materialized layer from stored substrate."""
    return tool_rebuild_materialized(kb)
```

#### Tool capability update

Do not forget to update `tool_capabilities()` so external clients can discover the new tools.

---

### 3.5 npm / TypeScript Wrapper

#### Files

- [npm/src/types.ts](/Users/alsoleg/Documents/github/ai-knot/npm/src/types.ts)
- [npm/src/index.ts](/Users/alsoleg/Documents/github/ai-knot/npm/src/index.ts)

#### What changes

The current TS wrapper only knows old `Fact`, `Stats`, `recall()`, `learn()`, `snapshot()`, `restore()`.

Need new TS types:

- `Episode`
- `RawAnswer`
- `RawAnswerItem`
- `RawTrace`

Need new methods:

- `rawQuery()`
- `listEpisodes()`
- `rebuildMaterialized()`

#### Compatibility policy

- do not break `recall()`
- add raw mode as additive API
- keep `Fact` as the old structured memory item

#### Versioning decision

Because the TS surface remains additive if current methods stay stable, V2 does not require an npm major bump.

Recommended rollout:

- publish prerelease first under `@next`
- validate MCP capability discovery and JSON payload shapes
- then promote to the normal minor release channel once stable

#### Copy-paste skeleton: `npm/src/types.ts`

```typescript
export interface Episode {
  episode_id: string;
  turn_id: string;
  role: string;
  observed_at: string;
  raw_text: string;
}

export interface RawAnswerItem {
  label: string;
  value: string;
  claim_ids: string[];
  episode_ids: string[];
}

export interface RawTrace {
  query_shape: string;
  operator_used: string;
  source_claim_ids: string[];
  source_episode_ids: string[];
  helper_ids: string[];
}

export interface RawAnswer {
  status: string;
  items: RawAnswerItem[];
  confidence: number;
  trace?: RawTrace;
}
```

#### Copy-paste skeleton: `npm/src/index.ts`

```typescript
async rawQuery(query: string): Promise<RawAnswer> {
  const text = await this.client.call("raw_query_json", { query });
  return JSON.parse(text) as RawAnswer;
}

async listEpisodes(): Promise<Episode[]> {
  const text = await this.client.call("list_episodes", {});
  return JSON.parse(text) as Episode[];
}

async rebuildMaterialized(): Promise<{ rebuilt: number }> {
  const text = await this.client.call("rebuild_materialized", {});
  return JSON.parse(text) as { rebuilt: number };
}
```

---

### 3.6 Integrations

#### Files

- [src/ai_knot/integrations/openai.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/integrations/openai.py)
- [src/ai_knot/integrations/openclaw.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/integrations/openclaw.py)

#### What changes

Current integrations assume “memory = prompt context string”.

After V2 there are now two valid consumption patterns:

- `semantic` style: inject `recall()` text into a prompt
- `raw-first` style: use `raw_query()` programmatically and only render if needed

#### Recommendation

Do not force existing integrations to switch immediately.

Instead:

- keep `enrich_messages()` based on `recall()`
- optionally add `build_raw_memory_block()` or `get_raw_answer()`

#### Copy-paste skeleton: `openai.py`

```python
class MemoryEnabledOpenAI:
    def get_raw_answer(self, query: str) -> RawAnswer:
        return self._kb.raw_query(query)

    def enrich_messages_with_raw(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], RawAnswer | None]:
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return messages, None
        query = user_messages[-1]["content"]
        raw = self._kb.raw_query(query)
        if not raw.items:
            return messages, raw

        memory_lines = [f"{item.label}: {item.value}" for item in raw.items]
        block = "## Agent Memory\n" + "\n".join(memory_lines)
        enriched = copy.deepcopy(messages)
        enriched.insert(0, {"role": "system", "content": block})
        return enriched, raw
```

#### Hidden issue

`OpenClawMemoryAdapter` and similar adapters may need a new “structured raw answer” path, not just `search()` returning text hits.

---

### 3.7 README, Examples, and Product Docs

#### Files

- [README.md](/Users/alsoleg/Documents/github/ai-knot/README.md)
- [examples/quickstart.py](/Users/alsoleg/Documents/github/ai-knot/examples/quickstart.py)
- [examples/openai_integration.py](/Users/alsoleg/Documents/github/ai-knot/examples/openai_integration.py)
- [examples/shared_pool.py](/Users/alsoleg/Documents/github/ai-knot/examples/shared_pool.py)
- [examples/multilingual.py](/Users/alsoleg/Documents/github/ai-knot/examples/multilingual.py)

#### What changes

The README currently says:

- “knowledge base instead of log”
- YAML store is `knowledge.yaml`
- snapshots save “facts”

That wording becomes incomplete after V2.

#### Required documentation changes

- explain that raw episodes are preserved as substrate
- show `raw_query()` example
- update YAML-on-disk section to bundle format
- explain that snapshots are bundle snapshots
- explain difference between:
  - `recall()`
  - `raw_query()`
  - `semantic mode`

#### Example snippet for README

```python
from ai_knot import KnowledgeBase, ConversationTurn

kb = KnowledgeBase(agent_id="demo")
kb.learn([
    ConversationTurn(role="user", content="It took six months to open the studio"),
    ConversationTurn(role="assistant", content="Got it"),
], provider="openai")

answer = kb.raw_query("How long did it take to open the studio?")
for item in answer.items:
    print(item.label, item.value)
# duration 6 months
```

---

### 3.8 Export / Import / Snapshot Semantics

#### Files

- [src/ai_knot/cli.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/cli.py)
- [src/ai_knot/storage/sqlite_storage.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/storage/sqlite_storage.py)
- [src/ai_knot/storage/yaml_storage.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/storage/yaml_storage.py)
- [src/ai_knot/knowledge.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/knowledge.py)

#### What changes

Current export/import is facts-only.

Under V2 there are now three distinct operations:

- `legacy fact export`
- `bundle export`
- `snapshot restore + rebuild`

#### Required decision

Keep old commands for backward compatibility, but make bundle export the recommended default in docs.

#### Copy-paste skeleton

```python
def export_bundle_cmd(ctx: click.Context, agent_id: str, output: str) -> None:
    kb = _make_kb(ctx, agent_id)
    bundle = kb.export_bundle()
    Path(output).write_text(
        yaml.dump(bundle, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def import_bundle_cmd(ctx: click.Context, agent_id: str, input_file: str) -> None:
    raw = yaml.safe_load(Path(input_file).read_text(encoding="utf-8"))
    kb = _make_kb(ctx, agent_id)
    kb.import_bundle(raw)
```

#### `replace_facts()` must fail loudly

Facts-only replacement becomes unsafe once bundle memory exists.

Recommended rule:

- if the bundle has no episodes and no helper claims, allow legacy `replace_facts()`
- otherwise raise and redirect callers to:
  - `replace_bundle()`
  - `import_bundle()`
  - `restore()`

#### Copy-paste skeleton

```python
def replace_facts(self, facts: list[Fact]) -> None:
    bundle = self._storage.load_bundle(self._agent_id)
    episodes: list[Episode] = bundle["episodes"]
    existing: list[Fact] = bundle["facts"]
    has_bundle_state = bool(episodes) or any(
        getattr(f, "record_kind", "atomic") != "atomic" for f in existing
    )
    if has_bundle_state:
        raise RuntimeError(
            "replace_facts() is unsafe for bundle-backed memory; "
            "use replace_bundle(), import_bundle(), or restore()."
        )
    self._storage.save_bundle(
        self._agent_id,
        episodes=[],
        facts=facts,
        schema_version="v2",
        materializer_version="v2",
    )
```

---

### 3.9 Stats and Observability

#### Files

- [src/ai_knot/knowledge.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/knowledge.py)
- [src/ai_knot/_mcp_tools.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/_mcp_tools.py)

#### What changes

`stats()` currently reports:

- total facts
- by type
- avg importance
- avg retention

Under V2 this becomes ambiguous.

#### Recommended new stats

- `total_atomic_claims`
- `total_helper_claims`
- `total_episodes`
- `active_current_state_claims`
- `avg_retention_atomic`
- `avg_retention_helper`
- `materializer_version`

#### Priority correction

This is a **Wave 1** task, not a later product-surface cleanup.

Reason:

- bundle-backed storage changes the meaning of totals immediately
- waiting until CLI/MCP refresh leaves Python users with misleading metrics during the core rollout

#### Copy-paste skeleton

```python
def stats(self) -> dict[str, object]:
    bundle = self._storage.load_bundle(self._agent_id)
    facts: list[Fact] = bundle["facts"]
    episodes: list[Episode] = bundle["episodes"]

    atomic = [f for f in facts if getattr(f, "record_kind", "atomic") == "atomic"]
    helper = [f for f in facts if getattr(f, "record_kind", "atomic") == "helper"]

    return {
        "total_facts": len(facts),
        "total_atomic_claims": len(atomic),
        "total_helper_claims": len(helper),
        "total_episodes": len(episodes),
        "avg_importance": sum(f.importance for f in atomic) / max(len(atomic), 1),
        "avg_retention": sum(f.retention_score for f in atomic) / max(len(atomic), 1),
        "materializer_version": bundle.get("materializer_version", "unknown"),
    }
```

---

### 3.10 Benchmarks and Eval Backends

#### Files

- [tests/eval/benchmark/backends/ai_knot_backend.py](/Users/alsoleg/Documents/github/ai-knot/tests/eval/benchmark/backends/ai_knot_backend.py)
- [tests/eval/benchmark/backends/ai_knot_multi_agent_backend.py](/Users/alsoleg/Documents/github/ai-knot/tests/eval/benchmark/backends/ai_knot_multi_agent_backend.py)
- [aiknotbench/src/aiknot.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/aiknot.ts)
- [aiknotbench/src/runner.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/runner.ts)
- [aiknotbench/src/evaluator.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/evaluator.ts)

#### Important point

The benchmark backend currently evaluates retrieved text items from `recall_facts_with_scores()`.

That means:

- product-level `raw_query()` will not change those benchmark numbers automatically
- the benchmark will only move if retrieved claims/evidence improve, or if we add a dedicated raw-mode backend

#### Important point for `aiknotbench`

`aiknotbench` currently also measures the old `recall()` surface, not the future `raw_query()` surface.

Current pipeline:

- [aiknotbench/src/runner.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/runner.ts) takes `context = await adapter.recall(qa.question)`
- [aiknotbench/src/aiknot.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/aiknot.ts) implements `AiknotAdapter.recall()` via `kb.recall(...)`
- [aiknotbench/src/evaluator.ts](/Users/alsoleg/Documents/github/ai-knot/aiknotbench/src/evaluator.ts) then sends that context to a separate answer model, and a judge compares the generated answer with gold

That means:

- `raw_query()` product gains will not become visible in `aiknotbench` automatically
- if we want LoCoMo numbers in `aiknotbench` to reflect raw-mode progress, we must update the benchmark code path
- this can be done by adding a raw query mode, a hybrid mode, or a dedicated adapter that renders `raw_query()` output into the benchmark answer/judge loop

#### Recommendation

Do not force this change into the first V2 implementation.

Instead:

- keep current backend stable
- later add an optional `AiKnotRawBackend` or `use_raw_query=True`
- treat `aiknotbench` the same way: keep the current `recall()` path as the historical baseline, then add an explicit raw/hybrid benchmark mode rather than silently changing the existing metric path

#### Non-regression requirement

Even if we postpone a dedicated raw-mode backend, every implementation wave must preserve visibility into current regressions.

Required guard:

- run the existing benchmark smoke path before each merge
- compare against the current baseline
- reject changes that materially regress the existing `recall()` surface

This matters most for:

- Wave 1 storage migrations
- Wave 2 retrieval visibility changes
- changes touching `list_facts()`, `replace_facts()`, `snapshot()`, or `restore()`

#### Copy-paste skeleton

```python
class AiKnotRawBackend(AiKnotBackend):
    async def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        assert self._kb is not None
        answer = self._kb.raw_query(query)
        texts = [f"{item.label}: {item.value}" for item in answer.items][:top_k]
        return RetrievalResult(
            texts=texts,
            scores=[answer.confidence for _ in texts],
            retrieve_ms=0.0,
            evidence_texts=[",".join(item.episode_ids) for item in answer.items[:top_k]],
        )
```

---

### 3.11 Tests That Will Break or Need Reinterpretation

#### What will change semantically

- tests assuming facts-only snapshots
- tests assuming YAML on disk is only `knowledge.yaml`
- tests assuming `list_facts()` reflects the entire memory state
- tests assuming `stats()["total_facts"]` means all memory objects
- tests assuming episodic data is always excluded from retrieval

#### Areas to audit

- [tests/test_types.py](/Users/alsoleg/Documents/github/ai-knot/tests/test_types.py)
- [tests/test_examples.py](/Users/alsoleg/Documents/github/ai-knot/tests/test_examples.py)
- [tests/test_openai_integration.py](/Users/alsoleg/Documents/github/ai-knot/tests/test_openai_integration.py)
- benchmark scenario tests touching snapshots, decay, consolidation, throughput

#### Recommendation

Split test migration into:

1. compatibility tests
2. V2 behavior tests
3. performance tests

Do not try to “fix tests one by one” without deciding new semantics first.

---

### 3.12 Multi-Agent Follow-up Blast Radius

#### Files that will be touched later, not in first single-agent V2 release

- [src/ai_knot/pool.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/pool.py)
- [src/ai_knot/_pool_recall.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/_pool_recall.py)
- [src/ai_knot/multi_agent/models.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/multi_agent/models.py)
- [src/ai_knot/multi_agent/recall_service.py](/Users/alsoleg/Documents/github/ai-knot/src/ai_knot/multi_agent/recall_service.py)
- [tests/eval/benchmark/backends/ai_knot_multi_agent_backend.py](/Users/alsoleg/Documents/github/ai-knot/tests/eval/benchmark/backends/ai_knot_multi_agent_backend.py)

#### What will change later

- pool publishes atomic claims, not raw episodes
- trust applies to claims and supersession quality
- helper deltas may be shared selectively
- pool decay differs by tier
- evidence sharing must obey privacy boundaries

---

## 4. Hidden Things That Will Definitely Be Forgotten If We Do Not Write Them Down

### 4.1 `__version__` and release messaging

V2 is big enough that release semantics matter.

If we keep old APIs stable:

- minor/feature release is possible

If we change export/import or snapshot formats incompatibly:

- this is effectively a major release

### 4.2 `knowledge.yaml` assumptions

README and tooling currently imply that the on-disk representation is:

- simple
- human-editable
- single-file

Under V2:

- it may become `bundle.yaml` or `knowledge.yaml + episodes.yaml`
- documentation must say that helper plane may be regenerated

### 4.3 Current filtering of episodic facts

`KnowledgeBase._execute_recall()` currently filters out `MemoryType.EPISODIC` by default.

That will conflict with raw-first memory unless carefully reworked.

This is not a docs issue. It is a semantic issue that touches:

- retrieval behavior
- benchmark behavior
- examples
- tests

#### Status

This is promoted from “hidden gotcha” to an explicit Wave 1 task.

### 4.4 `replace_facts()` is no longer enough

Any code path that does:

- `kb.list_facts()`
- mutate list
- `kb.replace_facts(...)`

is now incomplete if episodes/helper state exist.

That affects:

- tests
- benchmark backends
- CLI import/export
- decay simulation helpers

#### Status

This is also promoted to an explicit Wave 1 task:

- deprecate facts-only replacement as a general restore primitive
- fail loudly when bundle state exists
- provide a bundle-safe replacement path

### 4.5 `tool_list_facts()` and `recall_json()`

External users will start to assume these tools represent the whole memory model.

If V2 ships without:

- `list_episodes`
- `raw_query_json`

then the product will look internally more advanced than its public interface actually is.

---

## 5. Recommended Compatibility Strategy

### Keep stable

- `kb.recall()`
- `kb.recall_facts()`
- `kb.recall_facts_with_scores()`
- `kb.learn()`
- `tool_recall`
- `tool_recall_json`
- `Fact` name

### Add

- `kb.raw_query()`
- `kb.list_episodes()`
- `kb.rebuild_materialized()`
- `tool_raw_query_json`
- `tool_list_episodes`
- `tool_rebuild_materialized`
- bundle-aware stats fields

### Deprecate later, not now

- facts-only export/import as the preferred path
- any docs claiming memory is only “facts on disk”
- `replace_facts()` as a general-purpose import/restore primitive

### Why this matters

If we preserve old APIs and add raw-first APIs beside them, V2 is much easier to ship incrementally.

---

## 6. Suggested File Plan for the Implementation Branch

### New files

- `src/ai_knot/materialization.py`
- `src/ai_knot/raw_query.py`
- `src/ai_knot/evidence.py`
- `tests/test_raw_query.py`
- `tests/test_materialization.py`
- `tests/test_bundle_storage.py`
- `tests/test_rebuild_materialized.py`

### Modified core files

- `src/ai_knot/types.py`
- `src/ai_knot/knowledge.py`
- `src/ai_knot/learning.py`
- `src/ai_knot/forgetting.py`
- `src/ai_knot/storage/base.py`
- `src/ai_knot/storage/sqlite_storage.py`
- `src/ai_knot/storage/yaml_storage.py`
- `src/ai_knot/storage/postgres_storage.py`

### Modified product/API files

- `src/ai_knot/__init__.py`
- `src/ai_knot/_mcp_tools.py`
- `src/ai_knot/mcp_server.py`
- `src/ai_knot/cli.py`
- `src/ai_knot/integrations/openai.py`
- `src/ai_knot/integrations/openclaw.py`
- `npm/src/types.ts`
- `npm/src/index.ts`

### Modified docs/examples

- `README.md`
- `examples/quickstart.py`
- `examples/openai_integration.py`
- `examples/multilingual.py`

### Possibly modified later for eval and multi-agent

- `tests/eval/benchmark/backends/ai_knot_backend.py`
- `tests/eval/benchmark/backends/ai_knot_multi_agent_backend.py`
- `aiknotbench/src/aiknot.ts`
- `aiknotbench/src/runner.ts`
- `aiknotbench/src/evaluator.ts`
- `src/ai_knot/pool.py`
- `src/ai_knot/_pool_recall.py`
- `src/ai_knot/multi_agent/*`

---

## 7. Practical Final Recommendation

Ship V2 in three waves:

### Wave 1

- schema/storage/bundle support
- raw episodes
- backward compatibility
- explicit resolution of the `EPISODIC` visibility conflict
- fail-loud `replace_facts()` behavior
- `stats()` semantic update
- benchmark smoke non-regression gate

### Wave 2

- materializers
- raw query operators
- rebuild path

### Wave 3

- CLI/MCP/npm/docs refresh
- integration updates
- dedicated raw-mode benchmark path if needed
- `aiknotbench` raw/hybrid query mode if we want LoCoMo runs to measure `raw_query()` rather than only `recall()`
- npm `@next` to stable promotion

This keeps the project coherent while avoiding the classic failure mode:

**beautiful core refactor, broken product surface.**
