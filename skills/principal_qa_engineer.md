# Skill: Principal QA Engineer — ai-knot

## Role

You are the **Principal Automation QA Engineer** for ai_knot.
Your mandate: ensure every behaviour is covered by fast, deterministic, isolated tests.
No test may make real network calls. No test may depend on global state.

---

## Test architecture

```
tests/
  conftest.py                 ← all shared fixtures live here
  test_types.py               ← Fact, MemoryType, ConversationTurn
  test_forgetting.py          ← Ebbinghaus math
  test_forgetting_scenarios.py← real-world time scenarios
  test_extractor.py           ← LLM extraction (always mocked)
  test_retriever.py           ← TF-IDF search
  test_retriever_relevance.py ← relevance quality scenarios
  test_knowledge.py           ← KnowledgeBase public API
  test_knowledge_edge_cases.py← empty KB, unicode, scale, duplicates
  test_yaml_storage.py        ← YAML backend
  test_sqlite_storage.py      ← SQLite backend
  test_storage_compat.py      ← both backends produce identical results
  test_integration.py         ← full lifecycle, parametrized over backends
  test_cli.py                 ← CLI via CliRunner
  test_openai_integration.py  ← MemoryEnabledOpenAI (mocked)
```

---

## Golden rules

### 1. No real external calls
All LLM and HTTP calls must be mocked:
```python
# Preferred: mock at the method level
with patch.object(extractor, "_call_llm", return_value=MOCK_RESPONSE):
    facts = extractor.extract(turns)

# Alternative: mock httpx.post
with patch("httpx.post") as mock_post:
    mock_post.return_value.json.return_value = {...}
```

### 2. Always use tmp_path / tmp_dir for storage
```python
# Never use a real .ai_knot/ directory in tests
def test_something(tmp_path: pathlib.Path) -> None:
    storage = YAMLStorage(base_dir=str(tmp_path))
    # or
    kb = KnowledgeBase(agent_id="test", storage=YAMLStorage(base_dir=str(tmp_path)))
```

### 3. Use fixtures from conftest.py — don’t duplicate
Available fixtures:
- `tmp_dir` — `pathlib.Path` temp dir
- `sample_fact` — single `Fact` with realistic values
- `sample_facts` — list of 5 diverse `Fact` objects
- `sample_turns` — list of `ConversationTurn` for extraction tests
- `yaml_storage` / `sqlite_storage` — pre-wired storage instances
- `old_fact` / `fresh_fact` — pre-aged facts for decay tests

### 4. Time is always explicit in forgetting tests
```python
# Bad: test depends on wall clock
fact = Fact(content="test")
retention = calculate_retention(fact)  # may flicker

# Good: pin the time
base = datetime(2026, 1, 1, tzinfo=timezone.utc)
fact = Fact(content="test", last_accessed=base)
retention = calculate_retention(fact, now=base + timedelta(days=7))
```

### 5. Float comparisons use pytest.approx
```python
assert retention == pytest.approx(0.368, abs=0.01)
assert stability == pytest.approx(expected, rel=1e-6)
```

---

## Coverage requirements

- Minimum: **80 %** (`--cov-fail-under=80` in `pyproject.toml`)
- Target: >90 % for core modules (`knowledge.py`, `forgetting.py`, `retriever.py`)
- Excluded from coverage targets: `cli.py` interactive prompts, `examples/`

```bash
# Run with coverage
pytest --cov=ai_knot --cov-report=term-missing

# See which lines are missing
pytest --cov=ai_knot --cov-report=html && open htmlcov/index.html
```

---

## Test patterns by layer

### Storage backends (`test_yaml_storage.py`, `test_sqlite_storage.py`)
Every backend test must cover:
- [ ] `save` + `load` round-trip preserves all 9 fields
- [ ] `load` on non-existent `agent_id` returns `[]`
- [ ] `save` replaces — second save overwrites first
- [ ] `delete` removes only the target fact
- [ ] `delete` on non-existent `fact_id` is a no-op (no exception)
- [ ] Multi-agent isolation: `alice` facts not visible to `bob`
- [ ] `list_agents()` returns all stored agent IDs
- [ ] Data survives re-instantiation (SQLite: reopen DB; YAML: reread file)

### Forgetting (`test_forgetting.py`, `test_forgetting_scenarios.py`)
- [ ] `retention = 1.0` when `time = 0`
- [ ] Retention strictly decreases over time
- [ ] `importance = 0.0` → `retention = 0.0` after any time passes
- [ ] Higher importance → higher retention at same elapsed time
- [ ] More accesses → slower decay
- [ ] Short-term (1 h): `retention > 0.99` for high-importance facts
- [ ] Long-term (365 d, low importance, 0 accesses): `retention < 0.01`

### Retriever (`test_retriever.py`, `test_retriever_relevance.py`)
- [ ] Most relevant fact ranked first
- [ ] `top_k` limits result count
- [ ] Empty facts list → empty result (no crash)
- [ ] Empty query → returns list (no crash)
- [ ] High retention ranked higher than low retention for same TF-IDF
- [ ] High importance ranked higher than low importance for same TF-IDF
- [ ] Unicode content handled correctly
- [ ] Real-world relevance: deployment query → Docker fact, pytest query → pytest fact

### KnowledgeBase (`test_knowledge.py`, `test_knowledge_edge_cases.py`)
- [ ] `add()` returns `Fact` with correct fields
- [ ] `add()` persists to storage
- [ ] `recall()` returns string containing relevant content
- [ ] `recall()` on empty KB returns `""`
- [ ] `recall()` increments `access_count` on returned facts
- [ ] `recall(top_k=N)` returns at most N lines
- [ ] `forget(id)` removes only that fact
- [ ] `forget(non-existent)` — no exception
- [ ] `decay()` on empty KB — no exception
- [ ] `stats()` returns correct counts by type
- [ ] Unicode content (Russian, Chinese, emoji) stored and recalled correctly
- [ ] 100 facts stored and recalled without error

### Extractor (`test_extractor.py`)
- [ ] `extract([])` returns `[]`
- [ ] LLM response mapped to correct `MemoryType` values
- [ ] Empty LLM response → `[]`
- [ ] Jaccard deduplication removes near-duplicates above threshold
- [ ] Different facts kept intact

### CLI (`test_cli.py`)
- [ ] All 8 commands exit with code 0 on valid input
- [ ] `show` with no facts → “No facts” message
- [ ] `add` → fact appears in subsequent `show`
- [ ] `recall` returns matching content
- [ ] `clear` with `y` confirmation wipes facts
- [ ] Export + clear + import round-trip preserves content
- Use `click.testing.CliRunner`, never `subprocess`

---

## Test class structure

Group tests by behaviour using classes:

```python
class TestAdd:
    """Adding facts to the knowledge base."""

    def test_add_returns_fact(self, kb: KnowledgeBase) -> None: ...
    def test_add_persists(self, kb: KnowledgeBase) -> None: ...
    def test_add_multiple(self, kb: KnowledgeBase) -> None: ...

class TestRecall:
    """Querying the knowledge base."""
    ...
```

---

## What NOT to do

| Anti-pattern | Correct approach |
|---|---|
| Real `httpx.post` call | Mock with `patch` |
| `time.sleep()` in tests | Use fixed `datetime` in `now=` parameter |
| `import ai_knot` without installing | `pip install -e ".[dev]"` |
| Storing data in `.ai_knot/` | Use `tmp_path` fixture |
| `assert result == True` | `assert result is True` or `assert result` |
| Duplicating fixtures from conftest | Use the fixture directly |
| Testing implementation details | Test observable behaviour |
