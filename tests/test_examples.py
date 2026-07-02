"""Tests for README examples — run without LLM API keys."""

import importlib
import json
import logging
import pathlib

import pytest

from ai_knot import KnowledgeBase, MemoryType
from ai_knot.integrations.crewai import AiKnotCrewAIMemory
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter, generate_mcp_config
from ai_knot.storage import SQLiteStorage, YAMLStorage, create_storage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="test", storage=YAMLStorage(base_dir=str(tmp_path)))


def test_example1_manual_add_recall(kb: KnowledgeBase) -> None:
    kb.add("User prefers Python", type=MemoryType.PROCEDURAL, importance=0.9)
    kb.add("User deploys with Docker", importance=0.85)
    kb.add("Deploy failed last Tuesday", type=MemoryType.EPISODIC, importance=0.4)

    context = kb.recall("how to deploy?")
    assert isinstance(context, str)
    assert "Docker" in context


def test_example2_sqlite_init(tmp_path: pathlib.Path) -> None:
    storage = SQLiteStorage(db_path=str(tmp_path / "bot.db"))
    kb = KnowledgeBase(agent_id="bot", storage=storage)
    kb.add("User works with Python and FastAPI")
    context = kb.recall("what stack does user use?")
    assert "FastAPI" in context


def test_example3_yaml_init(tmp_path: pathlib.Path) -> None:
    storage = YAMLStorage(base_dir=str(tmp_path))
    kb = KnowledgeBase(agent_id="bot", storage=storage)
    kb.add("Always write tests with pytest", type=MemoryType.PROCEDURAL)
    context = kb.recall("testing preferences")
    assert "pytest" in context


@pytest.mark.parametrize("backend", ["yaml", "sqlite"])
def test_example4_create_storage_factory(tmp_path: pathlib.Path, backend: str) -> None:
    storage = create_storage(backend, base_dir=str(tmp_path))
    kb = KnowledgeBase(agent_id="assistant", storage=storage)
    kb.add("Prefer concise answers", type=MemoryType.PROCEDURAL)
    assert kb.stats()["total_facts"] == 1


def test_example5_per_customer(tmp_path: pathlib.Path) -> None:
    def handle_ticket(customer_id: str, message: str) -> str:
        kb = KnowledgeBase(
            agent_id=f"customer_{customer_id}",
            storage=YAMLStorage(base_dir=str(tmp_path)),
        )
        return kb.recall(message)

    # Seed customer facts
    kb = KnowledgeBase(
        agent_id="customer_123",
        storage=YAMLStorage(base_dir=str(tmp_path)),
    )
    kb.add("Customer prefers email notifications")
    kb.add("Customer is on premium tier")

    result = handle_ticket("123", "notification preferences")
    assert isinstance(result, str)
    assert "email" in result.lower() or "notification" in result.lower()


def test_example6_project_context(tmp_path: pathlib.Path) -> None:
    kb = KnowledgeBase(agent_id="project", storage=YAMLStorage(str(tmp_path)))
    kb.add("Stack: FastAPI + PostgreSQL + Docker", importance=1.0)
    kb.add("No unittest — use pytest only", type=MemoryType.PROCEDURAL, importance=0.9)
    kb.add("All endpoints require JWT auth", importance=0.95)

    context = kb.recall("how should I write tests?")
    assert "pytest" in context


def test_example7_shared_knowledge(tmp_path: pathlib.Path) -> None:
    storage = SQLiteStorage(db_path=str(tmp_path / "team.db"))
    researcher = KnowledgeBase(agent_id="team_alpha", storage=storage)
    writer = KnowledgeBase(agent_id="team_alpha", storage=storage)

    researcher.add("API rate limit is 100 req/s")
    context = writer.recall("rate limits")
    assert "100" in context or "rate" in context.lower()


def test_example8_stats_and_decay(kb: KnowledgeBase) -> None:
    kb.add("User likes dark mode")
    kb.add("User timezone is UTC+3")

    stats = kb.stats()
    assert stats["total_facts"] == 2
    assert "avg_importance" in stats
    assert "avg_retention" in stats
    assert "by_type" in stats

    kb.decay()  # should not raise


def test_recall_does_not_need_provider(kb: KnowledgeBase) -> None:
    kb.add("User deploys everything in Docker")
    context = kb.recall("how should I deploy this?")
    assert "Docker" in context


def test_example9_persistence_across_restarts(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "hero-demo.db"
    writer = KnowledgeBase(agent_id="assistant", storage=SQLiteStorage(db_path=str(db_path)))
    writer.add("User prefers Python", type=MemoryType.PROCEDURAL)
    writer.add("User deploys with Docker and Kubernetes")

    reader = KnowledgeBase(agent_id="assistant", storage=SQLiteStorage(db_path=str(db_path)))
    context = reader.recall("what stack and preferences should I use?")
    assert "Python" in context
    assert "Docker" in context or "Kubernetes" in context


def test_example10_openclaw_config_and_adapter(tmp_path: pathlib.Path) -> None:
    config = generate_mcp_config(agent_id="bot", data_dir=str(tmp_path), storage="sqlite")
    env = config["mcpServers"]["ai-knot"]["env"]
    assert env["AI_KNOT_AGENT_ID"] == "bot"
    assert env["AI_KNOT_STORAGE"] == "sqlite"
    assert env["AI_KNOT_DATA_DIR"] == str(tmp_path.resolve())

    kb = KnowledgeBase(agent_id="bot", storage=SQLiteStorage(db_path=str(tmp_path / "bot.db")))
    memory = OpenClawMemoryAdapter(kb)
    created = memory.add([{"role": "user", "content": "Deploy with Docker Compose on Fridays"}])

    results = memory.search("deployment schedule")
    assert results
    assert "Docker Compose" in results[0]["memory"]
    assert memory.list()
    memory.forget(created["results"][0]["id"])
    assert memory.get_all() == []
    assert memory.get_all(include_inactive=True) == []


def test_example11_crewai_adapter_without_crewai(tmp_path: pathlib.Path) -> None:
    kb = KnowledgeBase(agent_id="crew-demo", storage=YAMLStorage(base_dir=str(tmp_path)))
    memory = AiKnotCrewAIMemory(kb, top_k=3)
    researcher = memory.scope("/agent/researcher")

    researcher.remember("Use PostgreSQL for billing", categories=["database"])

    results = memory.recall("postgresql", scope="/agent/researcher")
    assert results
    assert "PostgreSQL" in results[0].record.content


def test_example12_crewai_surface_demo_scopes(tmp_path: pathlib.Path) -> None:
    kb = KnowledgeBase(agent_id="crew-surface-demo", storage=YAMLStorage(base_dir=str(tmp_path)))
    memory = AiKnotCrewAIMemory(kb, top_k=3)
    researcher = memory.scope("/agent/researcher")
    writer = memory.scope("/agent/writer")

    researcher.remember("User deploys APIs with Docker and Kubernetes", categories=["ops"])
    researcher.remember("Primary database is PostgreSQL", categories=["database"])
    writer.remember("Release notes should stay concise and numbered", categories=["docs"])

    research_results = researcher.recall("database and deployment")
    writer_results = writer.recall("release notes")

    assert memory.list_scopes("/") == ["/agent"]
    assert memory.info("/agent").record_count == 3
    assert research_results
    assert (
        "PostgreSQL" in research_results[0].record.content
        or "Docker" in research_results[0].record.content
    )
    assert writer_results
    assert "Release notes" in writer_results[0].record.content


def test_example13_claude_mcp_setup_config(tmp_path: pathlib.Path) -> None:
    config = generate_mcp_config(agent_id="claude-demo", data_dir=str(tmp_path), storage="sqlite")
    env = config["mcpServers"]["ai-knot"]["env"]

    assert config["mcpServers"]["ai-knot"]["command"] == "ai-knot-mcp"
    assert env["AI_KNOT_AGENT_ID"] == "claude-demo"
    assert env["AI_KNOT_STORAGE"] == "sqlite"
    assert env["AI_KNOT_DATA_DIR"] == str(tmp_path.resolve())


def test_example14_browser_inspector_demo_seed(tmp_path: pathlib.Path) -> None:
    from examples.browser_inspector_demo import build_demo_kb

    kb = build_demo_kb(base_dir=tmp_path, agent_id="browser-demo")
    facts = kb.list_facts()

    assert len(facts) == 7
    assert any("FastAPI" in fact.content for fact in facts)
    assert any("PostgreSQL" in fact.content for fact in facts)
    assert any(
        "2099" in fact.content and not fact.is_active()
        for fact in facts
    )


def test_example15_notebook_walkthrough_exists_and_mentions_core_flow() -> None:
    notebook_path = pathlib.Path("examples/notebook_walkthrough.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert len(cells) >= 6

    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
    )
    assert "Point-in-time recall" in source_text
    assert "browser_inspector_demo.py" in source_text
    assert "what language, framework, and database does the user use?" in source_text


def test_example16_pydanticai_surface_demo_builds_runtime_instructions() -> None:
    from examples.pydanticai_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.user_prompt == "Write a deployment checklist for my stack."
    assert isinstance(result.instructions, str)
    assert "You are a concise staff engineer." in result.instructions
    assert "Agent Memory" in result.instructions
    assert "Python" in result.instructions or "Docker Compose" in result.instructions
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "Docker Compose" in result.simulated_answer


def test_example17_openai_agents_surface_demo_builds_augmented_instructions() -> None:
    from examples.openai_agents_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.user_prompt == "Write a deployment checklist for my API stack."
    assert result.workflow_name == "surface-demo"
    assert isinstance(result.instructions, str)
    assert "You are a concise staff engineer." in result.instructions
    assert "Agent Memory" in result.instructions
    assert "Python" in result.instructions or "Docker Compose" in result.instructions
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "Docker Compose" in result.simulated_answer


def test_example18_autogen_surface_demo_injects_system_message() -> None:
    from examples.autogen_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.user_prompt == "Write a deployment checklist for my Python API."
    assert result.recalled_items
    assert any("Python" in item or "Docker" in item for item in result.recalled_items)
    assert "Relevant memory content" in result.injected_system_message
    assert "Python" in result.injected_system_message or "Docker" in result.injected_system_message
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "container" in result.simulated_answer


def test_example19_langgraph_surface_demo_exposes_tool_loop() -> None:
    from examples.langgraph_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.tool_names == ["add_memory", "search_memory", "list_memory", "delete_memory"]
    assert "Stored fact" in result.add_output
    assert "Docker Compose" in result.search_output or "Python" in result.search_output
    assert "Docker Compose" in result.listed_output
    assert "Deleted fact" in result.deleted_output
    assert result.remaining_facts == 1
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "Docker Compose" in result.simulated_answer


def test_example20_quickstart_exposes_list_and_delete() -> None:
    script = pathlib.Path("examples/quickstart.py").read_text(encoding="utf-8")

    assert "kb.list()" in script
    assert "kb.delete(" in script


def test_example21_structured_correction_demo_preserves_active_and_history_views() -> None:
    from examples.structured_correction import build_demo_result

    result = build_demo_result()

    assert "Globex" in result.search_output
    assert result.active_contents == ["User now works at Globex"]
    assert [row.content for row in result.history_rows] == [
        "User now works at Globex",
        "User works at Acme",
        "User works from Berlin",
    ]
    assert [row.active for row in result.history_rows] == [True, False, False]
    assert result.employer_lineage == ["User now works at Globex", "User works at Acme"]


def test_example22_npm_basic_memory_loop_exposes_search_list_delete() -> None:
    script = pathlib.Path("npm/examples/basic-memory-loop.ts").read_text(encoding="utf-8")

    assert "await kb.search(" in script
    assert "await kb.list()" in script
    assert "await kb.delete(" in script


def test_example23_cli_memory_loop_exposes_add_search_list_delete() -> None:
    script = pathlib.Path("examples/cli_memory_loop.py").read_text(encoding="utf-8")

    assert '"-m"' in script
    assert '"ai_knot.cli"' in script
    assert '"add", agent_id' in script
    assert '"search", agent_id' in script
    assert '"list", agent_id' in script
    assert '"delete", agent_id' in script


def test_example24_openclaw_integration_exposes_provider_and_ai_knot_loops() -> None:
    script = pathlib.Path("examples/openclaw_integration.py").read_text(encoding="utf-8")

    assert "memory.get_all()" in script
    assert "memory.lineage(" in script
    assert "memory.forget(" in script
    assert "memory.list()" in script


def test_example25_llamaindex_surface_demo_injects_system_memory() -> None:
    from examples.llamaindex_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.user_prompt == "Write a deployment checklist for my API stack."
    assert result.injected_role == "system"
    assert "Agent Memory" in result.injected_content
    assert "Docker Compose" in result.injected_content or "Python" in result.injected_content
    assert result.history_length == 1
    assert result.stored_fact_count == 3
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "Docker Compose" in result.simulated_answer


def test_example26_llamaindex_integration_file_exposes_real_install_path() -> None:
    script = pathlib.Path("examples/llamaindex_integration.py").read_text(encoding="utf-8")

    assert 'pip install "ai-knot[llamaindex]" "llama-index-llms-openai"' in script
    assert "SimpleChatEngine.from_defaults" in script
    assert "AiKnotLlamaIndexMemory.from_defaults" in script
    assert 'os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")' in script


def test_example27_npm_http_sidecar_exposes_http_knowledge_base() -> None:
    script = pathlib.Path("npm/examples/http-sidecar.ts").read_text(encoding="utf-8")

    assert "HttpKnowledgeBase" in script
    assert 'AI_KNOT_BASE_URL' in script
    assert "await kb.health()" in script
    assert "await kb.search(" in script
    assert "await kb.list()" in script
    assert 'includeInactive: true' in script
    assert 'op: "update"' in script


def test_example28_http_sidecar_surface_demo_exposes_http_memory_loop() -> None:
    from examples.http_sidecar_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.health_status == "ok"
    assert result.version
    assert result.added_fact_id
    assert "Docker Compose" in result.search_context
    assert "User deploys APIs with Docker Compose" in result.search_fact_contents
    assert "User deploys APIs with Docker Compose" in result.listed_fact_contents
    assert "User prefers Python over Java" in result.listed_fact_contents
    assert result.fetched_content == "User deploys APIs with Docker Compose"
    assert result.delete_status == 204
    assert result.remaining_fact_contents == ["User prefers Python over Java"]


def test_example29_function_calling_surface_demo_exposes_plain_callables() -> None:
    from examples.function_calling_surface_demo import build_demo_result

    result = build_demo_result()

    assert result.function_names == [
        "add_memory",
        "search_memory",
        "list_memory",
        "delete_memory",
        "get_memory",
    ]
    assert "Stored fact" in result.add_output
    assert "Docker Compose" in result.search_output or "Python" in result.search_output
    assert "Docker Compose" in result.listed_output
    assert "Docker Compose" in result.fetched_output
    assert "Deleted fact" in result.deleted_output
    assert result.remaining_facts == 1
    assert "Simulated next-turn answer" in result.simulated_answer
    assert "Python" in result.simulated_answer or "Docker Compose" in result.simulated_answer


@pytest.mark.parametrize(
    ("module_name", "callable_name"),
    [
        ("crewai_surface_demo", "main"),
        ("http_sidecar_surface_demo", "build_demo_result"),
        ("function_calling_surface_demo", "build_demo_result"),
        ("pydanticai_surface_demo", "build_demo_result"),
        ("openai_agents_surface_demo", "build_demo_result"),
        ("autogen_surface_demo", "build_demo_result"),
        ("langgraph_surface_demo", "build_demo_result"),
        ("llamaindex_surface_demo", "build_demo_result"),
    ],
)
def test_zero_network_surface_demos_do_not_warn_about_missing_embeddings(
    caplog: pytest.LogCaptureFixture,
    module_name: str,
    callable_name: str,
) -> None:
    module = importlib.import_module(f"examples.{module_name}")
    entrypoint = getattr(module, callable_name)

    with caplog.at_level(logging.WARNING):
        entrypoint()

    assert not any(
        "Embedding endpoint unreachable" in record.getMessage()
        for record in caplog.records
    )
