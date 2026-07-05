"""Command-line interface for ai_knot."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from ai_knot import __version__
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage
from ai_knot.storage.base import ACLStoreCapable, EventLedgerCapable
from ai_knot.types import ConversationTurn, Fact, MemoryOp, MemoryType


class AiKnotGroup(click.Group):
    """Keep the first-run memory loop visible near the top of ``--help``."""

    COMMAND_ORDER = {
        "demo": 5,
        "add": 10,
        "add-resolved": 15,
        "learn": 20,
        "search": 30,
        "recall": 40,
        "list": 50,
        "show": 60,
        "get": 65,
        "delete": 70,
        "forget": 80,
        "doctor": 90,
        "setup": 100,
        "serve": 110,
        "serve-mcp": 115,
        "stats": 120,
        "lineage": 130,
        "export": 140,
        "import": 150,
        "audit-export": 155,
        "clear": 160,
        "decay": 170,
    }

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = super().list_commands(ctx)
        return sorted(commands, key=lambda name: (self.COMMAND_ORDER.get(name, 999), name))


CLI_HELP = (
    "ai-knot - Agent Knowledge Layer CLI.\n\n"
    "Quick proof:\n\n"
    "ai-knot demo\n\n"
    "Fastest memory loop:\n\n"
    'ai-knot add <agent_id> "User deploys APIs with Docker"\n\n'
    'ai-knot search <agent_id> "what does the user deploy with?"\n\n'
    "ai-knot list <agent_id>\n\n"
    "ai-knot delete <agent_id> <fact_id>\n\n"
    "Use ai-knot get <agent_id> <fact_id> for targeted inspection.\n\n"
    "Use ai-knot add-resolved for slot-aware update/delete/noop with lineage.\n\n"
    "Use learn / recall / forget if you prefer agent-memory verbs."
)


def _create_kb(
    *,
    storage_backend: str,
    data_dir: str,
    dsn: str | None,
    agent_id: str,
) -> KnowledgeBase:
    """Create a KnowledgeBase from explicit storage settings."""
    storage = create_storage(
        storage_backend,
        base_dir=data_dir,
        dsn=dsn,
    )
    # Honour AI_KNOT_EMBED_URL so `serve` / the container can run BM25-only
    # (set it to "") or point the dense channel at a reachable endpoint. Without
    # this the CLI always used the localhost:11434 default and ignored the env var.
    embed_url = os.environ.get("AI_KNOT_EMBED_URL", "http://localhost:11434")
    return KnowledgeBase(agent_id=agent_id, storage=storage, embed_url=embed_url)


def _make_kb(ctx: click.Context, agent_id: str) -> KnowledgeBase:
    """Create a KnowledgeBase from CLI context options."""
    return _create_kb(
        storage_backend=ctx.obj["storage"],
        data_dir=ctx.obj["data_dir"],
        dsn=ctx.obj.get("dsn"),
        agent_id=agent_id,
    )


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _module_available(name: str) -> bool:
    """Return True when *name* can be imported in this environment."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _build_mcp_config(agent_id: str, data_dir: str, storage: str) -> dict[str, Any]:
    """Build the MCP config JSON used by Claude/OpenClaw clients."""
    from typing import Literal

    from ai_knot.integrations.openclaw import generate_mcp_config

    storage_typed: Literal["sqlite", "yaml"] = "sqlite" if storage == "sqlite" else "yaml"
    return generate_mcp_config(agent_id=agent_id, data_dir=data_dir, storage=storage_typed)


def _load_existing_client_config(path: Path) -> dict[str, Any]:
    """Load an existing client config file as a mapping.

    The config must already be valid JSON/YAML. This preserves sibling keys
    while allowing ai-knot to merge ``mcpServers.ai-knot`` into the file.
    """
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise click.ClickException(
            f"Could not parse existing config at {path} as JSON/YAML. "
            "Use manual paste for commented/JSON5 configs."
        ) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise click.ClickException(f"Expected {path} to contain a top-level object.")
    return loaded


def _write_mcp_config(path: Path, config: dict[str, Any]) -> Path:
    """Write or merge the ai-knot MCP server entry into *path*."""
    target = path.expanduser()
    existing = _load_existing_client_config(target)

    existing_servers = existing.get("mcpServers")
    if existing_servers is None:
        merged_servers: dict[str, Any] = {}
    elif isinstance(existing_servers, dict):
        merged_servers = dict(existing_servers)
    else:
        raise click.ClickException(f"Expected 'mcpServers' in {target} to be an object.")

    new_servers = config.get("mcpServers")
    if not isinstance(new_servers, dict):  # pragma: no cover - generated internally
        raise click.ClickException("Generated MCP config is invalid: missing object 'mcpServers'.")

    merged_servers.update(new_servers)
    merged = dict(existing)
    merged["mcpServers"] = merged_servers

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return target.resolve()


def _default_client_config_path(client: str) -> Path:
    """Return the platform-default config path for a supported MCP client."""
    system = platform.system()

    if client == "openclaw":
        if system == "Windows":
            appdata = os.environ.get("APPDATA")
            if not appdata:
                raise click.ClickException(
                    "APPDATA is not set, so the default OpenClaw config path cannot be resolved."
                )
            return Path(appdata) / "OpenClaw" / "openclaw.json"
        return Path.home() / ".openclaw" / "openclaw.json"

    if client == "claude":
        if system == "Darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
        if system == "Windows":
            appdata = os.environ.get("APPDATA")
            if not appdata:
                raise click.ClickException(
                    "APPDATA is not set, so the default Claude config path cannot be resolved."
                )
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
        raise click.ClickException(
            "Claude's default config path is only known for macOS and Windows. "
            "Use --write-config on this platform."
        )

    raise click.ClickException(f"Unsupported MCP client {client!r}.")


def _resolve_client_write_path(
    *,
    client: str,
    write_config: Path | None,
    write_default_config: bool,
) -> Path | None:
    """Resolve the config path requested by the user for MCP setup."""
    if write_config is not None and write_default_config:
        raise click.ClickException("Use either --write-config or --write-default-config, not both.")

    if write_default_config:
        return _default_client_config_path(client)
    return write_config


def _inspect_mcp_client_config(client: str) -> dict[str, Any]:
    """Inspect the default MCP client config path for onboarding diagnostics."""
    try:
        path = _default_client_config_path(client)
    except click.ClickException as exc:
        return {
            "supported": False,
            "default_config_path": None,
            "exists": False,
            "parseable": None,
            "mcp_servers_object": None,
            "ai_knot_registered": None,
            "registered_servers": [],
            "error": str(exc),
        }

    result: dict[str, Any] = {
        "supported": True,
        "default_config_path": str(path.expanduser()),
        "exists": path.exists(),
        "parseable": None,
        "mcp_servers_object": None,
        "ai_knot_registered": None,
        "registered_servers": [],
    }
    if not path.exists():
        return result

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        result["error"] = f"Could not read config: {exc}"
        return result

    try:
        loaded = yaml.safe_load(raw) if raw else {}
    except yaml.YAMLError as exc:
        result["parseable"] = False
        result["error"] = f"Could not parse config as JSON/YAML: {exc}"
        return result

    result["parseable"] = True
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        result["mcp_servers_object"] = False
        result["error"] = "Top-level config is not an object."
        return result

    servers = loaded.get("mcpServers")
    if servers is None:
        result["mcp_servers_object"] = False
        result["ai_knot_registered"] = False
        return result
    if not isinstance(servers, dict):
        result["mcp_servers_object"] = False
        result["error"] = "'mcpServers' is not an object."
        return result

    registered_servers = sorted(str(name) for name in servers)
    result["mcp_servers_object"] = True
    result["registered_servers"] = registered_servers
    result["ai_knot_registered"] = "ai-knot" in servers
    return result


def _emit_or_write_mcp_config(
    *,
    agent_id: str,
    data_dir: str,
    storage: str,
    write_config: Path | None,
    client: str | None = None,
) -> None:
    """Print MCP config JSON or merge it into a target config file."""
    config = _build_mcp_config(agent_id, data_dir, storage)
    if write_config is None:
        click.echo(json.dumps(config, indent=2))
        return

    written = _write_mcp_config(write_config, config)
    click.echo(f"Updated {written} with mcpServers.ai-knot.")
    if client is None:
        click.echo("Run `ai-knot doctor --json` if the client still does not detect the server.")
        return

    client_name = "Claude" if client == "claude" else "OpenClaw"
    click.echo(f"Next: restart {client_name} if it was already running.")
    click.echo(
        f"Then run `ai-knot doctor --json` and check `mcp_clients.{client}.ai_knot_registered`."
    )
    click.echo("Inside the client, the memory loop stays `add -> search -> list -> delete`.")


def _doctor_payload(ctx: click.Context) -> dict[str, Any]:
    """Collect machine-readable environment diagnostics for install triage."""
    data_dir = str(Path(ctx.obj["data_dir"]).resolve())
    return {
        "ai_knot_version": __version__,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cli": {
            "storage_backend": ctx.obj["storage"],
            "data_dir": data_dir,
            "dsn_configured": bool(ctx.obj.get("dsn")),
        },
        "commands": {
            "ai_knot_mcp_on_path": shutil.which("ai-knot-mcp") is not None,
        },
        "mcp_clients": {
            "claude": _inspect_mcp_client_config("claude"),
            "openclaw": _inspect_mcp_client_config("openclaw"),
        },
        "modules": {
            "mcp": _module_available("mcp"),
            "psycopg": _module_available("psycopg"),
            "openai": _module_available("openai"),
            "fastapi": _module_available("fastapi"),
            "uvicorn": _module_available("uvicorn"),
            "crewai": _module_available("crewai"),
            "autogen_agentchat": _module_available("autogen_agentchat"),
            "autogen_ext": _module_available("autogen_ext"),
            "openai_agents_sdk": _module_available("agents"),
            "langchain_core": _module_available("langchain_core"),
            "llama_index_core": _module_available("llama_index.core"),
        },
        "env": {
            "AI_KNOT_DSN": bool(os.environ.get("AI_KNOT_DSN")),
            "AI_KNOT_PROVIDER": bool(os.environ.get("AI_KNOT_PROVIDER")),
            "AI_KNOT_API_KEY": bool(os.environ.get("AI_KNOT_API_KEY")),
            "AI_KNOT_SERVER_TOKEN": bool(os.environ.get("AI_KNOT_SERVER_TOKEN")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "LLM_API_KEY": bool(os.environ.get("LLM_API_KEY")),
        },
    }


def _render_fact_list(agent_id: str, facts: list[Fact], *, anchor: datetime | None = None) -> str:
    """Format facts the same way the CLI `show` / `list` commands do."""
    if not facts:
        return f"No facts stored for agent '{agent_id}'."

    lines: list[str] = []
    for fact in facts:
        retention_pct = f"{fact.retention_score * 100:.0f}%"
        status = "active" if fact.is_active(anchor) else "inactive"
        lines.append(
            f"  [{status}] [{fact.type.value}] {fact.content}\n"
            f"    id={fact.id}  importance={fact.importance:.2f}  "
            f"retention={retention_pct}  accessed={fact.access_count}x"
        )
    lines.append("")
    lines.append(f"{len(facts)} facts total.")
    return "\n".join(lines)


def _render_fact_detail(agent_id: str, fact_id: str, fact: Fact | None) -> str:
    """Format one fact for the CLI `get` command."""
    if fact is None:
        return f"No fact '{fact_id}' for agent '{agent_id}'."

    retention_pct = f"{fact.retention_score * 100:.0f}%"
    active = "yes" if fact.is_active() else "no"
    lines = [
        f"[{fact.type.value}] {fact.content}",
        f"id={fact.id}",
        f"importance={fact.importance:.2f}",
        f"retention={retention_pct}",
        f"accessed={fact.access_count}x",
        f"active={active}",
    ]
    if fact.tags:
        lines.append(f"tags={', '.join(fact.tags)}")
    if fact.event_time is not None:
        lines.append(f"event_time={fact.event_time.isoformat()}")
    lines.append(f"valid_from={fact.valid_from.isoformat() if fact.valid_from else 'n/a'}")
    lines.append(f"valid_until={fact.valid_until.isoformat() if fact.valid_until else '(active)'}")
    return "\n".join(lines)


def _echo_demo_step(command: str, output: str) -> None:
    """Print one demo step in a shell-transcript style."""
    click.echo(f"$ {command}")
    click.echo(output)
    click.echo()


@click.group(cls=AiKnotGroup, help=CLI_HELP)
@click.version_option()
@click.option(
    "--storage",
    "-s",
    type=click.Choice(["yaml", "sqlite", "postgres"]),
    default="yaml",
    help="Storage backend.",
)
@click.option("--data-dir", default=".ai_knot", help="Storage directory (yaml/sqlite).")
@click.option("--dsn", envvar="AI_KNOT_DSN", default=None, help="Database DSN (postgres).")
@click.pass_context
def main(ctx: click.Context, storage: str, data_dir: str, dsn: str | None) -> None:
    """Entry point for the ai-knot CLI."""
    ctx.ensure_object(dict)
    ctx.obj["storage"] = storage
    ctx.obj["data_dir"] = data_dir
    ctx.obj["dsn"] = dsn


@main.command()
@click.option("--agent-id", default="demo", show_default=True, help="Agent namespace for the demo.")
@click.option(
    "--keep-data",
    is_flag=True,
    default=False,
    help="Keep the demo store at --data-dir instead of using temporary storage.",
)
@click.pass_context
def demo(ctx: click.Context, agent_id: str, keep_data: bool) -> None:
    """Run a deterministic local proof of the add/search/list/get/delete loop."""
    storage_backend = ctx.obj["storage"]
    dsn = ctx.obj.get("dsn")

    if storage_backend == "postgres":
        raise click.ClickException(
            "ai-knot demo is a local-first proof path. Use --storage yaml or --storage sqlite."
        )

    if keep_data:
        data_dir = str(Path(ctx.obj["data_dir"]).expanduser().resolve())
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="ai-knot-demo-")
        data_dir = temp_dir.name

    click.echo(
        f"Running ai-knot demo for agent '{agent_id}' with {storage_backend} storage in {data_dir}"
    )
    click.echo()

    try:
        kb = _create_kb(
            storage_backend=storage_backend,
            data_dir=data_dir,
            dsn=dsn,
            agent_id=agent_id,
        )

        noisy = kb.add("Team standup is at 10am")
        frontend = kb.add("User prefers TypeScript for frontend work")
        deploy = kb.add("User deploys APIs with Docker Compose")

        _echo_demo_step(
            f'ai-knot add {agent_id} "Team standup is at 10am"',
            f"Added fact {noisy.id}: Team standup is at 10am",
        )
        _echo_demo_step(
            f'ai-knot add {agent_id} "User prefers TypeScript for frontend work"',
            f"Added fact {frontend.id}: User prefers TypeScript for frontend work",
        )
        _echo_demo_step(
            f'ai-knot add {agent_id} "User deploys APIs with Docker Compose"',
            f"Added fact {deploy.id}: User deploys APIs with Docker Compose",
        )
        _echo_demo_step(
            f'ai-knot search {agent_id} "what does the user deploy with?"',
            kb.search("what does the user deploy with?") or "No relevant facts found.",
        )
        _echo_demo_step(f"ai-knot list {agent_id}", _render_fact_list(agent_id, kb.list_facts()))
        _echo_demo_step(
            f"ai-knot get {agent_id} {noisy.id}",
            _render_fact_detail(agent_id, noisy.id, kb.get(noisy.id)),
        )

        forgotten = kb.get(noisy.id)
        kb.forget(noisy.id)
        _echo_demo_step(
            f"ai-knot delete {agent_id} {noisy.id}",
            f"Forgot fact {noisy.id}: {forgotten.content}",
        )
        _echo_demo_step(f"ai-knot list {agent_id}", _render_fact_list(agent_id, kb.list_facts()))

        if keep_data:
            click.echo(
                f"Demo complete. Data kept in {data_dir}. Re-run list/get/search against agent "
                f"'{agent_id}' to keep exploring."
            )
        else:
            click.echo(
                "Demo complete. Temporary data will be removed. "
                "Use --keep-data to inspect the on-disk store afterward."
            )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


@main.command()
@click.argument("agent_id")
@click.option(
    "--include-inactive",
    is_flag=True,
    default=False,
    help="Include superseded / inactive facts in the listing.",
)
@click.option(
    "--now",
    default=None,
    help="Point-in-time activity anchor for active/inactive listing status.",
)
@click.pass_context
def show(
    ctx: click.Context,
    agent_id: str,
    include_inactive: bool,
    now: str | None,
) -> None:
    """Show active stored facts for an agent by default."""
    kb = _make_kb(ctx, agent_id)
    anchor = _parse_dt(now) if now else None
    facts = kb.list_facts()
    if not include_inactive:
        facts = [fact for fact in facts if fact.is_active(anchor)]
    click.echo(_render_fact_list(agent_id, facts, anchor=anchor))


# CLI alias for users who expect a CRUD-style "list" verb.
main.add_command(show, "list")


@main.command()
@click.argument("agent_id")
@click.argument("content")
@click.option("--importance", "-i", default=0.8, type=float, help="Importance (0.0-1.0).")
@click.option(
    "--type",
    "-t",
    "fact_type",
    default="semantic",
    type=click.Choice(["semantic", "procedural", "episodic"]),
)
@click.pass_context
def add(ctx: click.Context, agent_id: str, content: str, importance: float, fact_type: str) -> None:
    """Add a fact to an agent's knowledge base."""
    if not 0.0 <= importance <= 1.0:
        raise click.BadParameter(
            f"must be between 0.0 and 1.0, got {importance}",
            param_hint="'--importance'",
        )
    if not content.strip():
        raise click.BadParameter("content must not be empty", param_hint="CONTENT")
    kb = _make_kb(ctx, agent_id)
    fact = kb.add(content, importance=importance, type=MemoryType(fact_type))
    click.echo(f"Added fact {fact.id}: {content}")


@main.command("add-resolved")
@click.argument("agent_id")
@click.argument("content")
@click.option("--entity", default=None, help="Structured entity label.")
@click.option("--attribute", default=None, help="Structured attribute name.")
@click.option("--value-text", default=None, help="Canonical value for slot resolution.")
@click.option(
    "--slot-key",
    default=None,
    help="Explicit slot address (defaults to entity::attribute).",
)
@click.option(
    "--op",
    "memory_op",
    default=MemoryOp.ADD.value,
    show_default=True,
    type=click.Choice([op.value for op in MemoryOp]),
    help="Structured memory op: add, update, delete, or noop.",
)
@click.option(
    "--event-time",
    default=None,
    help="ISO-8601 real-world anchor for this structured fact.",
)
@click.pass_context
def add_resolved(
    ctx: click.Context,
    agent_id: str,
    content: str,
    entity: str | None,
    attribute: str | None,
    value_text: str | None,
    slot_key: str | None,
    memory_op: str,
    event_time: str | None,
) -> None:
    """Insert a structured fact through supersession (no LLM extraction)."""
    if not content.strip():
        raise click.BadParameter("content must not be empty", param_hint="CONTENT")

    fact = Fact(
        content=content.strip(),
        entity=entity or "",
        attribute=attribute or "",
        value_text=value_text or "",
        slot_key=slot_key or "",
    )
    fact.op = MemoryOp(memory_op)
    if event_time is not None:
        try:
            fact.event_time = _parse_dt(event_time)
        except ValueError as exc:
            raise click.BadParameter(
                f"invalid ISO-8601 datetime: {event_time!r}",
                param_hint="'--event-time'",
            ) from exc

    kb = _make_kb(ctx, agent_id)
    inserted = kb.add_resolved([fact])
    if inserted:
        stored = inserted[0]
        resolved_slot = (
            stored.slot_key
            or fact.slot_key
            or (
                f"{fact.entity}::{fact.attribute}"
                if fact.entity and fact.attribute
                else "(unslotted)"
            )
        )
        click.echo(f"Stored resolved fact {stored.id}: {stored.content}")
        click.echo(f"slot_key={resolved_slot}  version={stored.version}  op={fact.op.value}")
        return

    target = fact.slot_key or (
        f"{fact.entity}::{fact.attribute}" if fact.entity and fact.attribute else fact.content
    )
    if fact.op == MemoryOp.DELETE:
        click.echo(f"Closed matching memory for {target}. No replacement inserted.")
        return
    if fact.op == MemoryOp.NOOP:
        click.echo(f"No changes applied for {target} (op=noop).")
        return
    click.echo("No fact inserted.")


@main.command()
@click.argument("agent_id")
@click.argument("content")
@click.option(
    "--provider",
    default=None,
    help="LLM provider for extraction. Falls back to AI_KNOT_PROVIDER or openai.",
)
@click.option(
    "--api-key",
    default=None,
    help="LLM API key. Falls back to AI_KNOT_API_KEY or provider-specific env vars.",
)
@click.option("--model", default=None, help="Optional provider model override.")
@click.option(
    "--role",
    default="user",
    type=click.Choice(["user", "assistant", "system"]),
    show_default=True,
    help="Role of the supplied message.",
)
@click.option(
    "--base-url",
    default=None,
    help="Optional base URL for openai-compat providers.",
)
@click.pass_context
def learn(
    ctx: click.Context,
    agent_id: str,
    content: str,
    provider: str | None,
    api_key: str | None,
    model: str | None,
    role: str,
    base_url: str | None,
) -> None:
    """Extract facts from raw text using an LLM and store them."""
    if not content.strip():
        raise click.BadParameter("content must not be empty", param_hint="CONTENT")

    kb = _make_kb(ctx, agent_id)
    provider_name = provider or os.environ.get("AI_KNOT_PROVIDER")
    resolved_api_key = api_key or os.environ.get("AI_KNOT_API_KEY")
    provider_kwargs: dict[str, Any] = {}
    if base_url:
        provider_kwargs["base_url"] = base_url

    turns = [ConversationTurn(role=role, content=content)]
    try:
        facts = kb.learn(
            turns,
            provider=provider_name,
            api_key=resolved_api_key,
            model=model,
            **provider_kwargs,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not facts:
        click.echo("No facts extracted.")
        return

    click.echo(f"Learned {len(facts)} fact(s):")
    for fact in facts:
        click.echo(f"  [{fact.type.value}] {fact.id}: {fact.content}")


@main.command()
@click.argument("agent_id")
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Max results.")
@click.option(
    "--now",
    default=None,
    help="Point-in-time recall: ISO datetime; facts superseded by then are excluded.",
)
@click.option(
    "--include-unsupported",
    is_flag=True,
    default=False,
    help="Include facts that carry no provenance/evidence pointer.",
)
@click.pass_context
def recall(
    ctx: click.Context,
    agent_id: str,
    query: str,
    top_k: int,
    now: str | None,
    include_unsupported: bool,
) -> None:
    """Retrieve relevant facts for a query.

    Pass --now to ask "what was true at this point in time": facts superseded
    by a later one (as of that instant) are filtered out — the bi-temporal recall.
    """
    kb = _make_kb(ctx, agent_id)
    now_dt = _parse_dt(now) if now else None
    result = kb.recall(query, top_k=top_k, now=now_dt, include_unsupported=include_unsupported)
    if result:
        click.echo(result)
    else:
        click.echo("No relevant facts found.")


# CLI alias for users who think in search terms rather than recall semantics.
main.add_command(recall, "search")


@main.command()
@click.argument("agent_id")
@click.argument("fact_id")
@click.pass_context
def get(ctx: click.Context, agent_id: str, fact_id: str) -> None:
    """Inspect one stored fact by ID."""
    kb = _make_kb(ctx, agent_id)
    try:
        fact = kb.get(fact_id)
    except KeyError:
        click.echo(_render_fact_detail(agent_id, fact_id, None))
        return

    click.echo(_render_fact_detail(agent_id, fact_id, fact))


@main.command()
@click.argument("agent_id")
@click.argument("fact_id")
@click.pass_context
def forget(ctx: click.Context, agent_id: str, fact_id: str) -> None:
    """Remove a single fact by ID."""
    kb = _make_kb(ctx, agent_id)
    facts = kb.list_facts()
    target = next((fact for fact in facts if fact.id == fact_id), None)
    if target is None:
        click.echo(f"No fact '{fact_id}' for agent '{agent_id}'.")
        return
    kb.forget(fact_id)
    click.echo(f"Forgot fact {fact_id}: {target.content}")


# CLI alias for users who think in CRUD rather than memory terminology.
main.add_command(forget, "delete")


@main.command()
@click.argument("agent_id")
@click.argument("fact_id")
@click.pass_context
def lineage(ctx: click.Context, agent_id: str, fact_id: str) -> None:
    """Show the supersession chain for FACT_ID (newest -> oldest).

    The audit trail of how a slot's value evolved: each fact and the one it
    replaced, including superseded (inactive) entries.
    """
    kb = _make_kb(ctx, agent_id)
    chain = kb.lineage(fact_id)
    if not chain:
        click.echo(f"No fact '{fact_id}' for agent '{agent_id}'.")
        return
    for i, fact in enumerate(chain):
        marker = "current" if i == 0 and fact.is_active() else "superseded"
        click.echo(
            f"  [{marker}] {fact.content}\n"
            f"    id={fact.id}  type={fact.type.value}  created={fact.created_at.isoformat()}"
        )
    click.echo(f"\n{len(chain)} version(s) in chain.")


@main.command()
@click.argument("agent_id")
@click.pass_context
def stats(ctx: click.Context, agent_id: str) -> None:
    """Show statistics for an agent's knowledge base."""
    kb = _make_kb(ctx, agent_id)
    s = kb.stats()
    click.echo(f"Agent: {agent_id}")
    click.echo(f"Total facts: {s['total_facts']}")
    click.echo(f"By type: {s['by_type']}")
    click.echo(f"Avg importance: {s['avg_importance']:.2f}")
    click.echo(f"Avg retention: {s['avg_retention']:.2f}")


@main.command()
@click.option("--json", "json_output", is_flag=True, help="Emit JSON for bug reports and triage.")
@click.pass_context
def doctor(ctx: click.Context, json_output: bool) -> None:
    """Print environment diagnostics for install and integration triage.

    The report avoids printing secret values; it only reports whether the
    relevant environment variables are present. Use ``--json`` when filing an
    issue so the output is easy to paste and inspect.
    """
    payload = _doctor_payload(ctx)
    if json_output:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


@main.command()
@click.argument("agent_id")
@click.pass_context
def decay(ctx: click.Context, agent_id: str) -> None:
    """Apply Ebbinghaus forgetting curve to all facts."""
    kb = _make_kb(ctx, agent_id)
    kb.decay()
    click.echo(f"Decay applied to agent '{agent_id}'.")


@main.command()
@click.argument("agent_id")
@click.pass_context
def clear(ctx: click.Context, agent_id: str) -> None:
    """Clear all facts for an agent."""
    if not click.confirm(f"Delete all facts for agent '{agent_id}'?"):
        click.echo("Aborted.")
        return
    kb = _make_kb(ctx, agent_id)
    kb.clear_all()
    click.echo(f"Cleared all facts for agent '{agent_id}'.")


@main.command("export")
@click.argument("agent_id")
@click.argument("output", type=click.Path())
@click.pass_context
def export_cmd(ctx: click.Context, agent_id: str, output: str) -> None:
    """Export facts to a YAML file."""
    kb = _make_kb(ctx, agent_id)
    facts = kb.list_facts()
    data: dict[str, dict[str, Any]] = {}
    for f in facts:
        data[f.id] = {
            "content": f.content,
            "type": f.type.value,
            "importance": f.importance,
            "retention_score": f.retention_score,
            "access_count": f.access_count,
            "tags": f.tags,
            "created_at": f.created_at.isoformat(),
            "last_accessed": f.last_accessed.isoformat(),
        }
    Path(output).write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    click.echo(f"Exported {len(facts)} facts to {output}")


@main.command("import")
@click.argument("agent_id")
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def import_cmd(ctx: click.Context, agent_id: str, input_file: str) -> None:
    """Import facts from a YAML file."""
    try:
        raw = yaml.safe_load(Path(input_file).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise click.ClickException(f"Invalid YAML: {exc}") from exc

    if not raw:
        click.echo("No facts in file.")
        return

    if not isinstance(raw, dict):
        raise click.ClickException("Expected a YAML mapping at the top level.")

    facts: list[Fact] = []
    for fid, entry in raw.items():
        if not isinstance(entry, dict):
            raise click.ClickException(f"Invalid entry for fact {fid!r}: expected a mapping.")
        try:
            facts.append(
                Fact(
                    id=str(fid),
                    content=entry["content"],
                    type=MemoryType(entry["type"]),
                    importance=float(entry["importance"]),
                    retention_score=float(entry.get("retention_score", 1.0)),
                    access_count=int(entry.get("access_count", 0)),
                    tags=list(entry.get("tags", [])),
                    created_at=_parse_dt(entry["created_at"]),
                    last_accessed=_parse_dt(str(entry.get("last_accessed", entry["created_at"]))),
                )
            )
        except KeyError as exc:
            raise click.ClickException(f"Fact {fid!r} is missing required field {exc}.") from exc
        except (ValueError, TypeError) as exc:
            raise click.ClickException(f"Fact {fid!r} has invalid data: {exc}.") from exc

    kb = _make_kb(ctx, agent_id)
    kb.replace_facts(facts)
    click.echo(f"Imported {len(facts)} facts for agent '{agent_id}'.")


@main.command("audit-export")
@click.option(
    "-o", "--output", type=click.Path(), default="", help="Write JSON here (default: stdout)."
)
@click.option(
    "--agent", "agent_filter", default="", help="Only include trust events for this agent."
)
@click.pass_context
def audit_export(ctx: click.Context, output: str, agent_filter: str) -> None:
    """Export the multi-agent governance audit ledger as JSON.

    Emits the durable trust-change events, fact-usage events, and access-control
    grants a ``SharedMemoryPool`` records when created with
    ``persist_stats=True`` — the "who published or recalled what, when, and why
    trust changed" stream an audit needs, without writing any code. Requires a
    SQLite or PostgreSQL backend.
    """
    storage = create_storage(
        ctx.obj["storage"], base_dir=ctx.obj["data_dir"], dsn=ctx.obj.get("dsn")
    )
    if not isinstance(storage, EventLedgerCapable):
        raise click.ClickException(
            f"The {type(storage).__name__} backend does not expose an audit ledger."
        )
    trust_events = storage.load_trust_events(agent_filter or None)
    usage_events = storage.load_usage_events()
    grants: dict[str, list[str]] = {}
    if isinstance(storage, ACLStoreCapable):
        grants = {aid: sorted(scopes) for aid, scopes in storage.load_grants().items()}

    payload = {"trust_events": trust_events, "usage_events": usage_events, "grants": grants}
    text = json.dumps(payload, indent=2, sort_keys=False)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(
            f"Wrote {len(trust_events)} trust event(s), {len(usage_events)} usage "
            f"event(s), and {len(grants)} grant(s) to {output}"
        )
    else:
        click.echo(text)
    if not trust_events and not usage_events and not grants:
        click.echo(
            "Note: the ledger is empty. A SharedMemoryPool records events only when "
            "created with persist_stats=True.",
            err=True,
        )


@main.group()
def setup() -> None:
    """Set up ai-knot integrations."""


@setup.command("claude")
@click.option("--agent-id", default="default", show_default=True, help="Agent namespace.")
@click.option(
    "--data-dir",
    default=".ai_knot",
    show_default=True,
    help="Data directory (resolved to absolute path).",
)
@click.option(
    "--storage",
    default="sqlite",
    show_default=True,
    type=click.Choice(["sqlite", "yaml"]),
    help="Storage backend.",
)
@click.option(
    "--write-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write or merge mcpServers.ai-knot into this JSON config file.",
)
@click.option(
    "--write-default-config",
    is_flag=True,
    default=False,
    help="Write or merge into Claude's platform-default config path automatically.",
)
def setup_claude(
    agent_id: str,
    data_dir: str,
    storage: str,
    write_config: Path | None,
    write_default_config: bool,
) -> None:
    """Output a paste-ready MCP config for Claude Desktop / Claude Code.

    Copy the printed JSON into:
    \b
      macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
      Windows: %APPDATA%\\Claude\\claude_desktop_config.json

    Under the "mcpServers" key.
    If the target file is already valid JSON, ``--write-config`` will merge the
    ``ai-knot`` entry for you. On macOS and Windows, ``--write-default-config``
    will resolve Claude's standard config path automatically.
    """
    target_path = _resolve_client_write_path(
        client="claude",
        write_config=write_config,
        write_default_config=write_default_config,
    )
    _emit_or_write_mcp_config(
        agent_id=agent_id,
        data_dir=data_dir,
        storage=storage,
        write_config=target_path,
        client="claude",
    )


@setup.command("openclaw")
@click.option("--agent-id", default="default", show_default=True, help="Agent namespace.")
@click.option(
    "--data-dir",
    default=".ai_knot",
    show_default=True,
    help="Data directory (resolved to absolute path).",
)
@click.option(
    "--storage",
    default="sqlite",
    show_default=True,
    type=click.Choice(["sqlite", "yaml"]),
    help="Storage backend.",
)
@click.option(
    "--write-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write or merge mcpServers.ai-knot into this JSON config file.",
)
@click.option(
    "--write-default-config",
    is_flag=True,
    default=False,
    help="Write or merge into OpenClaw's platform-default config path automatically.",
)
def setup_openclaw(
    agent_id: str,
    data_dir: str,
    storage: str,
    write_config: Path | None,
    write_default_config: bool,
) -> None:
    """Output a paste-ready MCP config for OpenClaw.

    Copy the printed JSON into:
    \b
      macOS / Linux: ~/.openclaw/openclaw.json
      Windows:       %APPDATA%\\OpenClaw\\openclaw.json

    Under the "mcpServers" key.
    If the target file is already valid JSON, ``--write-config`` will merge the
    ``ai-knot`` entry for you. ``--write-default-config`` resolves the standard
    OpenClaw config path for the current platform automatically.
    """
    target_path = _resolve_client_write_path(
        client="openclaw",
        write_config=write_config,
        write_default_config=write_default_config,
    )
    _emit_or_write_mcp_config(
        agent_id=agent_id,
        data_dir=data_dir,
        storage=storage,
        write_config=target_path,
        client="openclaw",
    )


@main.command()
@click.argument("agent_id")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=8000, type=int, help="Bind port.")
@click.pass_context
def serve(ctx: click.Context, agent_id: str, host: str, port: int) -> None:
    """Run the HTTP sidecar for AGENT_ID (requires the 'server' extra).

    Exposes /health, /v1/search, /v1/recall, /v1/facts, /v1/stats, and /inspect over HTTP.
    Set the AI_KNOT_SERVER_TOKEN environment variable to require
    ``Authorization: Bearer <token>`` on the /v1/* routes and /inspect.

    \b
      pip install "ai-knot[server]"
      ai-knot --storage sqlite serve my-agent --port 8000
    """
    try:
        import uvicorn

        from ai_knot.server import create_app
    except ImportError as exc:  # pragma: no cover - exercised via the error path only
        raise click.ClickException(
            "HTTP sidecar requires the 'server' extra: pip install 'ai-knot[server]'"
        ) from exc

    kb = _make_kb(ctx, agent_id)
    app = create_app(kb, token=os.environ.get("AI_KNOT_SERVER_TOKEN") or None)
    click.echo(f"ai-knot HTTP sidecar on http://{host}:{port}  (agent: {agent_id})")
    click.echo(f"Browser inspector: http://{host}:{port}/inspect")
    uvicorn.run(app, host=host, port=port)


@main.command("serve-mcp")
@click.argument("agent_id")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", "-p", default=8765, show_default=True, type=int, help="Bind port.")
@click.option(
    "--path",
    default="/mcp",
    show_default=True,
    help="Streamable HTTP MCP path.",
)
@click.pass_context
def serve_mcp(ctx: click.Context, agent_id: str, host: str, port: int, path: str) -> None:
    """Run the MCP server over Streamable HTTP for AGENT_ID.

    This is the remote-MCP path for HTTP-capable clients and MCP Registry
    packaging. The default stdio path remains ``ai-knot-mcp`` or the
    paste-ready config from ``ai-knot setup claude`` / ``setup openclaw``.

    \b
      pip install "ai-knot[mcp]"
      ai-knot --storage sqlite serve-mcp my-agent --port 8765
    """
    if not path.startswith("/"):
        raise click.BadParameter("path must start with '/'", param_hint="'--path'")

    try:
        from ai_knot.mcp_server import _make_server
    except ImportError as exc:  # pragma: no cover - exercised via the error path only
        raise click.ClickException(
            "MCP server requires the 'mcp' extra: pip install 'ai-knot[mcp]'"
        ) from exc

    kb = _make_kb(ctx, agent_id)
    app = _make_server(kb, host=host, port=port, streamable_http_path=path)
    click.echo(f"ai-knot MCP server on http://{host}:{port}{path}  (agent: {agent_id})")
    app.run("streamable-http")


if __name__ == "__main__":
    main()
