"""Command-line interface for ai_knot."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from ai_knot import __version__
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage
from ai_knot.types import ConversationTurn, Fact, MemoryType


def _make_kb(ctx: click.Context, agent_id: str) -> KnowledgeBase:
    """Create a KnowledgeBase from CLI context options."""
    storage = create_storage(
        ctx.obj["storage"],
        base_dir=ctx.obj["data_dir"],
        dsn=ctx.obj.get("dsn"),
    )
    return KnowledgeBase(agent_id=agent_id, storage=storage)


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


@click.group()
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
    """ai-knot — Agent Knowledge Layer CLI."""
    ctx.ensure_object(dict)
    ctx.obj["storage"] = storage
    ctx.obj["data_dir"] = data_dir
    ctx.obj["dsn"] = dsn


@main.command()
@click.argument("agent_id")
@click.pass_context
def show(ctx: click.Context, agent_id: str) -> None:
    """Show all stored facts for an agent."""
    kb = _make_kb(ctx, agent_id)
    facts = kb.list_facts()

    if not facts:
        click.echo(f"No facts stored for agent '{agent_id}'.")
        return

    for fact in facts:
        retention_pct = f"{fact.retention_score * 100:.0f}%"
        click.echo(
            f"  [{fact.type.value}] {fact.content}\n"
            f"    id={fact.id}  importance={fact.importance:.2f}  "
            f"retention={retention_pct}  accessed={fact.access_count}x"
        )
    click.echo(f"\n{len(facts)} facts total.")


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
    provider_kwargs: dict[str, str] = {}
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


@main.group()
def setup() -> None:
    """Set up ai-knot integrations."""


def _emit_mcp_config(agent_id: str, data_dir: str, storage: str) -> None:
    """Print the MCP config JSON used by Claude/OpenClaw clients."""
    from typing import Literal

    from ai_knot.integrations.openclaw import generate_mcp_config

    storage_typed: Literal["sqlite", "yaml"] = "sqlite" if storage == "sqlite" else "yaml"
    config = generate_mcp_config(agent_id=agent_id, data_dir=data_dir, storage=storage_typed)
    click.echo(json.dumps(config, indent=2))


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
def setup_claude(agent_id: str, data_dir: str, storage: str) -> None:
    """Output a paste-ready MCP config for Claude Desktop / Claude Code.

    Copy the printed JSON into:
    \b
      macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
      Windows: %APPDATA%\\Claude\\claude_desktop_config.json

    Under the "mcpServers" key.
    """
    _emit_mcp_config(agent_id, data_dir, storage)


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
def setup_openclaw(agent_id: str, data_dir: str, storage: str) -> None:
    """Output a paste-ready MCP config for OpenClaw.

    Copy the printed JSON into:
    \b
      macOS / Linux: ~/.openclaw/openclaw.json
      Windows:       %APPDATA%\\OpenClaw\\openclaw.json

    Under the "mcpServers" key.
    """
    _emit_mcp_config(agent_id, data_dir, storage)


@main.command()
@click.argument("agent_id")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=8000, type=int, help="Bind port.")
@click.pass_context
def serve(ctx: click.Context, agent_id: str, host: str, port: int) -> None:
    """Run the HTTP sidecar for AGENT_ID (requires the 'server' extra).

    Exposes /health, /v1/recall, /v1/facts, /v1/stats, and /inspect over HTTP.
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


if __name__ == "__main__":
    main()
