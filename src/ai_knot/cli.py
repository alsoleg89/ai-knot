"""Command-line interface for ai_knot."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage
from ai_knot.types import Fact, MemoryType


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
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Max results.")
@click.pass_context
def recall(ctx: click.Context, agent_id: str, query: str, top_k: int) -> None:
    """Retrieve relevant facts for a query."""
    kb = _make_kb(ctx, agent_id)
    result = kb.recall(query, top_k=top_k)
    if result:
        click.echo(result)
    else:
        click.echo("No relevant facts found.")


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
    import json
    from typing import Literal

    from ai_knot.integrations.openclaw import generate_mcp_config

    storage_typed: Literal["sqlite", "yaml"] = "sqlite" if storage == "sqlite" else "yaml"
    config = generate_mcp_config(agent_id=agent_id, data_dir=data_dir, storage=storage_typed)
    click.echo(json.dumps(config, indent=2))
