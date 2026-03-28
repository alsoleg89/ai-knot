"""Command-line interface for agentmemo."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml

from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import Fact, MemoryType


def _make_kb(agent_id: str, data_dir: str) -> KnowledgeBase:
    """Create a KnowledgeBase backed by the given data directory."""
    storage = YAMLStorage(base_dir=data_dir)
    return KnowledgeBase(agent_id=agent_id, storage=storage)


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-format datetime string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@click.group()
@click.version_option()
def main() -> None:
    """agentmemo — Agent Knowledge Layer CLI."""


@main.command()
@click.argument("agent_id")
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def show(agent_id: str, data_dir: str) -> None:
    """Show all stored facts for an agent."""
    kb = _make_kb(agent_id, data_dir)
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
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def add(agent_id: str, content: str, importance: float, fact_type: str, data_dir: str) -> None:
    """Add a fact to an agent's knowledge base."""
    if not 0.0 <= importance <= 1.0:
        raise click.BadParameter(
            f"must be between 0.0 and 1.0, got {importance}",
            param_hint="'--importance'",
        )
    if not content.strip():
        raise click.BadParameter("content must not be empty", param_hint="CONTENT")
    kb = _make_kb(agent_id, data_dir)
    fact = kb.add(content, importance=importance, type=MemoryType(fact_type))
    click.echo(f"Added fact {fact.id}: {content}")


@main.command()
@click.argument("agent_id")
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="Max results.")
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def recall(agent_id: str, query: str, top_k: int, data_dir: str) -> None:
    """Retrieve relevant facts for a query."""
    kb = _make_kb(agent_id, data_dir)
    result = kb.recall(query, top_k=top_k)
    if result:
        click.echo(result)
    else:
        click.echo("No relevant facts found.")


@main.command()
@click.argument("agent_id")
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def stats(agent_id: str, data_dir: str) -> None:
    """Show statistics for an agent's knowledge base."""
    kb = _make_kb(agent_id, data_dir)
    s = kb.stats()
    click.echo(f"Agent: {agent_id}")
    click.echo(f"Total facts: {s['total_facts']}")
    click.echo(f"By type: {s['by_type']}")
    click.echo(f"Avg importance: {s['avg_importance']:.2f}")
    click.echo(f"Avg retention: {s['avg_retention']:.2f}")


@main.command()
@click.argument("agent_id")
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def decay(agent_id: str, data_dir: str) -> None:
    """Apply Ebbinghaus forgetting curve to all facts."""
    kb = _make_kb(agent_id, data_dir)
    kb.decay()
    click.echo(f"Decay applied to agent '{agent_id}'.")


@main.command()
@click.argument("agent_id")
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def clear(agent_id: str, data_dir: str) -> None:
    """Clear all facts for an agent."""
    if not click.confirm(f"Delete all facts for agent '{agent_id}'?"):
        click.echo("Aborted.")
        return
    kb = _make_kb(agent_id, data_dir)
    kb.clear_all()
    click.echo(f"Cleared all facts for agent '{agent_id}'.")


@main.command("export")
@click.argument("agent_id")
@click.argument("output", type=click.Path())
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def export_cmd(agent_id: str, output: str, data_dir: str) -> None:
    """Export facts to a YAML file."""
    kb = _make_kb(agent_id, data_dir)
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
@click.option("--data-dir", default=".agentmemo", help="Storage directory.")
def import_cmd(agent_id: str, input_file: str, data_dir: str) -> None:
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
                    last_accessed=_parse_dt(
                        str(entry.get("last_accessed", entry["created_at"]))
                    ),
                )
            )
        except KeyError as exc:
            raise click.ClickException(f"Fact {fid!r} is missing required field {exc}.") from exc
        except (ValueError, TypeError) as exc:
            raise click.ClickException(f"Fact {fid!r} has invalid data: {exc}.") from exc

    kb = _make_kb(agent_id, data_dir)
    kb.replace_facts(facts)
    click.echo(f"Imported {len(facts)} facts for agent '{agent_id}'.")
