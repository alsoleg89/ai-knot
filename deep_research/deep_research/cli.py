from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from deep_research.memory import SemanticMemory

from deep_research.config import CampaignConfig
from deep_research.corpus import Corpus


@click.group()
def main() -> None:
    """Deep Research Engine — autonomous multi-agent memory breakthrough engine."""


@main.command()
@click.option("--brief", "brief_file", type=click.Path(exists=True), default=None)
@click.option("--corpus-dir", default="campaign", show_default=True)
@click.option("--mock", is_flag=True, help="Use mock LLM — no API calls, for testing")
@click.option(
    "--cli",
    "use_cli",
    is_flag=True,
    help="Use local `claude` CLI binary (Claude Max subscription, no API key needed)",
)
@click.option("--model", default="claude-opus-4-7", show_default=True, help="Model name or alias")
@click.option("--tick-budget", default=10_000, show_default=True)
@click.option("--wall-hours", default=72.0, show_default=True, help="Max wall-clock hours")
@click.option("--token-budget", default=50_000_000, show_default=True)
@click.option(
    "--tick-sleep",
    default=0.0,
    show_default=True,
    help="Seconds to sleep between ticks (rate limiting)",
)
@click.option(
    "--embedding-model",
    default="BAAI/bge-m3",
    show_default=True,
    help="HF embedding model for semantic memory (requires [semantic] extra)",
)
@click.option(
    "--reranker-model",
    default="BAAI/bge-reranker-v2-m3",
    show_default=True,
    help="HF cross-encoder reranker model",
)
@click.option(
    "--no-semantic", "disable_semantic", is_flag=True, help="Disable semantic memory index"
)
def start(
    brief_file: str | None,
    corpus_dir: str,
    mock: bool,
    use_cli: bool,
    model: str,
    tick_budget: int,
    wall_hours: float,
    token_budget: int,
    tick_sleep: float,
    embedding_model: str,
    reranker_model: str,
    disable_semantic: bool,
) -> None:
    """Start or resume a research campaign."""
    corpus = Corpus(Path(corpus_dir))
    state_path = corpus.root / "state.json"

    effective_model = "mock" if mock else model

    if state_path.exists():
        # Resume existing campaign
        state = corpus.load_state()
        if state.status == "exhausted":
            click.echo(
                f"Campaign {state.campaign_id[:8]} is exhausted (tick {state.tick}). "
                "Remove the corpus directory to start fresh.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Resuming campaign {state.campaign_id[:8]} from tick {state.tick}.")
        brief_text = Path(brief_file).read_text() if brief_file else (state.brief or state.focus)
        if not state.brief:
            state.brief = brief_text
        if state.focus == "initial exploration" and brief_text:
            state.focus = brief_text
        state.status = "running"
        corpus.save_state(state)
        config = CampaignConfig(
            brief_text=brief_text,
            model=effective_model,
            tick_budget=tick_budget,
            wall_clock_seconds=int(wall_hours * 3600),
            token_budget=token_budget,
            tick_sleep_seconds=tick_sleep,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    else:
        if not brief_file:
            click.echo("--brief is required for a new campaign.", err=True)
            sys.exit(1)
        brief_text = Path(brief_file).read_text()
        config = CampaignConfig(
            brief_text=brief_text,
            model=effective_model,
            tick_budget=tick_budget,
            wall_clock_seconds=int(wall_hours * 3600),
            token_budget=token_budget,
            tick_sleep_seconds=tick_sleep,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        state = corpus.initialize(
            campaign_id=str(uuid.uuid4()),
            config_hash=config.hash(),
            brief=brief_text,
        )
        click.echo(f"Campaign {state.campaign_id[:8]} started. Config hash: {config.hash()}")

    backend = "mock" if mock else ("claude-cli" if use_cli else "anthropic-sdk")
    click.echo(f"Corpus: {corpus_dir} | Backend: {backend} | Model: {effective_model}")

    if mock:
        from deep_research.llm import MockLLMClient

        llm: object = MockLLMClient()
    elif use_cli:
        from deep_research.llm import ClaudeCliLLMClient

        llm = ClaudeCliLLMClient(model=model)
    else:
        from deep_research.llm import AnthropicLLMClient

        llm = AnthropicLLMClient(model=model)

    from deep_research.llm import LLMClient
    from deep_research.orchestrator import Orchestrator

    assert isinstance(llm, LLMClient)

    semantic: SemanticMemory | None = None
    if not mock and not disable_semantic:
        from deep_research.memory import SemanticMemory as _SemanticMemory
        from deep_research.semantic import CrossEncoderReranker, SentenceTransformerEmbedder

        click.echo(f"Loading embedding model {config.embedding_model}...")
        _embedder = SentenceTransformerEmbedder(model=config.embedding_model)
        click.echo(f"Loading reranker {config.reranker_model}...")
        _reranker = CrossEncoderReranker(model=config.reranker_model)
        semantic = _SemanticMemory(corpus, _embedder, _reranker)

    orch = Orchestrator(config=config, corpus=corpus, llm=llm, semantic=semantic)
    try:
        orch.run()
        final = corpus.load_state()
        click.echo(f"Campaign {final.status} at tick {final.tick}.")
    except KeyboardInterrupt:
        state = corpus.load_state()
        state.status = "stopped"
        corpus.save_state(state)
        click.echo(f"\nCampaign stopped at tick {state.tick}.")


@main.command()
@click.option("--corpus-dir", default="campaign", show_default=True)
def status(corpus_dir: str) -> None:
    """Show campaign status and budgets."""
    corpus = Corpus(Path(corpus_dir))
    try:
        state = corpus.load_state()
    except FileNotFoundError:
        click.echo("No campaign found. Run 'start --brief <file>' first.", err=True)
        sys.exit(1)
    click.echo(json.dumps(state.to_dict(), indent=2))


@main.command()
@click.option("--corpus-dir", default="campaign", show_default=True)
def stop(corpus_dir: str) -> None:
    """Stop a running campaign (checkpoint is preserved)."""
    corpus = Corpus(Path(corpus_dir))
    try:
        state = corpus.load_state()
    except FileNotFoundError:
        click.echo("No campaign found.", err=True)
        sys.exit(1)
    state.status = "stopped"
    corpus.save_state(state)
    click.echo(f"Campaign stopped at tick {state.tick}.")


@main.command()
@click.option("--corpus-dir", default="campaign", show_default=True)
def theory(corpus_dir: str) -> None:
    """Show the current live theory."""
    corpus = Corpus(Path(corpus_dir))
    click.echo(corpus.read_theory())


@main.command()
@click.option("--corpus-dir", default="campaign", show_default=True)
@click.option("-n", default=20, show_default=True, help="Number of recent entries")
def journal(corpus_dir: str, n: int) -> None:
    """Show recent journal entries."""
    corpus = Corpus(Path(corpus_dir))
    entries = corpus.read_journal(n)
    if not entries:
        click.echo("(journal empty)")
        return
    for entry in entries:
        click.echo(json.dumps(entry))


@main.command()
@click.argument("exp_id")
@click.option("--corpus-dir", default="campaign", show_default=True)
def approve(exp_id: str, corpus_dir: str) -> None:
    """Approve a pending experiment (allows paid benchmark run)."""
    corpus = Corpus(Path(corpus_dir))
    if corpus.approve_experiment(exp_id):
        click.echo(f"Approved: {exp_id}")
    else:
        click.echo(f"Experiment not found: {exp_id}", err=True)
        sys.exit(1)


@main.command()
@click.argument("exp_id")
@click.option("--corpus-dir", default="campaign", show_default=True)
def reject(exp_id: str, corpus_dir: str) -> None:
    """Reject a pending experiment."""
    corpus = Corpus(Path(corpus_dir))
    if corpus.reject_experiment(exp_id):
        click.echo(f"Rejected: {exp_id}")
    else:
        click.echo(f"Experiment not found: {exp_id}", err=True)
        sys.exit(1)
