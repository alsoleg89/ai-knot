from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from deep_research.cli import main
from deep_research.config import CampaignConfig
from deep_research.corpus import Corpus


def _init_corpus(tmp_path: Path, tick: int = 0, status: str = "running") -> Corpus:
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(brief_text="test brief", tick_budget=4)
    state = corpus.initialize("id-1", config.hash())
    state.tick = tick
    state.status = status
    corpus.save_state(state)
    return corpus


def test_start_new_campaign(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    brief.write_text("multi-agent memory research brief")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "start",
            "--brief",
            str(brief),
            "--mock",
            "--tick-budget",
            "2",
            "--corpus-dir",
            str(tmp_path / "campaign"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Campaign" in result.output
    corpus = Corpus(tmp_path / "campaign")
    assert corpus.load_state().tick == 2


def test_start_new_campaign_persists_brief_and_focus(tmp_path: Path) -> None:
    brief = tmp_path / "brief.txt"
    brief.write_text("one-off fact retrieval brief")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "start",
            "--brief",
            str(brief),
            "--mock",
            "--tick-budget",
            "0",
            "--corpus-dir",
            str(tmp_path / "campaign"),
        ],
    )
    assert result.exit_code == 0, result.output
    state = Corpus(tmp_path / "campaign").load_state()
    assert state.brief == "one-off fact retrieval brief"
    assert state.focus == "one-off fact retrieval brief"


def test_start_fails_without_brief_for_new_campaign(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--mock", "--corpus-dir", str(tmp_path / "new")])
    assert result.exit_code != 0
    assert "--brief is required" in result.output


def test_resume_from_stopped(tmp_path: Path) -> None:
    """Regression: resuming a 'stopped' campaign resets status and continues."""
    corpus = _init_corpus(tmp_path, tick=2, status="stopped")
    brief = tmp_path / "brief.txt"
    brief.write_text("brief")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "start",
            "--brief",
            str(brief),
            "--mock",
            "--tick-budget",
            "4",
            "--corpus-dir",
            str(tmp_path / "campaign"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Resuming" in result.output
    final = corpus.load_state()
    assert final.tick == 4
    assert final.status == "exhausted"


def test_resume_legacy_campaign_adopts_supplied_brief(tmp_path: Path) -> None:
    corpus = _init_corpus(tmp_path, tick=0, status="stopped")
    state = corpus.load_state()
    state.focus = "initial exploration"
    corpus.save_state(state)
    brief = tmp_path / "brief.txt"
    brief.write_text("legacy recovery brief")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "start",
            "--brief",
            str(brief),
            "--mock",
            "--tick-budget",
            "0",
            "--corpus-dir",
            str(tmp_path / "campaign"),
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = corpus.load_state()
    assert loaded.brief == "legacy recovery brief"
    assert loaded.focus == "legacy recovery brief"


def test_resume_exhausted_campaign_fails(tmp_path: Path) -> None:
    _init_corpus(tmp_path, tick=4, status="exhausted")
    brief = tmp_path / "brief.txt"
    brief.write_text("brief")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["start", "--brief", str(brief), "--mock", "--corpus-dir", str(tmp_path / "campaign")],
    )
    assert result.exit_code != 0
    assert "exhausted" in result.output


def test_stop_command(tmp_path: Path) -> None:
    _init_corpus(tmp_path, status="running")
    runner = CliRunner()
    result = runner.invoke(main, ["stop", "--corpus-dir", str(tmp_path / "campaign")])
    assert result.exit_code == 0
    corpus = Corpus(tmp_path / "campaign")
    assert corpus.load_state().status == "stopped"


def test_theory_command_empty(tmp_path: Path) -> None:
    _init_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["theory", "--corpus-dir", str(tmp_path / "campaign")])
    assert result.exit_code == 0
    assert "(no theory yet)" in result.output


def test_journal_command_empty(tmp_path: Path) -> None:
    _init_corpus(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["journal", "--corpus-dir", str(tmp_path / "campaign")])
    assert result.exit_code == 0
    assert "(journal empty)" in result.output


def test_approve_reject_experiment(tmp_path: Path) -> None:
    corpus = _init_corpus(tmp_path)
    corpus.add_to_approval_queue("exp-abc", "run benchmark X")
    runner = CliRunner()
    result = runner.invoke(main, ["approve", "exp-abc", "--corpus-dir", str(tmp_path / "campaign")])
    assert result.exit_code == 0
    assert "Approved" in result.output
    result = runner.invoke(main, ["reject", "exp-abc", "--corpus-dir", str(tmp_path / "campaign")])
    assert result.exit_code == 0
    assert "Rejected" in result.output
