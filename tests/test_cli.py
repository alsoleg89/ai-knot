"""Tests for ai-knot CLI — all commands via Click CliRunner."""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

from ai_knot.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


def _cmd(data_dir: str, args: list[str]) -> list[str]:
    """Prefix args with --data-dir (group option must come before subcommand)."""
    return ["--data-dir", data_dir, *args]


class TestCLIShow:
    """ai-knot show <agent_id>."""

    def test_show_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert result.exit_code == 0
        assert "No facts" in result.output or "0 facts" in result.output.lower()

    def test_show_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User likes Python"]))
        result = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert result.exit_code == 0
        assert "Python" in result.output


class TestCLIAdd:
    """ai-knot add <agent_id> <content>."""

    def test_add_fact(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User prefers Docker"]))
        assert result.exit_code == 0
        assert "Added" in result.output or "added" in result.output

    def test_add_with_importance(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, _cmd(data_dir, ["add", "myagent", "Critical fact", "--importance", "0.99"])
        )
        assert result.exit_code == 0


class TestCLIRecall:
    """ai-knot recall <agent_id> <query>."""

    def test_recall_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["recall", "myagent", "test"]))
        assert result.exit_code == 0

    def test_recall_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User deploys on K8s"]))
        result = runner.invoke(main, _cmd(data_dir, ["recall", "myagent", "deploy"]))
        assert result.exit_code == 0
        assert "K8s" in result.output or "deploy" in result.output.lower()


class TestCLIStats:
    """ai-knot stats <agent_id>."""

    def test_stats_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["stats", "myagent"]))
        assert result.exit_code == 0

    def test_stats_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "Fact 1"]))
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "Fact 2"]))
        result = runner.invoke(main, _cmd(data_dir, ["stats", "myagent"]))
        assert result.exit_code == 0
        assert "2" in result.output


class TestCLIClear:
    """ai-knot clear <agent_id>."""

    def test_clear(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "Temporary"]))
        result = runner.invoke(main, _cmd(data_dir, ["clear", "myagent"]), input="y\n")
        assert result.exit_code == 0

        show = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert "No facts" in show.output or "0 facts" in show.output.lower()


class TestCLIDecay:
    """ai-knot decay <agent_id>."""

    def test_decay(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "Some fact"]))
        result = runner.invoke(main, _cmd(data_dir, ["decay", "myagent"]))
        assert result.exit_code == 0


class TestCLIExportImport:
    """ai-knot export/import."""

    def test_export_import_round_trip(
        self, runner: CliRunner, data_dir: str, tmp_path: pathlib.Path
    ) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "Exported fact"]))

        export_path = str(tmp_path / "backup.yaml")
        result = runner.invoke(main, _cmd(data_dir, ["export", "myagent", export_path]))
        assert result.exit_code == 0

        # Clear and reimport
        runner.invoke(main, _cmd(data_dir, ["clear", "myagent"]), input="y\n")
        result = runner.invoke(main, _cmd(data_dir, ["import", "myagent", export_path]))
        assert result.exit_code == 0

        show = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert "Exported fact" in show.output


class TestCLIStorageOption:
    """Test --storage backend switching."""

    def test_sqlite_backend(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["--storage", "sqlite", "--data-dir", data_dir, "add", "myagent", "SQLite fact"]
        )
        assert result.exit_code == 0

        result = runner.invoke(
            main, ["--storage", "sqlite", "--data-dir", data_dir, "show", "myagent"]
        )
        assert result.exit_code == 0
        assert "SQLite fact" in result.output

    def test_invalid_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--storage", "redis", "show", "myagent"])
        assert result.exit_code != 0
