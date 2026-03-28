"""Tests for agentmemo CLI — all commands via Click CliRunner."""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

from agentmemo.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


class TestCLIShow:
    """agentmemo show <agent_id>."""

    def test_show_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, ["show", "myagent", "--data-dir", data_dir])
        assert result.exit_code == 0
        assert "No facts" in result.output or "0 facts" in result.output.lower()

    def test_show_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, ["add", "myagent", "User likes Python", "--data-dir", data_dir])
        result = runner.invoke(main, ["show", "myagent", "--data-dir", data_dir])
        assert result.exit_code == 0
        assert "Python" in result.output


class TestCLIAdd:
    """agentmemo add <agent_id> <content>."""

    def test_add_fact(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["add", "myagent", "User prefers Docker", "--data-dir", data_dir]
        )
        assert result.exit_code == 0
        assert "Added" in result.output or "added" in result.output

    def test_add_with_importance(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main,
            ["add", "myagent", "Critical fact", "--importance", "0.99", "--data-dir", data_dir],
        )
        assert result.exit_code == 0


class TestCLIRecall:
    """agentmemo recall <agent_id> <query>."""

    def test_recall_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, ["recall", "myagent", "test", "--data-dir", data_dir])
        assert result.exit_code == 0

    def test_recall_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, ["add", "myagent", "User deploys on K8s", "--data-dir", data_dir])
        result = runner.invoke(main, ["recall", "myagent", "deploy", "--data-dir", data_dir])
        assert result.exit_code == 0
        assert "K8s" in result.output or "deploy" in result.output.lower()


class TestCLIStats:
    """agentmemo stats <agent_id>."""

    def test_stats_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, ["stats", "myagent", "--data-dir", data_dir])
        assert result.exit_code == 0

    def test_stats_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, ["add", "myagent", "Fact 1", "--data-dir", data_dir])
        runner.invoke(main, ["add", "myagent", "Fact 2", "--data-dir", data_dir])
        result = runner.invoke(main, ["stats", "myagent", "--data-dir", data_dir])
        assert result.exit_code == 0
        assert "2" in result.output


class TestCLIClear:
    """agentmemo clear <agent_id>."""

    def test_clear(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, ["add", "myagent", "Temporary", "--data-dir", data_dir])
        result = runner.invoke(main, ["clear", "myagent", "--data-dir", data_dir], input="y\n")
        assert result.exit_code == 0

        show = runner.invoke(main, ["show", "myagent", "--data-dir", data_dir])
        assert "No facts" in show.output or "0 facts" in show.output.lower()


class TestCLIDecay:
    """agentmemo decay <agent_id>."""

    def test_decay(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, ["add", "myagent", "Some fact", "--data-dir", data_dir])
        result = runner.invoke(main, ["decay", "myagent", "--data-dir", data_dir])
        assert result.exit_code == 0


class TestCLIExportImport:
    """agentmemo export/import."""

    def test_export_import_round_trip(
        self, runner: CliRunner, data_dir: str, tmp_path: pathlib.Path
    ) -> None:
        runner.invoke(main, ["add", "myagent", "Exported fact", "--data-dir", data_dir])

        export_path = str(tmp_path / "backup.yaml")
        result = runner.invoke(main, ["export", "myagent", export_path, "--data-dir", data_dir])
        assert result.exit_code == 0

        # Clear and reimport
        runner.invoke(main, ["clear", "myagent", "--data-dir", data_dir], input="y\n")
        result = runner.invoke(
            main, ["import", "myagent", export_path, "--data-dir", data_dir]
        )
        assert result.exit_code == 0

        show = runner.invoke(main, ["show", "myagent", "--data-dir", data_dir])
        assert "Exported fact" in show.output
