"""Tests for ai-knot CLI — all commands via Click CliRunner."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
from click.testing import CliRunner

from ai_knot import Fact, KnowledgeBase
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
    """ai-knot show/list <agent_id>."""

    def test_show_empty(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert result.exit_code == 0
        assert "No facts" in result.output or "0 facts" in result.output.lower()

    def test_show_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User likes Python"]))
        result = runner.invoke(main, _cmd(data_dir, ["show", "myagent"]))
        assert result.exit_code == 0
        assert "Python" in result.output

    def test_list_alias_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User likes Python"]))
        result = runner.invoke(main, _cmd(data_dir, ["list", "myagent"]))
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


class TestCLILearn:
    """ai-knot learn <agent_id> <content>."""

    def test_learn_prints_extracted_facts(
        self,
        runner: CliRunner,
        data_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded: dict[str, object] = {}

        def fake_learn(
            self: KnowledgeBase,
            turns: list[object],
            *,
            api_key: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            **provider_kwargs: str,
        ) -> list[Fact]:
            recorded["turns"] = turns
            recorded["api_key"] = api_key
            recorded["provider"] = provider
            recorded["model"] = model
            recorded["provider_kwargs"] = provider_kwargs
            return [
                Fact(id="abc12345", content="User deploys in Docker"),
                Fact(id="def67890", content="User uses PostgreSQL"),
            ]

        monkeypatch.setattr(KnowledgeBase, "learn", fake_learn)

        result = runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "learn",
                    "myagent",
                    "User deploys in Docker and uses PostgreSQL",
                    "--provider",
                    "openai",
                ],
            ),
        )

        assert result.exit_code == 0
        assert "Learned 2 fact" in result.output
        assert "User deploys in Docker" in result.output
        assert recorded["provider"] == "openai"

    def test_learn_reports_no_facts_extracted(
        self,
        runner: CliRunner,
        data_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_learn(
            self: KnowledgeBase,
            turns: list[object],
            *,
            api_key: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            **provider_kwargs: str,
        ) -> list[Fact]:
            return []

        monkeypatch.setattr(KnowledgeBase, "learn", fake_learn)

        result = runner.invoke(
            main,
            _cmd(data_dir, ["learn", "myagent", "Nothing extractable", "--provider", "openai"]),
        )

        assert result.exit_code == 0
        assert "No facts extracted" in result.output

    def test_learn_value_error_becomes_click_exception(
        self,
        runner: CliRunner,
        data_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_learn(
            self: KnowledgeBase,
            turns: list[object],
            *,
            api_key: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            **provider_kwargs: str,
        ) -> list[Fact]:
            raise ValueError("No API key for provider 'openai'.")

        monkeypatch.setattr(KnowledgeBase, "learn", fake_learn)

        result = runner.invoke(main, _cmd(data_dir, ["learn", "myagent", "raw note"]))

        assert result.exit_code != 0
        assert "No API key" in result.output


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

    def test_search_alias_with_facts(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "myagent", "User deploys on K8s"]))
        result = runner.invoke(main, _cmd(data_dir, ["search", "myagent", "deploy"]))
        assert result.exit_code == 0
        assert "K8s" in result.output or "deploy" in result.output.lower()


class TestCLIForget:
    """ai-knot forget/delete <agent_id> <fact_id>."""

    def test_forget_removes_single_fact(self, runner: CliRunner, data_dir: str) -> None:
        add = runner.invoke(main, _cmd(data_dir, ["add", "agent", "Fact to remove"]))
        fact_id = add.output.split()[2].rstrip(":")

        result = runner.invoke(main, _cmd(data_dir, ["forget", "agent", fact_id]))
        assert result.exit_code == 0
        assert "Forgot fact" in result.output

        show = runner.invoke(main, _cmd(data_dir, ["show", "agent"]))
        assert "Fact to remove" not in show.output

    def test_delete_alias_removes_single_fact(self, runner: CliRunner, data_dir: str) -> None:
        add = runner.invoke(main, _cmd(data_dir, ["add", "agent", "Alias target"]))
        fact_id = add.output.split()[2].rstrip(":")

        result = runner.invoke(main, _cmd(data_dir, ["delete", "agent", fact_id]))
        assert result.exit_code == 0
        assert "Forgot fact" in result.output

    def test_forget_unknown_fact_is_non_fatal(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["forget", "agent", "deadbeef"]))
        assert result.exit_code == 0
        assert "No fact" in result.output


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


class TestCLIDoctor:
    """ai-knot doctor."""

    def test_doctor_text_output(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["doctor"]))
        assert result.exit_code == 0
        assert "ai_knot_version:" in result.output
        assert "storage_backend: yaml" in result.output

    def test_doctor_json_output(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main,
            ["--storage", "sqlite", "--data-dir", data_dir, "doctor", "--json"],
        )
        assert result.exit_code == 0

        import json

        payload = json.loads(result.output)
        assert payload["ai_knot_version"] == "0.11.0"
        assert payload["cli"]["storage_backend"] == "sqlite"
        assert payload["cli"]["data_dir"] == str(pathlib.Path(data_dir).resolve())
        assert "modules" in payload
        assert "env" in payload

    def test_doctor_module_entrypoint(self, data_dir: str) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_knot.cli",
                "--data-dir",
                data_dir,
                "doctor",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert '"ai_knot_version"' in result.stdout


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


class TestCLIRecallNow:
    """ai-knot recall --now <iso> — point-in-time (bi-temporal) recall."""

    def test_recall_with_now_accepts_iso_and_returns_fact(
        self, runner: CliRunner, data_dir: str
    ) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "agent", "User deploys with Docker"]))
        result = runner.invoke(
            main, _cmd(data_dir, ["recall", "agent", "deploy", "--now", "2030-01-01T00:00:00"])
        )
        assert result.exit_code == 0
        assert "Docker" in result.output

    def test_recall_with_invalid_now_errors(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(main, _cmd(data_dir, ["add", "agent", "User deploys with Docker"]))
        result = runner.invoke(
            main, _cmd(data_dir, ["recall", "agent", "deploy", "--now", "not-a-date"])
        )
        assert result.exit_code != 0


class TestCLILineage:
    """ai-knot lineage <agent_id> <fact_id> — supersession audit trail."""

    def test_lineage_unknown_fact(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["lineage", "agent", "deadbeef"]))
        assert result.exit_code == 0
        assert "No fact" in result.output

    def test_lineage_single_version(self, runner: CliRunner, data_dir: str) -> None:
        add = runner.invoke(main, _cmd(data_dir, ["add", "agent", "User likes Go"]))
        fact_id = add.output.split()[2].rstrip(":")  # "Added fact <id>: ..."
        result = runner.invoke(main, _cmd(data_dir, ["lineage", "agent", fact_id]))
        assert result.exit_code == 0
        assert "current" in result.output
        assert "1 version" in result.output


class TestCLISetup:
    """ai-knot setup <client>."""

    def test_setup_claude_outputs_mcp_json(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["setup", "claude", "--agent-id", "bot"])
        assert result.exit_code == 0
        assert '"mcpServers"' in result.output
        assert '"AI_KNOT_AGENT_ID": "bot"' in result.output

    def test_setup_openclaw_outputs_mcp_json(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["setup", "openclaw", "--agent-id", "bot"])
        assert result.exit_code == 0
        assert '"mcpServers"' in result.output
        assert '"AI_KNOT_AGENT_ID": "bot"' in result.output
        assert '"AI_KNOT_STORAGE": "sqlite"' in result.output
