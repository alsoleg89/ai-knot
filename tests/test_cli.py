"""Tests for ai-knot CLI — all commands via Click CliRunner."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai_knot import Fact, KnowledgeBase
from ai_knot.cli import main
from ai_knot.storage import create_storage


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

    def test_list_hides_inactive_by_default(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User works at Acme",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Acme",
                ],
            ),
        )
        runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User now works at Globex",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Globex",
                    "--op",
                    "update",
                ],
            ),
        )
        result = runner.invoke(main, _cmd(data_dir, ["list", "myagent"]))
        assert result.exit_code == 0
        assert "Globex" in result.output
        assert "Acme" not in result.output

    def test_list_can_include_inactive_history(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User works at Acme",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Acme",
                ],
            ),
        )
        runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User now works at Globex",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Globex",
                    "--op",
                    "update",
                ],
            ),
        )
        result = runner.invoke(main, _cmd(data_dir, ["list", "myagent", "--include-inactive"]))
        assert result.exit_code == 0
        assert "Globex" in result.output
        assert "Acme" in result.output
        assert "[inactive]" in result.output


class TestCLIHelp:
    """Top-level help should surface the first-run memory loop."""

    def test_help_surfaces_memory_loop_and_command_order(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

        output = result.output
        assert "Quick proof:" in output
        assert "ai-knot demo" in output
        assert "Fastest memory loop:" in output
        assert 'ai-knot add <agent_id> "User deploys APIs with Docker"' in output
        assert 'ai-knot search <agent_id> "what does the user deploy with?"' in output
        assert "ai-knot list <agent_id>" in output
        assert "ai-knot delete <agent_id> <fact_id>" in output
        assert "ai-knot get <agent_id> <fact_id> for targeted inspection." in output
        assert "Use ai-knot add-resolved for slot-aware update/delete/noop with lineage." in output

        assert output.index("  demo") < output.index("  add")
        assert output.index("  add") < output.index("  add-resolved")
        assert output.index("  add-resolved") < output.index("  learn")
        assert output.index("  learn") < output.index("  search")
        assert output.index("  search") < output.index("  recall")
        assert output.index("  recall") < output.index("  list")
        assert output.index("  list") < output.index("  show")
        assert output.index("  show") < output.index("  get")
        assert output.index("  show") < output.index("  delete")
        assert output.index("  get") < output.index("  delete")
        assert output.index("  delete") < output.index("  forget")
        assert output.index("  forget") < output.index("  doctor")
        assert output.index("  doctor") < output.index("  setup")


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


class TestCLIDemo:
    """ai-knot demo."""

    def test_demo_runs_full_loop_and_keeps_data(
        self,
        runner: CliRunner,
        data_dir: str,
    ) -> None:
        result = runner.invoke(
            main,
            [
                "--storage",
                "sqlite",
                "--data-dir",
                data_dir,
                "demo",
                "--keep-data",
                "--agent-id",
                "proof",
            ],
        )

        assert result.exit_code == 0
        output = result.output
        assert "Running ai-knot demo for agent 'proof' with sqlite storage" in output
        assert '$ ai-knot add proof "Team standup is at 10am"' in output
        assert '$ ai-knot search proof "what does the user deploy with?"' in output
        assert "$ ai-knot list proof" in output
        assert "$ ai-knot get proof" in output
        assert "$ ai-knot delete proof" in output
        assert "Demo complete. Data kept in" in output

        db_path = Path(data_dir) / "ai_knot.db"
        assert db_path.exists()

        kb = KnowledgeBase(agent_id="proof", storage=create_storage("sqlite", base_dir=data_dir))
        facts = kb.list_facts()
        assert len(facts) == 2
        assert any("TypeScript" in fact.content for fact in facts)
        assert any("Docker Compose" in fact.content for fact in facts)

    def test_demo_rejects_postgres(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main,
            ["--storage", "postgres", "--data-dir", data_dir, "demo"],
        )

        assert result.exit_code != 0
        assert "local-first proof path" in result.output


class TestCLIAddResolved:
    """ai-knot add-resolved <agent_id> <content>."""

    def test_add_resolved_stores_structured_update(self, runner: CliRunner, data_dir: str) -> None:
        initial = runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User works at Acme",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Acme",
                ],
            ),
        )
        assert initial.exit_code == 0
        assert "slot_key=user::employer" in initial.output
        assert "version=0" in initial.output

        updated = runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User now works at Globex",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Globex",
                    "--op",
                    "update",
                ],
            ),
        )
        assert updated.exit_code == 0
        assert "slot_key=user::employer" in updated.output
        assert "version=1" in updated.output
        assert "op=update" in updated.output

        kb = KnowledgeBase(agent_id="myagent", storage=create_storage("yaml", base_dir=data_dir))
        active = [
            fact
            for fact in kb.list_facts()
            if fact.is_active() and fact.slot_key == "user::employer"
        ]
        assert len(active) == 1
        assert active[0].value_text == "Globex"

    def test_add_resolved_delete_closes_slot(self, runner: CliRunner, data_dir: str) -> None:
        runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User works at Acme",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--value-text",
                    "Acme",
                ],
            ),
        )

        deleted = runner.invoke(
            main,
            _cmd(
                data_dir,
                [
                    "add-resolved",
                    "myagent",
                    "User no longer works at Acme",
                    "--entity",
                    "user",
                    "--attribute",
                    "employer",
                    "--slot-key",
                    "user::employer",
                    "--op",
                    "delete",
                ],
            ),
        )
        assert deleted.exit_code == 0
        assert "No replacement inserted." in deleted.output

        kb = KnowledgeBase(agent_id="myagent", storage=create_storage("yaml", base_dir=data_dir))
        active = [
            fact
            for fact in kb.list_facts()
            if fact.is_active() and fact.slot_key == "user::employer"
        ]
        assert active == []


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


class TestCLIGet:
    """ai-knot get <agent_id> <fact_id>."""

    def test_get_fact_by_id(self, runner: CliRunner, data_dir: str) -> None:
        add = runner.invoke(main, _cmd(data_dir, ["add", "agent", "Fact to inspect"]))
        fact_id = add.output.split()[2].rstrip(":")

        result = runner.invoke(main, _cmd(data_dir, ["get", "agent", fact_id]))
        assert result.exit_code == 0
        assert "Fact to inspect" in result.output
        assert f"id={fact_id}" in result.output
        assert "active=yes" in result.output

    def test_get_unknown_fact_is_non_fatal(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, _cmd(data_dir, ["get", "agent", "deadbeef"]))
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
        assert "mcp_clients" in payload
        assert "modules" in payload
        assert "env" in payload

    def test_doctor_reports_default_mcp_client_registration(
        self,
        runner: CliRunner,
        data_dir: str,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)

        claude_path = (
            tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        )
        claude_path.parent.mkdir(parents=True, exist_ok=True)
        claude_path.write_text(
            json.dumps({"mcpServers": {"ai-knot": {"command": "ai-knot-mcp"}}}),
            encoding="utf-8",
        )

        openclaw_path = tmp_path / ".openclaw" / "openclaw.json"
        openclaw_path.parent.mkdir(parents=True, exist_ok=True)
        openclaw_path.write_text(
            json.dumps({"mcpServers": {"other": {"command": "demo"}}}),
            encoding="utf-8",
        )

        result = runner.invoke(main, _cmd(data_dir, ["doctor", "--json"]))
        assert result.exit_code == 0

        payload = json.loads(result.output)
        assert payload["mcp_clients"]["claude"]["supported"] is True
        assert payload["mcp_clients"]["claude"]["exists"] is True
        assert payload["mcp_clients"]["claude"]["ai_knot_registered"] is True
        assert payload["mcp_clients"]["claude"]["registered_servers"] == ["ai-knot"]
        assert payload["mcp_clients"]["openclaw"]["supported"] is True
        assert payload["mcp_clients"]["openclaw"]["ai_knot_registered"] is False
        assert payload["mcp_clients"]["openclaw"]["registered_servers"] == ["other"]

    def test_doctor_reports_unsupported_claude_default_path_on_linux(
        self,
        runner: CliRunner,
        data_dir: str,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)

        result = runner.invoke(main, _cmd(data_dir, ["doctor", "--json"]))
        assert result.exit_code == 0

        payload = json.loads(result.output)
        assert payload["mcp_clients"]["claude"]["supported"] is False
        assert payload["mcp_clients"]["claude"]["default_config_path"] is None
        assert "macOS and Windows" in payload["mcp_clients"]["claude"]["error"]
        assert payload["mcp_clients"]["openclaw"]["supported"] is True

    def test_doctor_reports_unparseable_openclaw_config(
        self,
        runner: CliRunner,
        data_dir: str,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)

        openclaw_path = tmp_path / ".openclaw" / "openclaw.json"
        openclaw_path.parent.mkdir(parents=True, exist_ok=True)
        openclaw_path.write_text("{", encoding="utf-8")

        result = runner.invoke(main, _cmd(data_dir, ["doctor", "--json"]))
        assert result.exit_code == 0

        payload = json.loads(result.output)
        assert payload["mcp_clients"]["openclaw"]["exists"] is True
        assert payload["mcp_clients"]["openclaw"]["parseable"] is False
        assert payload["mcp_clients"]["openclaw"]["ai_knot_registered"] is None
        assert "Could not parse config as JSON/YAML" in payload["mcp_clients"]["openclaw"]["error"]

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

    def test_setup_openclaw_writes_new_config_file(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        config_path = tmp_path / "openclaw.json"

        result = runner.invoke(
            main,
            [
                "setup",
                "openclaw",
                "--agent-id",
                "bot",
                "--write-config",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        assert "Updated" in result.output
        assert "restart OpenClaw" in result.output
        assert "mcp_clients.openclaw.ai_knot_registered" in result.output
        assert "add -> search -> list -> delete" in result.output
        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["mcpServers"]["ai-knot"]["env"]["AI_KNOT_AGENT_ID"] == "bot"
        assert written["mcpServers"]["ai-knot"]["env"]["AI_KNOT_STORAGE"] == "sqlite"

    def test_setup_openclaw_writes_platform_default_config(
        self,
        runner: CliRunner,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)

        result = runner.invoke(
            main,
            [
                "setup",
                "openclaw",
                "--agent-id",
                "bot",
                "--write-default-config",
            ],
        )

        assert result.exit_code == 0
        assert "restart OpenClaw" in result.output
        config_path = tmp_path / ".openclaw" / "openclaw.json"
        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["mcpServers"]["ai-knot"]["env"]["AI_KNOT_AGENT_ID"] == "bot"

    def test_setup_claude_merges_existing_config_file(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        config_path = tmp_path / "claude.json"
        config_path.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "mcpServers": {
                        "existing": {
                            "command": "demo-mcp",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            [
                "setup",
                "claude",
                "--agent-id",
                "bot",
                "--write-config",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        merged = json.loads(config_path.read_text(encoding="utf-8"))
        assert merged["theme"] == "dark"
        assert merged["mcpServers"]["existing"]["command"] == "demo-mcp"
        assert merged["mcpServers"]["ai-knot"]["env"]["AI_KNOT_AGENT_ID"] == "bot"

    def test_setup_claude_writes_platform_default_config_on_macos(
        self,
        runner: CliRunner,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cli_module.Path, "home", lambda: tmp_path)

        result = runner.invoke(
            main,
            [
                "setup",
                "claude",
                "--agent-id",
                "bot",
                "--write-default-config",
            ],
        )

        assert result.exit_code == 0
        assert "restart Claude" in result.output
        assert "mcp_clients.claude.ai_knot_registered" in result.output
        config_path = (
            tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        )
        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["mcpServers"]["ai-knot"]["env"]["AI_KNOT_AGENT_ID"] == "bot"

    def test_setup_openclaw_rejects_non_object_mcp_servers(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        config_path = tmp_path / "openclaw.json"
        config_path.write_text(json.dumps({"mcpServers": []}), encoding="utf-8")

        result = runner.invoke(
            main,
            [
                "setup",
                "openclaw",
                "--write-config",
                str(config_path),
            ],
        )

        assert result.exit_code != 0
        assert "Expected 'mcpServers'" in result.output

    def test_setup_claude_rejects_write_default_config_on_unsupported_platform(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import ai_knot.cli as cli_module

        monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")

        result = runner.invoke(
            main,
            [
                "setup",
                "claude",
                "--write-default-config",
            ],
        )

        assert result.exit_code != 0
        assert "default config path is only known for macOS and Windows" in result.output

    def test_setup_rejects_both_write_flags(
        self,
        runner: CliRunner,
        tmp_path: pathlib.Path,
    ) -> None:
        result = runner.invoke(
            main,
            [
                "setup",
                "openclaw",
                "--write-default-config",
                "--write-config",
                str(tmp_path / "config.json"),
            ],
        )

        assert result.exit_code != 0
        assert "Use either --write-config or --write-default-config" in result.output


class TestCLIServeMCP:
    """ai-knot serve-mcp <agent_id>."""

    def test_serve_mcp_runs_streamable_http(
        self,
        runner: CliRunner,
        data_dir: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ai_knot import mcp_server as mcp_server_module

        called: dict[str, object] = {}

        class DummyApp:
            def run(self, transport: str) -> None:
                called["transport"] = transport

        def fake_make_server(*args: object, **kwargs: object) -> DummyApp:
            called["kwargs"] = kwargs
            return DummyApp()

        monkeypatch.setattr(mcp_server_module, "_make_server", fake_make_server)

        result = runner.invoke(
            main,
            ["--data-dir", data_dir, "serve-mcp", "bot", "--port", "8765", "--path", "/mcp"],
        )

        assert result.exit_code == 0
        assert "http://127.0.0.1:8765/mcp" in result.output
        assert called["transport"] == "streamable-http"
        assert called["kwargs"] == {
            "host": "127.0.0.1",
            "port": 8765,
            "streamable_http_path": "/mcp",
        }

    def test_serve_mcp_rejects_path_without_leading_slash(
        self, runner: CliRunner, data_dir: str
    ) -> None:
        result = runner.invoke(
            main,
            ["--data-dir", data_dir, "serve-mcp", "bot", "--path", "mcp"],
        )

        assert result.exit_code != 0
        assert "path must start with '/'" in result.output


class TestAuditExport:
    """ai-knot audit-export dumps the governance ledger as JSON."""

    def test_audit_export_emits_trust_usage_and_grants(
        self, runner: CliRunner, data_dir: str
    ) -> None:
        from ai_knot.knowledge import SharedMemoryPool

        storage = create_storage("sqlite", base_dir=data_dir, dsn=None)
        pool = SharedMemoryPool(storage=storage, persist_stats=True)
        pool.register("planner")
        pool.register("coder")
        kb = KnowledgeBase("planner", storage=storage, embed_url="")
        fact = Fact(
            content="Deploy target is GKE",
            entity="prod",
            attribute="deploy",
            slot_key="prod::deploy",
            value_text="gke",
            topic_channel="arch",
        )
        kb.replace_facts([fact])
        pool.publish("planner", [fact.id], kb=kb)
        pool.grant_read("coder", "team:secops")
        pool.recall("deploy target?", "coder", top_k=3, topic_channel="arch")
        pool.flush_stats()

        result = runner.invoke(
            main, ["--storage", "sqlite", "--data-dir", data_dir, "audit-export"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert any(
            e["event_type"] == "publish" and e["agent_id"] == "planner"
            for e in payload["trust_events"]
        )
        assert any(e["agent_id"] == "coder" for e in payload["usage_events"])
        assert payload["grants"]["coder"] == ["team:secops"]

    def test_audit_export_empty_ledger_is_valid_json(
        self, runner: CliRunner, data_dir: str
    ) -> None:
        result = runner.invoke(
            main, ["--storage", "sqlite", "--data-dir", data_dir, "audit-export"]
        )
        assert result.exit_code == 0
        assert '"trust_events": []' in result.output
        assert '"usage_events": []' in result.output


class TestEmbedUrlEnv:
    """The CLI honours AI_KNOT_EMBED_URL so serve / the container can go BM25-only."""

    def test_create_kb_disables_dense_when_env_empty(
        self, monkeypatch: pytest.MonkeyPatch, data_dir: str
    ) -> None:
        from ai_knot.cli import _create_kb

        monkeypatch.setenv("AI_KNOT_EMBED_URL", "")
        kb = _create_kb(storage_backend="yaml", data_dir=data_dir, dsn=None, agent_id="a")
        assert kb._embed_url == ""

    def test_create_kb_is_offline_by_default_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch, data_dir: str
    ) -> None:
        from ai_knot.cli import _create_kb

        # No AI_KNOT_EMBED_URL -> BM25-only: no embedding server, no network call.
        monkeypatch.delenv("AI_KNOT_EMBED_URL", raising=False)
        kb = _create_kb(storage_backend="yaml", data_dir=data_dir, dsn=None, agent_id="a")
        assert kb._embed_url == ""
