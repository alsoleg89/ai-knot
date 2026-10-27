import json

from click.testing import CliRunner

from ai_knot.cli import main


def test_transport_retry_keeps_identity(tmp_path):
    runner = CliRunner()
    prefix = ["--storage", "sqlite", "--data-dir", str(tmp_path)]
    command = ["work", "create", "owner", "--client-key", "k", "--goal", "g", "--json"]
    first = runner.invoke(main, prefix + command)
    second = runner.invoke(main, prefix + command)
    assert first.exit_code == second.exit_code == 0, first.output
    assert json.loads(first.output)["id"] == json.loads(second.output)["id"]
    bad = runner.invoke(
        main, prefix + ["work", "checkpoint", "owner", "missing", "--json"], input='{"state":NaN}'
    )
    assert bad.exit_code != 0
    assert (
        len(json.loads(runner.invoke(main, prefix + ["work", "list", "owner", "--json"]).output))
        == 1
    )


def test_yaml_command_reports_supported_backends(tmp_path):
    result = CliRunner().invoke(
        main, ["--storage", "yaml", "--data-dir", str(tmp_path), "work", "list", "owner", "--json"]
    )
    assert result.exit_code != 0
    assert "SQLite or PostgreSQL" in result.output


def test_cli_changes_and_maintenance(tmp_path):
    runner = CliRunner()
    prefix = ["--storage", "sqlite", "--data-dir", str(tmp_path)]
    assert (
        runner.invoke(main, prefix + ["changes", "subscribe", "owner", "watch", "--json"]).exit_code
        == 0
    )
    batch = runner.invoke(main, prefix + ["changes", "poll", "owner", "watch", "--json"])
    assert json.loads(batch.output)["delivery_id"] is None
    assert (
        runner.invoke(
            main, prefix + ["changes", "unsubscribe", "owner", "watch", "--json"]
        ).exit_code
        == 0
    )
    report = runner.invoke(
        main, prefix + ["maintenance", "run", "owner", "--run-key", "r", "--json"]
    )
    assert json.loads(report.output)["model_calls_reserved"] == 0
