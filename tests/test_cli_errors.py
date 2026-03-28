"""Tests for CLI error paths — invalid inputs, missing files, bad YAML."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from click.testing import CliRunner

from agentmemo.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


class TestAddValidation:
    """agentmemo add — input validation."""

    def test_importance_above_1_rejected(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["add", "agent", "some fact", "--importance", "1.5", "--data-dir", data_dir]
        )
        assert result.exit_code != 0
        assert "1.0" in result.output or "importance" in result.output.lower()

    def test_importance_below_0_rejected(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["add", "agent", "some fact", "--importance", "-0.1", "--data-dir", data_dir]
        )
        assert result.exit_code != 0

    def test_empty_content_rejected(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(main, ["add", "agent", "   ", "--data-dir", data_dir])
        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "content" in result.output.lower()

    def test_valid_importance_boundary_0(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["add", "agent", "minimal fact", "--importance", "0.0", "--data-dir", data_dir]
        )
        assert result.exit_code == 0

    def test_valid_importance_boundary_1(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["add", "agent", "critical fact", "--importance", "1.0", "--data-dir", data_dir]
        )
        assert result.exit_code == 0


class TestImportValidation:
    """agentmemo import — file and YAML validation."""

    def test_missing_file_rejected(self, runner: CliRunner, data_dir: str) -> None:
        result = runner.invoke(
            main, ["import", "agent", "/nonexistent/path.yaml", "--data-dir", data_dir]
        )
        assert result.exit_code != 0

    def test_invalid_yaml_rejected(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket", encoding="utf-8")
        result = runner.invoke(
            main, ["import", "agent", str(bad_yaml), "--data-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "yaml" in result.output.lower() or "invalid" in result.output.lower()

    def test_non_mapping_yaml_rejected(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        result = runner.invoke(
            main, ["import", "agent", str(list_yaml), "--data-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "mapping" in result.output.lower()

    def test_empty_yaml_is_ok(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        result = runner.invoke(
            main, ["import", "agent", str(empty_yaml), "--data-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "No facts" in result.output

    def test_fact_missing_required_field(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        data = {
            "fact1": {
                # missing 'content' and 'created_at'
                "type": "semantic",
                "importance": 0.8,
            }
        }
        yaml_file = tmp_path / "missing_field.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        result = runner.invoke(
            main, ["import", "agent", str(yaml_file), "--data-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "field" in result.output.lower()

    def test_invalid_type_value_rejected(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        from datetime import UTC, datetime

        data = {
            "fact1": {
                "content": "some fact",
                "type": "invalid_type",
                "importance": 0.8,
                "created_at": datetime.now(UTC).isoformat(),
            }
        }
        yaml_file = tmp_path / "bad_type.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        result = runner.invoke(
            main, ["import", "agent", str(yaml_file), "--data-dir", str(tmp_path)]
        )
        assert result.exit_code != 0

    def test_valid_import_roundtrip(self, runner: CliRunner, tmp_path: pathlib.Path) -> None:
        """Export then re-import should preserve fact count."""
        data_dir = str(tmp_path)
        runner.invoke(main, ["add", "agent", "Python is great", "--data-dir", data_dir])
        runner.invoke(main, ["add", "agent", "Uses Docker", "--data-dir", data_dir])

        export_file = tmp_path / "export.yaml"
        runner.invoke(main, ["export", "agent", str(export_file), "--data-dir", data_dir])

        import_dir = tmp_path / "imported"
        import_dir.mkdir()
        result = runner.invoke(
            main,
            ["import", "agent2", str(export_file), "--data-dir", str(import_dir)],
        )
        assert result.exit_code == 0
        assert "2" in result.output
