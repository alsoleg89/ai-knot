"""Tests for CLI error paths — invalid inputs, missing files, bad YAML."""

from __future__ import annotations

import pathlib

import yaml
from click.testing import CliRunner

from ai_knot.cli import main


def _cmd(data_dir: str, args: list[str]) -> list[str]:
    """Prefix args with --data-dir (group option must come before subcommand)."""
    return ["--data-dir", data_dir, *args]


class TestAddValidation:
    """ai-knot add — input validation."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_importance_above_1_rejected(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(
            main, _cmd(str(tmp_path), ["add", "agent", "some fact", "--importance", "1.5"])
        )
        assert result.exit_code != 0
        assert "1.0" in result.output or "importance" in result.output.lower()

    def test_importance_below_0_rejected(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(
            main, _cmd(str(tmp_path), ["add", "agent", "some fact", "--importance", "-0.1"])
        )
        assert result.exit_code != 0

    def test_empty_content_rejected(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["add", "agent", "   "]))
        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "content" in result.output.lower()

    def test_valid_importance_boundary_0(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(
            main, _cmd(str(tmp_path), ["add", "agent", "minimal fact", "--importance", "0.0"])
        )
        assert result.exit_code == 0

    def test_valid_importance_boundary_1(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(
            main, _cmd(str(tmp_path), ["add", "agent", "critical fact", "--importance", "1.0"])
        )
        assert result.exit_code == 0


class TestImportValidation:
    """ai-knot import — file and YAML validation."""

    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_missing_file_rejected(self, tmp_path: pathlib.Path) -> None:
        result = self.runner.invoke(
            main, _cmd(str(tmp_path), ["import", "agent", "/nonexistent/path.yaml"])
        )
        assert result.exit_code != 0

    def test_invalid_yaml_rejected(self, tmp_path: pathlib.Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket", encoding="utf-8")
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["import", "agent", str(bad_yaml)]))
        assert result.exit_code != 0
        assert "yaml" in result.output.lower() or "invalid" in result.output.lower()

    def test_non_mapping_yaml_rejected(self, tmp_path: pathlib.Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["import", "agent", str(list_yaml)]))
        assert result.exit_code != 0
        assert "mapping" in result.output.lower()

    def test_empty_yaml_is_ok(self, tmp_path: pathlib.Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["import", "agent", str(empty_yaml)]))
        assert result.exit_code == 0
        assert "No facts" in result.output

    def test_fact_missing_required_field(self, tmp_path: pathlib.Path) -> None:
        data = {
            "fact1": {
                # missing 'content' and 'created_at'
                "type": "semantic",
                "importance": 0.8,
            }
        }
        yaml_file = tmp_path / "missing_field.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["import", "agent", str(yaml_file)]))
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "field" in result.output.lower()

    def test_invalid_type_value_rejected(self, tmp_path: pathlib.Path) -> None:
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
        result = self.runner.invoke(main, _cmd(str(tmp_path), ["import", "agent", str(yaml_file)]))
        assert result.exit_code != 0

    def test_valid_import_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Export then re-import should preserve fact count."""
        runner = self.runner
        dd = str(tmp_path)
        runner.invoke(main, _cmd(dd, ["add", "agent", "Python is great"]))
        runner.invoke(main, _cmd(dd, ["add", "agent", "Uses Docker"]))

        export_file = tmp_path / "export.yaml"
        runner.invoke(main, _cmd(dd, ["export", "agent", str(export_file)]))

        import_dir = tmp_path / "imported"
        import_dir.mkdir()
        result = runner.invoke(main, _cmd(str(import_dir), ["import", "agent2", str(export_file)]))
        assert result.exit_code == 0
        assert "2" in result.output
