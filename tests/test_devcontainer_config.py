"""Regression checks for the install-free Codespaces/devcontainer path."""

from __future__ import annotations

import json
from pathlib import Path


def test_devcontainer_has_python_node_and_postcreate_contract() -> None:
    config_path = Path(__file__).resolve().parent.parent / ".devcontainer" / "devcontainer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["image"] == "mcr.microsoft.com/devcontainers/python:1-3.12-bullseye"

    node_feature = config["features"]["ghcr.io/devcontainers/features/node:1"]
    assert node_feature["version"] == "22"

    post_create = config["postCreateCommand"]
    assert 'pip install -e ".[dev,mcp,postgres,server]"' in post_create
    assert "cd npm" in post_create
    assert "AI_KNOT_SKIP_PYTHON_INSTALL=1 npm ci" in post_create

    remote_env = config["remoteEnv"]
    assert remote_env["AI_KNOT_SKIP_PYTHON_INSTALL"] == "1"
