"""Tests for the optional HTTP sidecar (ai_knot.server)."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("fastapi")  # server extra; always present in the dev/CI env

from fastapi.testclient import TestClient  # noqa: E402

from ai_knot import __version__  # noqa: E402
from ai_knot.knowledge import KnowledgeBase  # noqa: E402
from ai_knot.server import create_app  # noqa: E402
from ai_knot.storage.sqlite_storage import SQLiteStorage  # noqa: E402


def _kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="svc", storage=SQLiteStorage(db_path=str(tmp_path / "kb.db")))


def _client(tmp_path: pathlib.Path, token: str | None = None) -> TestClient:
    return TestClient(create_app(_kb(tmp_path), token=token))


def test_health_is_open_and_reports_version(tmp_path: pathlib.Path) -> None:
    r = _client(tmp_path).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_add_then_recall(tmp_path: pathlib.Path) -> None:
    client = _client(tmp_path)
    add = client.post("/v1/facts", json={"content": "User deploys on Kubernetes", "tags": ["ops"]})
    assert add.status_code == 201
    body = add.json()
    assert body["content"] == "User deploys on Kubernetes"
    assert body["type"] == "semantic"
    assert "ops" in body["tags"]

    rec = client.post("/v1/recall", json={"query": "where does the user deploy", "top_k": 5})
    assert rec.status_code == 200
    out = rec.json()
    assert "Kubernetes" in out["context"]
    assert any("Kubernetes" in f["content"] for f in out["facts"])


def test_recall_threads_now_anchor(tmp_path: pathlib.Path) -> None:
    # A just-added fact (valid_from = ingest time) is not active before that time.
    client = _client(tmp_path)
    client.post("/v1/facts", json={"content": "User lives in Berlin"})
    past = client.post(
        "/v1/recall", json={"query": "where does the user live", "now": "2000-01-01"}
    )
    assert past.status_code == 200
    assert past.json()["facts"] == []  # now in the past → fact not yet active
    live = client.post("/v1/recall", json={"query": "where does the user live"})
    assert any("Berlin" in f["content"] for f in live.json()["facts"])


def test_add_rejects_invalid_type(tmp_path: pathlib.Path) -> None:
    r = _client(tmp_path).post("/v1/facts", json={"content": "x", "type": "bogus"})
    assert r.status_code == 422
    assert "invalid memory type" in r.json()["detail"]


def test_recall_rejects_bad_now(tmp_path: pathlib.Path) -> None:
    r = _client(tmp_path).post("/v1/recall", json={"query": "x", "now": "not-a-date"})
    assert r.status_code == 422
    assert "ISO-8601" in r.json()["detail"]


def test_recall_validates_empty_query(tmp_path: pathlib.Path) -> None:
    r = _client(tmp_path).post("/v1/recall", json={"query": ""})
    assert r.status_code == 422  # pydantic min_length


def test_stats(tmp_path: pathlib.Path) -> None:
    client = _client(tmp_path)
    client.post("/v1/facts", json={"content": "a fact"})
    r = client.get("/v1/stats")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_bearer_token_guards_v1_but_not_health(tmp_path: pathlib.Path) -> None:
    client = _client(tmp_path, token="s3cret")
    # /health stays open.
    assert client.get("/health").status_code == 200
    # /v1/* without the token is rejected.
    assert client.post("/v1/recall", json={"query": "x"}).status_code == 401
    assert client.get("/v1/stats").status_code == 401
    # ...and accepted with the right bearer token.
    ok = client.get("/v1/stats", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    bad = client.get("/v1/stats", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
