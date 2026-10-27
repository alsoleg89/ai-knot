import asyncio

import pytest
from fastapi.testclient import TestClient

from ai_knot._always_on_tools import invoke_always_on
from ai_knot.always_on import WatchSelector
from ai_knot.server.app import create_app


@pytest.mark.parametrize("watch", ["watch", "watch/team"])
def test_http_snapshot_and_context_are_strict_and_preserve_delivery(ao_factory, kb, watch):
    ao = ao_factory()
    client = TestClient(create_app(kb, token="secret", always_on=ao))
    headers = {"Authorization": "Bearer secret"}
    ao.subscribe(subscription_id=watch, selector=WatchSelector("owner"))
    fact = kb.add("Production uses PostgreSQL")
    base = "/v1/always-on/subscriptions/" + watch.replace("/", "%2F")
    path = base + "/resync"
    assert client.post(path).status_code == 401
    response = client.post(path, headers=headers, json={})
    assert response.status_code == 200, response.text
    snapshot = response.json()
    assert snapshot["is_snapshot"] and snapshot["through_seq"] == snapshot["head_seq"]
    assert snapshot["events"][0]["fact_id"] == fact.id
    ref = {
        "namespace": "owner",
        "fact_id": fact.id,
        "revision": snapshot["events"][0]["after_revision"],
    }
    review = client.post("/v1/always-on/evidence/review", headers=headers, json={"refs": [ref]})
    assert review.status_code == 200 and review.json()["complete"]
    assert review.json()["accepted"][0]["content"] == "Production uses PostgreSQL"
    assert (
        client.post(
            "/v1/always-on/evidence/review",
            headers=headers,
            json={"refs": [ref], "owner_agent_id": "peer"},
        ).status_code
        == 422
    )
    assert client.post(path, headers=headers, json={"limit": True}).status_code == 422
    kb.add("A second memory")
    overflow = client.post(path, headers=headers, json={"limit": 1})
    assert overflow.status_code == 413 and overflow.json()["error"]["code"] == "SnapshotTooLarge"
    assert ao.poll(watch).delivery_id == snapshot["delivery_id"]
    replay = client.post(base + "/poll", headers=headers, json={})
    assert replay.status_code == 200 and replay.json()["is_snapshot"]
    assert (
        client.post(
            base + "/ack", headers=headers, json={"delivery_id": snapshot["delivery_id"]}
        ).status_code
        == 204
    )
    assert client.delete(base, headers=headers).status_code == 204


def test_http_task_context_returns_truthful_bounded_requirements(ao_factory, kb):
    ao = ao_factory()
    kb.add("Rollback checkpoint verified")
    client = TestClient(create_app(kb, token="secret", always_on=ao))
    path = "/v1/always-on/evidence/select"
    body = {"requirements": ["rollback checkpoint", "certificate expiry"], "max_chars": 40}
    assert client.post(path, json=body).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    response = client.post(path, headers=headers, json=body)
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["covered_requirements"] == ["rollback checkpoint"]
    assert view["missing_requirements"] == ["certificate expiry"] and not view["complete"]
    assert sum(len(item["content"]) for item in view["facts"]) <= 40
    assert client.post(path, headers=headers, json={**body, "max_chars": True}).status_code == 422
    assert (
        client.post(path, headers=headers, json={**body, "owner_agent_id": "peer"}).status_code
        == 422
    )


def test_actor_cannot_be_spoofed(ao_factory, kb):
    client = TestClient(create_app(kb, token="secret", always_on=ao_factory()))
    path = "/v1/always-on/intents"
    assert client.post(path, json={"client_key": "k", "goal": "g"}).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    assert (
        client.post(
            path,
            headers=headers,
            json={"client_key": "k", "goal": "g", "owner_agent_id": "foreign"},
        ).status_code
        == 422
    )
    first = client.post(path, headers=headers, json={"client_key": "k", "goal": "g"})
    assert first.status_code == 200, first.text
    assert (
        client.post(path, headers=headers, json={"client_key": "k", "goal": "g"}).json()
        == first.json()
    )
    claim = client.post(
        path + "/" + first.json()["id"] + "/claim", headers=headers, json={"worker_id": "w"}
    )
    assert claim.status_code == 200, claim.text


def test_http_and_shared_invocation_checkpoint(ao_factory, kb):
    ao = ao_factory()
    client = TestClient(create_app(kb, always_on=ao))
    intent = asyncio.run(invoke_always_on(ao, "create_intent", {"client_key": "a", "goal": "b"}))
    lease = client.post(
        f"/v1/always-on/intents/{intent['id']}/claim", json={"worker_id": "host"}
    ).json()
    current = client.get(f"/v1/always-on/intents/{intent['id']}").json()
    response = client.post(
        f"/v1/always-on/intents/{intent['id']}/checkpoint",
        json={
            "lease_token": lease["token"],
            "expected_revision": current["revision"],
            "state": "waiting",
            "progress": {"arbitrary": [1, "x"]},
            "waiting_for": {"kind": "until", "at": "2026-10-28T00:00:00Z"},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["waiting_for"]["kind"] == "until"
    assert (
        client.post(
            "/v1/always-on/maintenance/run", json={"run_key": "http", "budget": {}}
        ).status_code
        == 200
    )


def test_strict_raw_json_and_no_legacy_routes(ao_factory, kb):
    assert TestClient(create_app(kb)).get("/v1/always-on/intents").status_code == 404
    client = TestClient(create_app(kb, always_on=ao_factory()))
    for raw in ['{"client_key":"a","goal":"b","goal":"c"}', '{"client_key":"a","goal":NaN}']:
        response = client.post("/v1/always-on/intents", content=raw)
        assert response.status_code == 422
        assert (
            "a" not in response.json()["error"]["message"]
            or response.json()["error"]["code"] == "InvalidInput"
        )
    response = client.get("/v1/always-on/intents/missing")
    assert response.status_code == 403
