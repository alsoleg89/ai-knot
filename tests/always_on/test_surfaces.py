import asyncio

from fastapi.testclient import TestClient

from ai_knot._always_on_tools import invoke_always_on
from ai_knot.server.app import create_app


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
