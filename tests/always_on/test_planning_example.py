"""The runnable host workflow uses both SQL engines with independent facades."""

from unittest.mock import patch

import pytest
from examples.always_on_planning import example


@pytest.mark.anyio
async def test_planning_example(sql_storage, tmp_path):
    address = getattr(sql_storage, "_db_path", None) or sql_storage._dsn
    with (
        patch("ai_knot.embedder.embed_texts", side_effect=AssertionError("embedding call")),
        patch("httpx.Client.send", side_effect=AssertionError("HTTP call")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("HTTP call")),
    ):
        result = await example(tmp_path, make_store=lambda: type(sql_storage)(address))
    assert result["initial_inspection"] == "missing_context"
    assert len(result["initial_missing"]) == 2
    assert [p["status"] for p in result["proposals"]] == ["waiting", "planned"]
    assert [p["model_calls_reserved"] for p in result["proposals"]] == [1, 1]
    assert result["quiet_statuses"] == ["idle"] * 5
    assert result["quiet_model_calls_reserved"] == result["repeat_model_calls_reserved"] == 0
    assert result["repeat_status"] == "unchanged"
    assert result["effect_count"] == 1 and result["unknown_model_calls_reserved"] == 0
    assert result["reconciliation_status"] == "reconcile"
    assert result["final_state"] == "completed"
