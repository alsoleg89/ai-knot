from datetime import UTC, datetime

import pytest

from ai_knot.always_on.codec import decode_value, encode_always_on, loads_strict
from ai_knot.always_on.models import EvidenceRef, WaitUntil


def test_strict_json_rejects_duplicates_and_nonfinite():
    for raw in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
        with pytest.raises(ValueError):
            loads_strict(raw)


def test_encode_wait_and_normalize_dates():
    assert encode_always_on(WaitUntil(datetime(2026, 1, 1, tzinfo=UTC))) == {
        "kind": "until",
        "at": "2026-01-01T00:00:00+00:00",
    }


def test_strict_nested_wire_values():
    with pytest.raises(ValueError):
        decode_value({"namespace": "x", "fact_id": "y", "revision": "r", "owner": "z"}, EvidenceRef)
    with pytest.raises(ValueError):
        decode_value(True, int)
    with pytest.raises(ValueError):
        decode_value("2026-01-01", datetime)
    assert decode_value("2026-01-01T03:00:00+03:00", datetime) == datetime(2026, 1, 1, tzinfo=UTC)


def test_wait_discriminator_and_nested_json_bound():
    from ai_knot.always_on.models import VerificationResult, WaitCondition

    with pytest.raises(ValueError):
        decode_value({"at": "2026-01-01T00:00:00Z"}, WaitCondition)
    with pytest.raises(ValueError):
        decode_value(
            {"outcome": "confirmed", "evidence": {"text": "я" * 40000}, "source_version": ""},
            VerificationResult,
        )
