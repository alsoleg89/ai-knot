import pytest


@pytest.mark.parametrize(
    "receipt,positive,expected",
    [
        ({"status": "pass", "tests_run": 5, "failures": 0, "errors": 0}, True, True),
        ({"status": "fail", "tests_run": 5, "failures": 1, "errors": 0}, False, True),
        ({"status": "fail", "tests_run": 5, "failures": 0, "errors": 1}, False, True),
        ({"status": "pass", "tests_run": 0, "failures": 0, "errors": 0}, True, False),
        ({"status": "pass", "tests_run": 4, "failures": 0, "errors": 0}, True, False),
        ({"status": "timeout", "tests_run": 5, "failures": 1, "errors": 0}, False, False),
        ({"status": "fail", "tests_run": 5, "failures": 0, "errors": 0}, False, False),
        ({"status": "fail", "tests_run": 5, "failures": True, "errors": 0}, False, False),
    ],
)
def test_controls_do_not_accept_incomplete_or_infrastructure_failure(receipt, positive, expected):
    from tests.eval.team_work.controls import receipt_matches

    assert receipt_matches(receipt, expected_pass=positive, expected_tests=5) is expected
