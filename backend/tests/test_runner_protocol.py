import sys

from circuit_api.runner import _run_bounded


def test_worker_result_protocol_parses_submission_and_output():
    script = 'print("__CIRCUIT_RESULT__={\\"submission\\":{\\"components\\":[]},\\"events\\":[],\\"stdout\\":\\"hello\\"}")'
    result = _run_bounded([sys.executable, "-c", script], timeout=3)

    assert result["status"] == "complete"
    assert result["submission"] == {"components": []}
    assert result["stdout"] == "hello"


def test_worker_timeout_is_reported():
    result = _run_bounded([sys.executable, "-c", "import time; time.sleep(3)"], timeout=1)

    assert result["status"] == "error"
    assert "time limit" in result["error"]


def test_worker_output_limit_is_reported():
    result = _run_bounded([sys.executable, "-c", "print('x' * 1_100_000)"], timeout=3)

    assert result["status"] == "error"
    assert "output exceeded" in result["error"]
