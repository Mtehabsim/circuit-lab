from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from circuit_api.model import ModelAdapter, load_model, make_tokenizer
from circuit_api.tools import ExperimentTools


CHECKPOINT = Path(os.environ.get("CIRCUIT_CHECKPOINT", "/models/previous-token.pt"))
RESULT_MARKER = "__CIRCUIT_RESULT__="
MAX_CAPTURED_OUTPUT = 64_000


class CappedOutput(io.StringIO):
    def write(self, value: str) -> int:
        remaining = MAX_CAPTURED_OUTPUT - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        return len(value)


def load_challenge(challenge_id: str) -> dict[str, Any]:
    if Path(challenge_id).name != challenge_id:
        raise ValueError("invalid challenge id")
    path = Path(os.environ.get("CIRCUIT_JOB_DIR", "/job")) / "challenge.json"
    if not path.is_file():
        raise ValueError("challenge data was not mounted")
    challenge = json.loads(path.read_text())
    if challenge.get("id") != challenge_id or "hidden_dataset" in challenge:
        raise ValueError("invalid public challenge data")
    return challenge


def main() -> None:
    user_output = CappedOutput()
    result: dict[str, Any]
    try:
        challenge_id = os.environ.get("CIRCUIT_CHALLENGE", "")
        challenge = load_challenge(challenge_id)
        model_name = challenge.get("model", {}).get("name", "prev-token-toy")
        model = ModelAdapter(load_model(model_name, str(CHECKPOINT)))
        tokenizer = make_tokenizer(model_name, model.d_model)
        dataset = [{"tokens": list(tokens)} for tokens in challenge["public_dataset"]]
        tools = ExperimentTools(model)
        namespace = {"model": model, "tokenizer": tokenizer, "dataset": dataset, "tools": tools}
        source = (Path(os.environ.get("CIRCUIT_JOB_DIR", "/job")) / "solution.py").read_text(encoding="utf-8")
        with contextlib.redirect_stdout(user_output):
            code = compile(source, "solution.py", "exec")
            exec(code, namespace, namespace)
            solve = namespace.get("solve")
            if not callable(solve):
                raise ValueError("Define solve(model, tokenizer, dataset, tools) in solution.py")
            submission = solve(model, tokenizer, dataset, tools)
        if not isinstance(submission, dict):
            raise ValueError("solve(...) must return a dictionary submission")
        encoded = json.dumps(submission, allow_nan=False)
        if len(encoded) > 64_000:
            raise ValueError("Submission data exceeds the 64 KB limit")
        result = {
            "result_version": 1,
            "submission": submission,
            "events": tools.events,
            "traces": tools.traces,
            "baseline": tools.evaluate(dataset),
        }
    except BaseException as error:
        result = {
            "result_version": 1,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
            "events": [],
            "traces": {},
            "baseline": None,
        }
    stdout = user_output.getvalue()[-MAX_CAPTURED_OUTPUT:]
    result["stdout"] = stdout
    print(RESULT_MARKER + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
