from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from circuit_api.grader import grade_ablation_control, grade_activation_patch, grade_induction, grade_ioi, grade_previous_token, grade_value_path


GRADERS = {
    "previous-token-head": grade_previous_token,
    "necessary-or-noise": grade_ablation_control,
    "what-gets-written": grade_value_path,
    "find-the-signal": grade_activation_patch,
    "complete-the-pattern": grade_induction,
    "who-gets-the-book": grade_ioi,
}


def main() -> None:
    result = {"result_version": 1}
    try:
        job_dir = Path(os.environ.get("CIRCUIT_JOB_DIR", "/job"))
        payload = json.loads((job_dir / "grading.json").read_text(encoding="utf-8"))
        grader = GRADERS.get(payload.get("challenge_id"))
        if grader is None:
            raise ValueError("unknown challenge grader")
        result["grade"] = grader(
            payload["claim"], payload["run_result"], payload["public_dataset"], payload["hidden_dataset"],
            os.environ.get("CIRCUIT_CHECKPOINT", "/models/previous-token.pt")
        )
    except BaseException as error:
        result = {"result_version": 1, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=8)}
    print("__CIRCUIT_RESULT__=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
