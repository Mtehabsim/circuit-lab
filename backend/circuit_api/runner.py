from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
TIME_LIMIT_SECONDS = int(os.environ.get("CIRCUIT_RUN_TIMEOUT", "20"))
OUTPUT_LIMIT_BYTES = 1_000_000
CHECKPOINT_REVISION = 2


class RunnerUnavailable(RuntimeError):
    pass


def run_code(source: str, challenge_id: str, challenge_data: dict[str, Any]) -> dict[str, Any]:
    if Path(challenge_id).name != challenge_id:
        raise ValueError("invalid challenge id")
    checkpoint = _ensure_checkpoint(challenge_data.get("model", {}).get("name", "prev-token-toy"))
    jobs = ROOT / "var" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=jobs) as temp_dir:
        job_dir = Path(temp_dir)
        source_path = job_dir / "solution.py"
        source_path.write_text(source, encoding="utf-8")
        public_challenge = {key: value for key, value in challenge_data.items() if key != "hidden_dataset"}
        (job_dir / "challenge.json").write_text(json.dumps(public_challenge), encoding="utf-8")
        env = _worker_environment(job_dir, checkpoint)
        env["CIRCUIT_CHALLENGE"] = challenge_id
        return _run_bounded(
            [sys.executable, str(ROOT / "worker" / "entrypoint.py")],
            TIME_LIMIT_SECONDS,
            env=env,
            cwd=job_dir,
        )


def run_grade(payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _ensure_checkpoint(payload.get("model_name", "prev-token-toy"))
    jobs = ROOT / "var" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"grade-{uuid.uuid4().hex}-", dir=jobs) as temp_dir:
        job_dir = Path(temp_dir)
        (job_dir / "grading.json").write_text(json.dumps(payload), encoding="utf-8")
        result = _run_bounded(
            [sys.executable, str(ROOT / "worker" / "grader_entrypoint.py")],
            TIME_LIMIT_SECONDS,
            env=_worker_environment(job_dir, checkpoint),
            cwd=job_dir,
        )
    if result.get("status") != "complete":
        raise RunnerUnavailable(result.get("error", "The local grading process failed."))
    if result.get("error") or not isinstance(result.get("grade"), dict):
        raise RunnerUnavailable(result.get("error", "The local grader returned an invalid result."))
    return result["grade"]


def _ensure_checkpoint(model_name: str) -> Path:
    if Path(model_name).name != model_name:
        raise ValueError("invalid checkpoint model name")
    checkpoint = ROOT / "var" / "models" / f"{model_name}-v{CHECKPOINT_REVISION}.pt"
    if checkpoint.is_file():
        return checkpoint
    try:
        from .model import build_checkpoint, build_induction_checkpoint, build_ioi_checkpoint
    except ImportError as error:
        raise RunnerUnavailable("PyTorch is required. Install backend dependencies with `uv sync --extra dev`.") from error
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    builders = {"prev-token-toy": build_checkpoint, "decoy-head-toy": build_checkpoint,
                "ov-path-toy": build_checkpoint, "patching-toy": build_checkpoint,
                "induction-toy": build_induction_checkpoint, "ioi-toy": build_ioi_checkpoint}
    builder = builders.get(model_name)
    if builder is None:
        raise RunnerUnavailable(f"No checkpoint builder is registered for {model_name}.")
    builder(str(checkpoint))
    return checkpoint


def _worker_environment(job_dir: Path, checkpoint: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env["CIRCUIT_JOB_DIR"] = str(job_dir)
    env["CIRCUIT_CHECKPOINT"] = str(checkpoint)
    return env


def _run_bounded(
    command: list[str],
    timeout: int,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    started = time.monotonic()
    failure = None
    while selector.get_map():
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            failure = f"Execution exceeded the {timeout} second time limit."
            _kill_process_group(process)
            break
        for key, _ in selector.select(min(remaining, 0.25)):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            target = output[key.data]
            if len(target) + len(chunk) > OUTPUT_LIMIT_BYTES:
                failure = "Execution output exceeded the 1 MB limit."
                _kill_process_group(process)
                break
            target.extend(chunk)
        if failure:
            break
        if process.poll() is not None and not selector.get_map():
            break
    if failure:
        _kill_process_group(process)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        process.wait()
    stdout = output["stdout"].decode("utf-8", errors="replace")
    stderr = output["stderr"].decode("utf-8", errors="replace")
    if failure:
        return {"status": "error", "error": failure, "stdout": stdout, "stderr": stderr}
    if process.returncode != 0:
        return {"status": "error", "error": "Python execution failed.", "stdout": stdout, "stderr": stderr}
    marker = "__CIRCUIT_RESULT__="
    result_line = next((line for line in reversed(stdout.splitlines()) if line.startswith(marker)), None)
    if result_line is None:
        return {"status": "error", "error": "Worker completed without returning a result.", "stdout": stdout, "stderr": stderr}
    try:
        result = json.loads(result_line[len(marker):])
    except json.JSONDecodeError:
        return {"status": "error", "error": "Worker returned invalid result data.", "stdout": stdout, "stderr": stderr}
    user_stdout = result.get("stdout", "") or "\n".join(line for line in stdout.splitlines() if not line.startswith(marker))
    status = "error" if result.get("error") else "complete"
    return {"status": status, **result, "stdout": user_stdout, "stderr": stderr}


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
