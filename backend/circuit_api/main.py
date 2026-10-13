from __future__ import annotations

from contextlib import asynccontextmanager
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db
from .challenge_store import get_challenge, list_challenges, validate_challenges
from .runner import RunnerUnavailable, run_code, run_grade


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.initialize()
    validate_challenges()
    yield


app = FastAPI(title="Circuit Challenge API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


class RunRequest(BaseModel):
    challenge_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=100_000)


class SubmitRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=80)


class DraftRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    source: str = Field(max_length=100_000)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "circuit-api"}


@app.get("/api/challenges")
def challenges() -> list[dict[str, Any]]:
    return list_challenges()


@app.get("/api/challenges/{challenge_id}")
def challenge(challenge_id: str) -> dict[str, Any]:
    item = get_challenge(challenge_id)
    if item is None:
        raise HTTPException(404, "Challenge not found")
    return {key: value for key, value in item.items() if key not in {"hidden_dataset", "explanation"}}


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    item = get_challenge(request.challenge_id)
    if item is None:
        raise HTTPException(404, "Challenge not found")
    run_id = uuid.uuid4().hex
    db.create_run(run_id, request.session_id, request.challenge_id, request.source)
    try:
        result = run_code(request.source, request.challenge_id, item)
    except RunnerUnavailable as error:
        result = {"status": "unavailable", "error": str(error), "stdout": "", "stderr": ""}
    except (ValueError, OSError) as error:
        result = {"status": "error", "error": str(error), "stdout": "", "stderr": ""}
    db.finish_run(run_id, result["status"], result)
    return {"run_id": run_id, **result}


@app.get("/api/runs/{run_id}")
def read_run(run_id: str, session_id: str) -> dict[str, Any]:
    run = db.get_run(run_id)
    if run is None or run["session_id"] != session_id:
        raise HTTPException(404, "Run not found for this session")
    return {
        "run_id": run["id"],
        "challenge_id": run["challenge_id"],
        "status": run["status"],
        **(run["result"] or {}),
    }


@app.post("/api/submissions")
def submit(request: SubmitRequest) -> dict[str, Any]:
    run = db.get_run(request.run_id)
    if run is None or run["session_id"] != request.session_id:
        raise HTTPException(404, "Completed run not found for this session")
    if run["status"] != "complete" or not run["result"]:
        raise HTTPException(409, "Run code successfully before submitting")
    challenge_data = get_challenge(run["challenge_id"])
    if challenge_data is None:
        raise HTTPException(404, "Challenge not found")
    claim = run["result"].get("submission")
    if not isinstance(claim, dict):
        raise HTTPException(422, "solve(...) must return a submission dictionary")
    try:
        payload = {
            "challenge_id": run["challenge_id"],
            "model_name": challenge_data["model"]["name"],
            "claim": claim,
            "run_result": run["result"],
            "public_dataset": challenge_data["public_dataset"],
            "hidden_dataset": challenge_data["hidden_dataset"],
        }
        grade = run_grade(payload)
    except RunnerUnavailable as error:
        raise HTTPException(503, str(error)) from error
    submission_id = uuid.uuid4().hex
    db.save_submission(submission_id, run["id"], run["challenge_id"], request.session_id, claim, grade["score"], grade)
    return {"submission_id": submission_id, **grade, "explanation": challenge_data.get("explanation")}


@app.get("/api/sessions/{session_id}/submissions")
def session_submissions(session_id: str) -> list[dict[str, Any]]:
    return db.list_submissions(session_id)


@app.get("/api/sessions/{session_id}/runs")
def session_runs(session_id: str) -> list[dict[str, Any]]:
    return db.list_runs(session_id)


@app.get("/api/sessions/{session_id}/drafts/{challenge_id}")
def read_draft(session_id: str, challenge_id: str) -> dict[str, Any]:
    if get_challenge(challenge_id) is None:
        raise HTTPException(404, "Challenge not found")
    return db.get_draft(session_id, challenge_id) or {"source": None}


@app.put("/api/sessions/{session_id}/drafts/{challenge_id}")
def write_draft(session_id: str, challenge_id: str, request: DraftRequest) -> dict[str, Any]:
    if request.session_id != session_id:
        raise HTTPException(403, "Draft session does not match the URL")
    if get_challenge(challenge_id) is None:
        raise HTTPException(404, "Challenge not found")
    db.save_draft(session_id, challenge_id, request.source)
    return {"saved": True}
