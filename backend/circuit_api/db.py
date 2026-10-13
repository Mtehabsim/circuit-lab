from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB = Path(__file__).resolve().parents[2] / "var" / "circuit.sqlite3"
DB_PATH = Path(os.environ.get("CIRCUIT_DB_PATH", str(DEFAULT_DB)))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id),
                challenge_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                claim_json TEXT NOT NULL,
                score INTEGER NOT NULL,
                feedback_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS drafts (
                session_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, challenge_id)
            );
            CREATE TABLE IF NOT EXISTS mission_milestones (
                session_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                milestone_id TEXT NOT NULL,
                passed INTEGER NOT NULL DEFAULT 0,
                points INTEGER NOT NULL DEFAULT 0,
                grade_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, challenge_id, milestone_id)
            );
            CREATE TABLE IF NOT EXISTS revealed_solutions (
                session_id TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                revealed INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, challenge_id)
            );
            CREATE INDEX IF NOT EXISTS runs_by_session ON runs(session_id, created_at);
            CREATE INDEX IF NOT EXISTS submissions_by_session ON submissions(session_id, created_at);
            """
        )


def create_run(run_id: str, session_id: str, challenge_id: str, source: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO runs (id, session_id, challenge_id, status, source) VALUES (?, ?, ?, 'running', ?)",
            (run_id, session_id, challenge_id, source),
        )


def finish_run(run_id: str, status: str, result: dict[str, Any]) -> None:
    with connect() as db:
        db.execute("UPDATE runs SET status = ?, result_json = ? WHERE id = ?", (status, json.dumps(result), run_id))


def get_run(run_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["result"] = json.loads(result.pop("result_json")) if result.get("result_json") else None
    return result


def list_runs(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, challenge_id, status, source, result_json, created_at FROM runs WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 100",
            (session_id,),
        ).fetchall()
    return [
        {"run_id": row["id"], "challenge_id": row["challenge_id"], "status": row["status"], "source": row["source"],
         "result": json.loads(row["result_json"]) if row["result_json"] else None, "created_at": row["created_at"]}
        for row in rows
    ]


def get_draft(session_id: str, challenge_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            "SELECT source, updated_at FROM drafts WHERE session_id = ? AND challenge_id = ?",
            (session_id, challenge_id),
        ).fetchone()
    return dict(row) if row else None


def save_draft(session_id: str, challenge_id: str, source: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO drafts (session_id, challenge_id, source, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(session_id, challenge_id) DO UPDATE SET source = excluded.source, updated_at = CURRENT_TIMESTAMP",
            (session_id, challenge_id, source),
        )


def save_submission(
    submission_id: str,
    run_id: str,
    challenge_id: str,
    session_id: str,
    claim: dict[str, Any],
    score: int,
    feedback: dict[str, Any],
) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO submissions (id, run_id, challenge_id, session_id, claim_json, score, feedback_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (submission_id, run_id, challenge_id, session_id, json.dumps(claim), score, json.dumps(feedback)),
        )


def save_milestone_result(session_id: str, challenge_id: str, milestone_id: str, grade: dict[str, Any]) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO mission_milestones (session_id, challenge_id, milestone_id, passed, points, grade_json) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(session_id, challenge_id, milestone_id) DO UPDATE SET "
            "passed = MAX(mission_milestones.passed, excluded.passed), "
            "points = MAX(mission_milestones.points, excluded.points), grade_json = excluded.grade_json, "
            "updated_at = CURRENT_TIMESTAMP",
            (session_id, challenge_id, milestone_id, int(bool(grade.get("passed"))),
             int(grade.get("earned_points", 0)), json.dumps(grade)),
        )


def attach_grade_to_run(run_id: str, grade: dict[str, Any], milestone_id: str | None) -> None:
    run = get_run(run_id)
    if run is None or not isinstance(run.get("result"), dict):
        return
    result = run["result"]
    result["grade"] = grade
    result["graded_milestone"] = milestone_id
    finish_run(run_id, run["status"], result)


def get_mission_milestones(session_id: str, challenge_id: str) -> dict[str, dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT milestone_id, passed, points, grade_json, updated_at FROM mission_milestones "
            "WHERE session_id = ? AND challenge_id = ?", (session_id, challenge_id),
        ).fetchall()
    return {row["milestone_id"]: {"passed": bool(row["passed"]), "points": row["points"],
                                  "grade": json.loads(row["grade_json"]), "updated_at": row["updated_at"]}
            for row in rows}


def reveal_solution(session_id: str, challenge_id: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO revealed_solutions (session_id, challenge_id, revealed) VALUES (?, ?, 1) "
            "ON CONFLICT(session_id, challenge_id) DO UPDATE SET revealed = 1, updated_at = CURRENT_TIMESTAMP",
            (session_id, challenge_id),
        )


def has_revealed_solution(session_id: str, challenge_id: str) -> bool:
    with connect() as db:
        row = db.execute("SELECT revealed FROM revealed_solutions WHERE session_id = ? AND challenge_id = ?",
                         (session_id, challenge_id)).fetchone()
    return bool(row[0]) if row else False


def list_submissions(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, run_id, challenge_id, score, feedback_json, created_at FROM submissions WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    return [{**dict(row), "feedback": json.loads(row["feedback_json"])} for row in rows]
