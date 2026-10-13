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
            "SELECT id, challenge_id, status, result_json, created_at FROM runs WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 100",
            (session_id,),
        ).fetchall()
    return [
        {"run_id": row["id"], "challenge_id": row["challenge_id"], "status": row["status"],
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


def list_submissions(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, run_id, challenge_id, score, feedback_json, created_at FROM submissions WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    return [{**dict(row), "feedback": json.loads(row["feedback_json"])} for row in rows]
