from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CHALLENGE_DIR = Path(__file__).parent / "challenges"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    layers: int = Field(gt=0)
    heads: int = Field(gt=0)
    d_model: int = Field(gt=0)


class PublicExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: list[str]
    output: list[str | None]


class ChallengeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    id: str = Field(min_length=1, max_length=80)
    status: Literal["active", "archived"] = "active"
    order: int = Field(gt=0)
    title: str
    difficulty: Literal["Easy", "Medium", "Hard"]
    category: str
    model: ModelSpec
    description: str
    behavior: str
    task_goal: str
    milestones: list[dict[str, Any]] = Field(default_factory=list)
    metric: str
    examples: list[PublicExample]
    example_note: str
    learning_objectives: list[str]
    tools: list[str]
    hints: list[str]
    starter_code: str
    submission_schema: dict[str, Any]
    grading_contract: list[dict[str, Any]]
    tool_reference: list[dict[str, str]]
    public_dataset: list[list[str]]
    hidden_dataset: list[list[str]]
    explanation: dict[str, Any]


def load_challenge_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    challenge = ChallengeDefinition.model_validate(raw).model_dump(mode="json")
    if challenge["id"] != path.stem:
        raise ValueError(f"Challenge id {challenge['id']!r} does not match filename {path.name!r}")
    if not challenge["public_dataset"] or not challenge["hidden_dataset"]:
        raise ValueError(f"Challenge {challenge['id']!r} must have public and hidden examples")
    return challenge


def validate_challenges() -> list[dict[str, Any]]:
    paths = sorted(CHALLENGE_DIR.glob("*.json"))
    challenges = [load_challenge_file(path) for path in paths]
    ids = [challenge["id"] for challenge in challenges]
    orders = [challenge["order"] for challenge in challenges]
    if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
        raise ValueError("Challenge ids and progression order values must be unique")
    return sorted(challenges, key=lambda challenge: challenge["order"])


def get_challenge(challenge_id: str) -> dict[str, Any] | None:
    if not challenge_id or Path(challenge_id).name != challenge_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in challenge_id):
        return None
    path = CHALLENGE_DIR / f"{challenge_id}.json"
    if not path.is_file():
        return None
    return load_challenge_file(path)


def list_challenges() -> list[dict[str, Any]]:
    return [
        {key: challenge[key] for key in (
            "id", "title", "difficulty", "category", "model", "description", "behavior", "metric", "learning_objectives", "order"
        )}
        for challenge in validate_challenges() if challenge["status"] == "active"
    ]
