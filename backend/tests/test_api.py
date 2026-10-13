from fastapi.testclient import TestClient

from circuit_api import db, main
from circuit_api.challenge_store import get_challenge, validate_challenges


def test_challenge_endpoint_never_sends_hidden_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attempts.sqlite3")
    with TestClient(main.app) as client:
        response = client.get("/api/challenges/previous-token-head")

    assert response.status_code == 200
    assert "hidden_dataset" not in response.json()
    assert "explanation" not in response.json()
    assert response.json()["id"] == "previous-token-head"


def test_all_versioned_challenges_validate_and_list_in_curriculum_order():
    challenges = validate_challenges()

    assert len(challenges) == 7
    assert [item["order"] for item in challenges] == sorted(item["order"] for item in challenges)
    assert all(item["schema_version"] == 1 for item in challenges)
    assert [item["id"] for item in challenges if item["status"] == "active"] == ["palindrome-machine"]
    assert get_challenge("../pyproject") is None


def test_run_then_submit_persists_and_returns_grade(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attempts.sqlite3")
    monkeypatch.setattr(main, "run_code", lambda source, challenge_id, challenge: {
        "status": "complete",
        "stdout": "accuracy 1.0",
        "events": [{"type": "ablation", "layer": 2, "head": 1, "drop": 1.0}],
        "submission": {"hypothesis": "Previous-token routing", "components": [{"type": "attention_head", "layer": 2, "head": 1}]},
    })
    monkeypatch.setattr(main, "run_grade", lambda payload: {
        "score": 100,
        "dimensions": {"localization": 40, "causal_evidence": 35, "hidden_generalization": 25},
        "checks": {"generalizes": True},
        "feedback": ["All checks passed."],
    })
    with TestClient(main.app) as client:
        run_response = client.post("/api/runs", json={"challenge_id": "previous-token-head", "session_id": "test-session", "source": "def solve(...): pass"})
        assert run_response.status_code == 200
        run_id = run_response.json()["run_id"]
        stored_run = client.get(f"/api/runs/{run_id}", params={"session_id": "test-session"})
        submit_response = client.post("/api/submissions", json={"run_id": run_id, "session_id": "test-session"})
        attempts = client.get("/api/sessions/test-session/submissions")
        runs = client.get("/api/sessions/test-session/runs")

    assert stored_run.status_code == 200
    assert stored_run.json()["stdout"] == "accuracy 1.0"
    assert submit_response.status_code == 200
    assert submit_response.json()["score"] == 100
    assert submit_response.json()["explanation"]["canonical_solution"]
    assert len(attempts.json()) == 1
    assert len(runs.json()) == 1
    assert runs.json()[0]["run_id"] == run_id


def test_submission_cannot_use_another_sessions_run(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attempts.sqlite3")
    monkeypatch.setattr(main, "run_code", lambda source, challenge_id, challenge: {
        "status": "complete", "events": [], "submission": {"components": []}
    })
    with TestClient(main.app) as client:
        run = client.post("/api/runs", json={"challenge_id": "previous-token-head", "session_id": "owner", "source": "pass"}).json()
        response = client.post("/api/submissions", json={"run_id": run["run_id"], "session_id": "other"})
        private_run = client.get(f"/api/runs/{run['run_id']}", params={"session_id": "other"})

    assert response.status_code == 404
    assert private_run.status_code == 404


def test_drafts_are_saved_per_session_and_challenge(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attempts.sqlite3")
    with TestClient(main.app) as client:
        saved = client.put("/api/sessions/s1/drafts/previous-token-head", json={"session_id": "s1", "source": "# draft"})
        loaded = client.get("/api/sessions/s1/drafts/previous-token-head")
        other_session = client.get("/api/sessions/s2/drafts/previous-token-head")

    assert saved.status_code == 200
    assert loaded.json()["source"] == "# draft"
    assert other_session.json()["source"] is None


def test_palindrome_mission_runs_and_submits_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "palindrome-flow.sqlite3")
    source = """import matplotlib.pyplot as plt

def evaluate_by_length(model, samples):
    result = {}
    for row in samples:
        tokens = row["tokens"]
        bucket = result.setdefault(len(tokens), {"correct": 0, "total": 0, "accuracy": 0.0})
        bucket["correct"] += int(int(model(tokens)[-1].argmax()) == model.target_id(tokens))
        bucket["total"] += 1
    for bucket in result.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
    return result

print("Public accuracy:", tools.evaluate(dataset))
print("Reference mounted:", "explanation" in challenge)
plt.plot(sorted(evaluate_by_length(model, dataset)))
plt.show()
submission = {"hypothesis": "I am characterizing accuracy by length."}
"""
    with TestClient(main.app) as client:
        challenge_list = client.get("/api/challenges").json()
        assert [item["id"] for item in challenge_list] == ["palindrome-machine"]
        challenge = client.get("/api/challenges/palindrome-machine").json()
        initial_progress = client.get("/api/missions/palindrome-machine/progress", params={"session_id": "palindrome"}).json()
        run = client.post("/api/runs", json={"challenge_id": "palindrome-machine", "session_id": "palindrome", "source": source})
        assert run.status_code == 200 and run.json()["status"] == "complete", run.text
        assert run.json()["artifacts"]
        run_id = run.json()["run_id"]
        grade = client.post("/api/submissions", json={"run_id": run_id, "session_id": "palindrome"})
        journal = client.get("/api/sessions/palindrome/runs").json()

    assert grade.status_code == 200, grade.text
    assert initial_progress["next_milestone_id"] == "behavior"
    assert grade.json()["passed"] and grade.json()["score"] == 10
    assert grade.json()["progress"]["next_milestone_id"] == "attention"
    assert "explanation" not in grade.json()
    assert journal[0]["source"] == source
    assert journal[0]["result"]["artifacts"]
    assert "Reference mounted: False" in journal[0]["result"]["stdout"]


def test_palindrome_unlocks_milestones_and_reveals_solution_only_by_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "palindrome-unlock.sqlite3")
    with TestClient(main.app) as client:
        run = client.post("/api/runs", json={"challenge_id": "palindrome-machine", "session_id": "learner", "source": "submission = {}"}).json()
        locked_attempt = client.post("/api/submissions", json={"run_id": run["run_id"], "session_id": "learner", "milestone_id": "mechanism"})
        progress_before = client.get("/api/missions/palindrome-machine/progress", params={"session_id": "learner"}).json()
        early_reveal = client.post("/api/missions/palindrome-machine/reveal", json={"session_id": "learner"}).json()
        progress_after = client.get("/api/missions/palindrome-machine/progress", params={"session_id": "learner"}).json()
        challenge = client.get("/api/challenges/palindrome-machine").json()

    assert locked_attempt.status_code == 409
    assert progress_before["next_milestone_id"] == "behavior"
    assert progress_before["milestones"][0]["unlocked"]
    assert early_reveal["explanation"]["canonical_solution"]
    assert progress_after["status"] == "solution_revealed"
    assert progress_after["completed_independently"] is False
    assert "explanation" not in challenge
