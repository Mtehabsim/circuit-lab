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

tokens = dataset[0]["tokens"]
logits, cache = model.run_with_cache(tokens)
print("accuracy", tools.evaluate(dataset))
print("cache tensors", {name: tuple(value.shape) for name, value in cache.items()})
attn = model.get_attention(tokens, layer=0, head=0)
plt.imshow(attn.detach().cpu())
plt.title("Candidate attention pattern")
plt.show()

scores = []
for layer in range(model.n_layers):
    for head in range(model.n_heads):
        drops = [tools.ablate_head(row["tokens"], layer, head)["drop"] for row in dataset]
        scores.append((sum(drops) / len(drops), layer, head))
_, layer, head = max(scores)
submission = {
    "hypothesis": "A small set of early attention outputs contributes to the palindrome decision; I compared head ablations and checked the strongest candidate.",
    "components": [{"type": "attention_head", "layer": layer, "head": head}],
}
"""
    with TestClient(main.app) as client:
        challenge_list = client.get("/api/challenges").json()
        assert [item["id"] for item in challenge_list] == ["palindrome-machine"]
        challenge = client.get("/api/challenges/palindrome-machine").json()
        run = client.post("/api/runs", json={"challenge_id": "palindrome-machine", "session_id": "palindrome", "source": source})
        assert run.status_code == 200 and run.json()["status"] == "complete", run.text
        assert run.json()["artifacts"]
        run_id = run.json()["run_id"]
        grade = client.post("/api/submissions", json={"run_id": run_id, "session_id": "palindrome"})
        journal = client.get("/api/sessions/palindrome/runs").json()

    assert grade.status_code == 200, grade.text
    assert grade.json()["score"] >= 70
    assert grade.json()["checks"]["hidden_baseline_accuracy"] >= 0.95
    assert grade.json()["checks"]["replayable_public_intervention"]
    assert grade.json()["explanation"]["canonical_solution"]
    assert journal[0]["source"] == source
    assert journal[0]["result"]["artifacts"]
