import pytest

from circuit_api.challenge_store import get_challenge
from circuit_api.grader import grade_induction
from circuit_api.grader import grade_ioi
from circuit_api.model import build_induction_checkpoint, build_ioi_checkpoint, load_induction_checkpoint, load_ioi_checkpoint
from circuit_api.tools import ExperimentTools


@pytest.fixture
def induction_model(tmp_path):
    path = tmp_path / "induction-toy.pt"
    build_induction_checkpoint(str(path))
    return load_induction_checkpoint(str(path)), str(path)


def test_induction_checkpoint_and_both_circuit_stages(induction_model):
    model, _ = induction_model
    challenge = get_challenge("complete-the-pattern")
    tools = ExperimentTools(model)
    dataset = [{"tokens": tokens} for tokens in challenge["hidden_dataset"]]

    assert tools.evaluate(dataset)["accuracy"] == 1.0
    assert tools.ablate_head(dataset[0]["tokens"], 0, 0)["drop"] > 0.9
    assert tools.ablate_head(dataset[0]["tokens"], 1, 1)["drop"] > 0.9
    pattern = tools.attention(dataset[0]["tokens"], 1, 1)
    assert pattern[-1][1] > 0.99


def test_induction_grader_replays_hidden_circuit(induction_model):
    model, path = induction_model
    challenge = get_challenge("complete-the-pattern")
    tokens = challenge["public_dataset"][0]
    tools = ExperimentTools(model)
    first = tools.ablate_head(tokens, 0, 0)
    second = tools.ablate_head(tokens, 1, 1)
    claim = {"components": [
        {"type": "attention_head", "layer": 0, "head": 0},
        {"type": "attention_head", "layer": 1, "head": 1},
    ]}

    grade = grade_induction(claim, {"events": [first, second]}, challenge["public_dataset"], challenge["hidden_dataset"], path)

    assert grade["score"] == 100


def test_ioi_name_mover_generalizes_and_both_heads_are_causal(tmp_path):
    path = tmp_path / "ioi-toy.pt"
    build_ioi_checkpoint(str(path))
    model = load_ioi_checkpoint(str(path))
    challenge = get_challenge("who-gets-the-book")
    tools = ExperimentTools(model)
    hidden = [{"tokens": tokens} for tokens in challenge["hidden_dataset"]]
    assert tools.evaluate(hidden)["accuracy"] == 1.0
    assert tools.ablate_head(hidden[0]["tokens"], 0, 0)["drop"] == 1.0
    assert tools.ablate_head(hidden[0]["tokens"], 2, 1)["drop"] == 1.0
    claim = {"components": [
        {"type": "attention_head", "layer": 0, "head": 0},
        {"type": "attention_head", "layer": 2, "head": 1},
    ]}
    public_tokens = challenge["public_dataset"][0]
    events = [tools.ablate_head(public_tokens, 0, 0), tools.ablate_head(public_tokens, 2, 1)]
    grade = grade_ioi(claim, {"events": events}, challenge["public_dataset"], challenge["hidden_dataset"], str(path))

    assert grade["score"] == 100
