import pytest

from circuit_api.challenge_store import get_challenge
from circuit_api.grader import grade_previous_token
from circuit_api.model import ModelAdapter, build_checkpoint, load_checkpoint
from circuit_api.tools import ExperimentTools


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "previous-token.pt"
    build_checkpoint(str(path))
    return str(path)


@pytest.fixture
def model(checkpoint):
    return load_checkpoint(checkpoint)


def test_public_and_held_out_previous_token_behavior(model):
    challenge = get_challenge("previous-token-head")
    assert challenge is not None
    tools = ExperimentTools(model)
    public = [{"tokens": tokens} for tokens in challenge["public_dataset"]]
    hidden = challenge["hidden_dataset"]

    assert tools.evaluate(public)["accuracy"] == 1.0
    assert tools.evaluate([{"tokens": tokens} for tokens in hidden])["accuracy"] == 1.0


def test_true_head_ablation_has_large_effect_and_other_heads_do_not(model):
    tools = ExperimentTools(model)
    tokens = ["x", "y", "z", "x", "q"]

    target = tools.ablate_head(tokens, 2, 1)
    control = tools.ablate_head(tokens, 1, 1)

    assert target["before"] == 1.0
    assert target["after"] < 0.1
    assert target["drop"] > 0.9
    assert tools.events[-1]["event_version"] == 1
    assert control["drop"] == 0.0


def test_correlated_decoy_attention_is_not_a_causal_path(model):
    tools = ExperimentTools(model)
    tokens = ["a", "b", "c", "d"]
    target = tools.attention(tokens, 2, 1)
    decoy = tools.attention(tokens, 2, 0)

    assert target[3][2] > 0.99 and decoy[3][2] > 0.99
    assert tools.ablate_head(tokens, 2, 1)["drop"] > 0.9
    assert tools.ablate_head(tokens, 2, 0)["drop"] == 0.0


def test_model_adapter_exposes_named_activations_without_mutating_weights(model):
    adapter = ModelAdapter(model)
    tokens = ["a", "b", "c"]
    original = model.o_proj[2, 1].clone()

    pattern = adapter.activation(tokens, "blocks.2.attn.pattern")
    ablated_logits = adapter.activation(tokens, "logits", ablate=(2, 1))
    query = adapter.activation(tokens, "blocks.2.attn.q")
    assert adapter.activation(tokens, "blocks.2.resid.post").shape == (3, 64)

    assert pattern.shape == (4, 3, 3)
    assert ablated_logits.shape == (3, 64)
    assert query.shape == (4, 3, 64)
    assert (model.o_proj[2, 1] == original).all()


def test_activation_and_logit_lens_tools_return_json_safe_values(model):
    tools = ExperimentTools(ModelAdapter(model))
    tokens = ["a", "b", "c"]

    residual = tools.activation(tokens, "blocks.2.resid.post")
    logits = tools.logit_lens(tokens, layer=2)
    attention_logits = tools.activation(tokens, "blocks.2.attn.logits")
    output_weight = tools.weight("W_O", layer=2, head=1)

    assert len(residual) == len(logits) == 3
    assert attention_logits[0][0][1] < -1e8
    assert len(output_weight) == 64 and len(output_weight[0]) == 64
    assert all(event["event_version"] == 1 for event in tools.events)


def test_grader_replays_claimed_intervention_on_fresh_data(checkpoint, model):
    tools = ExperimentTools(model)
    tokens = ["a", "b", "c", "d"]
    tools.ablate_head(tokens, 2, 1)
    challenge = get_challenge("previous-token-head")
    assert challenge is not None
    claim = {
        "hypothesis": "The head routes information from the prior token.",
        "components": [{"type": "attention_head", "layer": 2, "head": 1}],
        "evidence": [],
    }

    result = grade_previous_token(
        claim, {"events": tools.events}, challenge["public_dataset"], challenge["hidden_dataset"], checkpoint
    )

    assert result["score"] == 100
    assert result["checks"]["generalizes"] is True


def test_attention_trace_is_causal_and_highlights_previous_position(model):
    tools = ExperimentTools(model)
    tokens = ["A", "B", "C", "D"]

    pattern = tools.attention(tokens, 2, 1)

    assert len(pattern) == len(tokens)
    assert pattern[3][2] > 0.99
    assert pattern[3][3] < 0.001
    assert tools.traces["attention"]["layer"] == 2


def test_value_path_and_activation_patch_report_causal_measurements(model):
    tools = ExperimentTools(model)
    value_path = tools.value_path(["A", "B", "C", "D"], 2, 1, 3)
    patch = tools.patch_head_output(
        ["A", "B", "C", "D"], ["A", "A", "C", "D"], 2, 1, 2
    )

    assert value_path["source_position"] == 2
    assert value_path["source_token"] == "C"
    assert value_path["target_logit_contribution"] > 1.0
    assert patch["corrupted_logit_difference"] < 0
    assert patch["patched_logit_difference"] > 0
    assert patch["recovery"] > 1.0
