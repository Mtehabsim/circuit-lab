import json
from pathlib import Path

from circuit_api.challenge_store import get_challenge
from circuit_api.grader import grade_palindrome
from circuit_api.model import ModelAdapter
from circuit_api.palindrome import generate_examples, load_checkpoint
from circuit_api.tools import ExperimentTools


CHECKPOINT = Path(__file__).parents[1] / "circuit_api" / "checkpoints" / "palindrome-machine.pt"

RESEARCHER_SOURCE = '''
import torch

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

def mirrored_attention_mass(attention, content_length):
    values = [attention[q, content_length + 1 - q]
              for q in range(1, content_length + 1)
              if content_length + 1 - q <= q]
    return torch.stack(values).mean() if values else attention.new_zeros(())

def predict_qk_logits(q, k):
    return q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)

def predict_ov_output(pattern, values, w_o):
    return pattern @ values @ w_o

def patch_experiment(model, clean_tokens, corrupted_tokens, component, target_position, control_position):
    clean = model.run_with_trace(clean_tokens)
    result = {}
    for key, position in (("target_margin", target_position), ("control_margin", control_position)):
        internal_position = position + 1
        if component["type"] == "attention_head":
            layer, head = component["layer"], component["head"]
            patched = model.run_with_trace(
                corrupted_tokens,
                patch=(layer, head, internal_position, clean.head_outputs[layer, head, internal_position]),
            )
        else:
            layer = component["layer"]
            patched = model.run_with_trace(
                corrupted_tokens,
                patch_mlp=(layer, internal_position, clean.mlp_outputs[layer, internal_position]),
            )
        result[key] = float(patched.logits[-1, 1] - patched.logits[-1, 0])
    return result
'''


def challenge():
    return get_challenge("palindrome-machine")


def grade(milestone, claim=None, run=None, source=RESEARCHER_SOURCE, hidden=None):
    item = challenge()
    return grade_palindrome(claim or {}, run or {"events": []}, item["public_dataset"],
                            hidden or item["hidden_dataset"], str(CHECKPOINT),
                            milestone_id=milestone, source=source)


def test_checkpoint_provenance_and_documented_behavior_are_real():
    model = ModelAdapter(load_checkpoint(str(CHECKPOINT)))
    item = challenge()
    provenance = json.loads(CHECKPOINT.with_suffix(".provenance.json").read_text())
    assert provenance["seed"] == 1729
    assert provenance["validation_accuracy"] >= 0.98
    assert model.n_layers == 3 and model.n_heads == 4
    public = [int(model(tokens)[-1].argmax()) == model.target_id(tokens) for tokens in item["public_dataset"]]
    hidden = [int(model(tokens)[-1].argmax()) == model.target_id(tokens) for tokens in item["hidden_dataset"]]
    assert sum(public) / len(public) == 1.0
    assert sum(hidden) / len(hidden) >= 0.97
    assert any(not correct for correct in hidden)
    assert sum(int(model(tokens)[-1].argmax()) == model.target_id(tokens)
               for tokens, _ in generate_examples(200, seed=992, lengths=range(13, 17))) / 200 < 0.7


def test_cached_trace_is_reused_and_exposes_real_tensor_shapes(monkeypatch):
    model = ModelAdapter(load_checkpoint(str(CHECKPOINT)))
    calls = 0
    original = model.run_with_trace

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "run_with_trace", counted)
    logits, cache = model.run_with_cache(list("ABCCBA"))
    assert calls == 1
    assert logits.shape == (8, 2)
    assert cache["blocks.0.attn.pattern"].shape == (4, 8, 8)
    assert cache["blocks.2.mlp.hook_post"].shape[-1] == 96
    assert model.W_Q.shape[:2] == (3, 4)


def test_shortcut_head_sweep_only_is_not_a_passing_research_mission():
    item = challenge()
    model = load_checkpoint(str(CHECKPOINT))
    event = ExperimentTools(model).ablate_head(item["public_dataset"][0], 0, 0)
    event["drop"] = 1e12  # fabricated display value is ignored
    result = grade("causality", {"components": [{"type": "attention_head", "layer": 0, "head": 0}]},
                   {"events": [event]}, source="submission = {'components': [{'type':'attention_head','layer':0,'head':0}]}\n")
    assert result["passed"]
    assert result["score"] == 15
    assert result["score"] < 40  # localization/ablation caps early evidence at 40 total points
    assert result["checks"]["hidden_accuracy_drop"] < 0.5


def test_wrong_candidate_and_destructive_component_list_do_not_pass_causality():
    item = challenge()
    model = load_checkpoint(str(CHECKPOINT))
    event = ExperimentTools(model).ablate_head(item["public_dataset"][0], 2, 0)
    wrong = grade("causality", {"components": [{"type": "attention_head", "layer": 2, "head": 0}]},
                  {"events": [event]})
    broad_components = [{"type": "attention_head", "layer": 0, "head": h} for h in range(4)]
    destructive = grade("causality", {"components": broad_components}, {"events": [event]})
    assert not wrong["passed"]
    assert wrong["score"] <= 5
    assert not destructive["passed"]
    assert destructive["score"] <= 5


def test_behavior_and_attention_functions_are_executed_on_private_inputs():
    behavior = grade("behavior")
    attention = grade("attention")
    wrong_behavior_source = RESEARCHER_SOURCE.replace('bucket["correct"] += int(int(model(tokens)[-1].argmax()) == model.target_id(tokens))', 'bucket["correct"] += 1')
    wrong = grade("behavior", source=wrong_behavior_source)
    assert behavior["passed"] and behavior["score"] == 10
    assert attention["passed"] and attention["score"] == 15
    assert not wrong["passed"]


def test_raw_qk_and_ov_tensor_implementations_pass_hidden_replay():
    result = grade("mechanism")
    assert result["passed"]
    assert result["score"] == 25
    checks = result["checks"]["tensor_reconstruction"]
    assert checks["qk_passed"] == checks["qk_total"] == 25
    assert checks["ov_passed"] == checks["ov_total"] == 25


def test_wrong_tensor_formula_fails_mechanism_milestone():
    bad = RESEARCHER_SOURCE.replace('q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)', 'q @ k.transpose(-1, -2)')
    result = grade("mechanism", source=bad)
    assert not result["passed"]
    assert result["score"] < 25


def test_target_patch_and_earlier_causal_position_control_generalize():
    claim = {"components": [{"type": "attention_head", "layer": 0, "head": 0}],
             "mechanism_prediction": {"component": {"type": "attention_head", "layer": 0, "head": 0}}}
    result = grade("information_flow", claim)
    check = result["checks"]["patch_experiment"]
    assert result["passed"]
    assert check["pairs_matched"] >= 8
    assert check["mean_absolute_target_change"] > check["mean_absolute_control_change"] + 1.5
    assert check["target_restoration_rate"] < 0.25


def test_final_residual_replacement_cannot_pass_upstream_information_flow():
    claim = {"residual_locations": [{"layer": 2, "position": -1}]}
    result = grade("information_flow", claim, source="submission = {'residual_locations': [{'layer':2,'position':-1}]}\n")
    assert not result["passed"]
    assert result["score"] == 0


def test_researcher_can_defend_mechanism_and_genuine_alternative_without_fixed_ids():
    claim = {
        "components": [{"type": "mlp", "layer": 0}],
        "mechanism_prediction": {"component": {"type": "mlp", "layer": 0}},
        "alternative_component": {"type": "attention_head", "layer": 0, "head": 2},
        "prediction": {"candidate_vs_alternative": "greater", "candidate_restoration": "rare"},
        "hypothesis": "The first MLP transforms early attention evidence into a larger class-margin change.",
        "alternative_hypothesis": "A correlated head is sufficient and the MLP only changes confidence.",
    }
    result = grade("defend", claim)
    comparison = result["checks"]["candidate_alternative_patch_comparison"]
    assert result["passed"]
    assert comparison["observed_relation"] == "greater"
    assert result["checks"]["relation_supported"]
    assert result["checks"]["restoration_supported"]


def test_incorrect_mechanistic_prediction_fails_defense_milestone():
    claim = {
        "components": [{"type": "attention_head", "layer": 0, "head": 0}],
        "mechanism_prediction": {"component": {"type": "attention_head", "layer": 0, "head": 0}},
        "alternative_component": {"type": "attention_head", "layer": 0, "head": 2},
        "prediction": {"candidate_vs_alternative": "less", "candidate_restoration": "common"},
        "hypothesis": "Candidate output should change the class margin more than the alternative.",
        "alternative_hypothesis": "The alternative is the stronger causal route.",
    }
    result = grade("defend", claim)
    assert not result["passed"]
    assert not result["checks"]["relation_supported"]
    assert not result["checks"]["restoration_supported"]


def test_new_seed_does_not_break_behavior_metric_implementation():
    model = load_checkpoint(str(CHECKPOINT))
    alternate = [tokens for tokens, _ in generate_examples(80, seed=44501)]
    result = grade("behavior", hidden=alternate)
    assert result["passed"]


def test_causal_score_is_an_effect_comparison_not_mislabeled_specificity():
    item = challenge()
    model = load_checkpoint(str(CHECKPOINT))
    tools = ExperimentTools(model)
    event = tools.ablate_mlp(item["public_dataset"][0], 0)
    result = grade("causality", {"components": [{"type": "mlp", "layer": 0}]}, {"events": [event]})
    assert result["passed"]
    assert "not task specificity" in result["checks"]["comparison_label"]


def test_three_research_journeys_separate_shortcut_experiment_and_researcher():
    item = challenge()
    model = load_checkpoint(str(CHECKPOINT))
    tools = ExperimentTools(model)

    # A: a sweep with a picked head cannot satisfy the first milestone, and
    # even when allowed to submit a causality claim earns only that stage.
    shortcut_source = """def sweep(model):
    return [(layer, head) for layer in range(model.n_layers) for head in range(model.n_heads)]
submission = {"components": [{"type": "attention_head", "layer": 0, "head": 0}]}
"""
    shortcut = grade("behavior", {"components": [{"type": "attention_head", "layer": 0, "head": 0}]},
                     {"events": []}, source=shortcut_source)
    head_event = tools.ablate_head(item["public_dataset"][0], 0, 0)
    shortcut_causal = grade("causality", {"components": [{"type": "attention_head", "layer": 0, "head": 0}]},
                            {"events": [head_event]}, source=shortcut_source)
    assert not shortcut["passed"] and shortcut["score"] == 0
    assert shortcut_causal["passed"] and shortcut_causal["score"] == 15

    # B: the experimentalist implements the behavior/attention measures and
    # establishes causal contribution, but has no QK/OV or patch evidence.
    early = [grade(stage) for stage in ("behavior", "attention")]
    mlp_event = tools.ablate_mlp(item["public_dataset"][0], 0)
    early.append(grade("causality", {"components": [{"type": "mlp", "layer": 0}]},
                       {"events": [mlp_event]}))
    experimentalist_points = sum(result["score"] for result in early)
    assert all(result["passed"] for result in early)
    assert experimentalist_points == 40
    assert not grade("mechanism", source="").get("passed")

    # C: all six code contracts and hidden causal predictions pass. The
    # selected MLP-vs-head comparison is intentionally not the reference head pair.
    researcher_claim = {
        "components": [{"type": "mlp", "layer": 0}],
        "mechanism_prediction": {"component": {"type": "mlp", "layer": 0}},
        "alternative_component": {"type": "attention_head", "layer": 0, "head": 2},
        "prediction": {"candidate_vs_alternative": "greater", "candidate_restoration": "rare"},
        "hypothesis": "Replacing MLP0 output at the changed position will move the class margin more than at its mirrored control.",
        "alternative_hypothesis": "A correlated attention head will produce a larger target-site margin change than MLP0.",
    }
    journey = [grade("behavior"), grade("attention")]
    mlp_event = tools.ablate_mlp(item["public_dataset"][0], 0)
    journey.append(grade("causality", researcher_claim, {"events": [mlp_event]}))
    journey.extend(grade(stage, researcher_claim) for stage in ("mechanism", "information_flow", "defend"))
    assert [result["passed"] for result in journey] == [True] * 6
    assert sum(result["score"] for result in journey) == 100
    assert journey[-1]["checks"]["relation_supported"]
