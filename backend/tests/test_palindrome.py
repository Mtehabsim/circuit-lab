import json
from pathlib import Path

from circuit_api.challenge_store import get_challenge
from circuit_api.grader import grade_palindrome
from circuit_api.model import ModelAdapter
from circuit_api.palindrome import generate_examples, load_checkpoint
from circuit_api.tools import ExperimentTools


CHECKPOINT = Path(__file__).parents[1] / "circuit_api" / "checkpoints" / "palindrome-machine.pt"


def test_trained_checkpoint_matches_independent_task_and_documents_provenance():
    model = ModelAdapter(load_checkpoint(str(CHECKPOINT)))
    challenge = get_challenge("palindrome-machine")
    provenance = json.loads(CHECKPOINT.with_suffix(".provenance.json").read_text())
    assert provenance["seed"] == 1729
    assert provenance["validation_accuracy"] >= 0.98
    assert model.n_layers == 3 and model.n_heads == 4
    accuracies = {}
    for split in ("public_dataset", "hidden_dataset"):
        rows = challenge[split]
        correct = sum(int(model(tokens)[-1].argmax()) == model.target_id(tokens) for tokens in rows)
        accuracies[split] = correct / len(rows)
    assert accuracies["public_dataset"] == 1.0
    assert accuracies["hidden_dataset"] >= 0.97


def test_code_access_exposes_real_cache_tensors_and_shapes():
    model = ModelAdapter(load_checkpoint(str(CHECKPOINT)))
    tokens = list("ABCCBA")
    logits, cache = model.run_with_cache(tokens)
    assert logits.shape == (len(tokens) + 2, 2)
    assert cache["blocks.0.attn.pattern"].shape == (4, len(tokens) + 2, len(tokens) + 2)
    assert cache["blocks.2.mlp.hook_post"].shape[-1] == 96
    assert model.W_Q.shape[0:2] == (3, 4)


def test_grader_replays_claimed_intervention_ignores_printed_score_and_prefers_effect():
    challenge = get_challenge("palindrome-machine")
    model = load_checkpoint(str(CHECKPOINT))
    tools = ExperimentTools(model)
    tokens = challenge["public_dataset"][0]
    effects = []
    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            event = tools.ablate_head(tokens, layer, head)
            effects.append((event["drop"], layer, head))
    _, layer, head = max(effects)
    claim = {"components": [{"type": "attention_head", "layer": layer, "head": head}]}
    run = {"events": [{"type": "ablation", "layer": layer, "head": head,
                       "tokens": tokens, "drop": 1_000_000}]}
    grade = grade_palindrome(claim, run, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT))
    no_evidence = grade_palindrome(claim, {"events": []}, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT))
    assert grade["checks"]["hidden_accuracy_drop"] < 1
    assert grade["checks"]["replayable_public_intervention"]
    assert grade["score"] < 100  # exact match plus a fabricated number cannot certify a full explanation
    assert no_evidence["score"] < 100


def test_random_guess_and_model_wide_ablation_do_not_earn_full_credit():
    challenge = get_challenge("palindrome-machine")
    model = load_checkpoint(str(CHECKPOINT))
    tokens = challenge["public_dataset"][0]
    tools = ExperimentTools(model)
    decoy = tools.ablate_head(tokens, 2, 0)
    decoy_grade = grade_palindrome(
        {"components": [{"type": "attention_head", "layer": 2, "head": 0}]},
        {"events": [decoy]}, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT)
    )
    destructive = [{"type": "attention_head", "layer": 0, "head": head} for head in range(model.n_heads)]
    destructive_grade = grade_palindrome(
        {"components": destructive}, {"events": []}, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT)
    )
    assert decoy_grade["score"] < 60
    assert destructive_grade["score"] <= 55


def test_patch_event_is_recomputed_and_hidden_patch_generalizes():
    challenge = get_challenge("palindrome-machine")
    model = load_checkpoint(str(CHECKPOINT))
    tools = ExperimentTools(model)
    clean = next(tokens for tokens in challenge["public_dataset"] if tokens == tokens[::-1])
    corrupted = list(clean)
    corrupted[-1] = next(symbol for symbol in "ABCDEFGH" if symbol != corrupted[-1])
    event = tools.patch_residual(clean, corrupted, layer=2, position=-1)
    event["recovery"] = -999999  # numeric claims are ignored
    grade = grade_palindrome(
        {"residual_locations": [{"layer": 2, "position": -1}]},
        {"events": [event]}, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT)
    )
    assert grade["checks"]["public_patch_recovery"] in (0.0, 1.0)
    assert grade["checks"]["hidden_patch_recovery"] >= 0.95
    assert grade["checks"]["replayable_public_intervention"]


def test_measured_mechanism_transfers_to_a_new_hidden_sample():
    challenge = get_challenge("palindrome-machine")
    model = load_checkpoint(str(CHECKPOINT))
    alternate_hidden = [tokens for tokens, _ in generate_examples(80, seed=44501)]
    event = ExperimentTools(model).ablate_head(challenge["public_dataset"][0], 0, 0)
    grade = grade_palindrome(
        {"components": [{"type": "attention_head", "layer": 0, "head": 0}]},
        {"events": [event]}, challenge["public_dataset"], alternate_hidden, str(CHECKPOINT)
    )
    assert grade["checks"]["hidden_baseline_accuracy"] >= 0.95
    assert grade["checks"]["hidden_accuracy_drop"] > 0.25


def test_residual_patch_is_replayed_on_new_near_misses():
    challenge = get_challenge("palindrome-machine")
    model = load_checkpoint(str(CHECKPOINT))
    tools = ExperimentTools(model)
    clean = next(tokens for tokens in challenge["public_dataset"] if tokens == tokens[::-1])
    corrupted = list(clean)
    corrupted[-1] = next(symbol for symbol in "ABCDEFGH" if symbol != corrupted[-1])
    event = tools.patch_residual(clean, corrupted, layer=2, position=-1)
    event["patched_logit_difference"] = -1_000_000
    grade = grade_palindrome(
        {"residual_locations": [{"layer": 2, "position": -1}]},
        {"events": [event]}, challenge["public_dataset"], challenge["hidden_dataset"], str(CHECKPOINT)
    )
    assert grade["checks"]["hidden_patch_recovery"] >= 0.95
    assert grade["checks"]["replayable_public_intervention"]
    assert grade["score"] < 100  # readout localization is not a full mechanistic explanation
