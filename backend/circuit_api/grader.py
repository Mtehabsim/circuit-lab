from __future__ import annotations

from typing import Any

from .model import load_checkpoint, load_induction_checkpoint, load_ioi_checkpoint
from .tools import ExperimentTools
from .research import behavior_accuracy


GROUND_TRUTH = {"type": "attention_head", "layer": 2, "head": 1}


def grade_previous_token(
    claim: dict[str, Any],
    run_result: dict[str, Any],
    public_dataset: list[list[str]],
    hidden_dataset: list[list[str]],
    checkpoint_path: str = "/models/previous-token.pt",
) -> dict[str, Any]:
    components = claim.get("components") if isinstance(claim, dict) else None
    first_component = components[0] if isinstance(components, list) and components else {}
    component = first_component if isinstance(first_component, dict) else {}
    identified = (
        isinstance(component, dict)
        and component.get("type") == GROUND_TRUTH["type"]
        and component.get("layer") == GROUND_TRUTH["layer"]
        and component.get("head") == GROUND_TRUTH["head"]
    )

    events = run_result.get("events", [])
    recorded_ablation = any(
        event.get("type") == "ablation"
        and event.get("layer") == component.get("layer")
        and event.get("head") == component.get("head")
        and float(event.get("drop", 0)) >= 0.5
        for event in events
        if isinstance(event, dict)
    )

    fresh_model = load_checkpoint(checkpoint_path)
    tools = ExperimentTools(fresh_model)
    public_records = [{"tokens": tokens} for tokens in public_dataset]
    hidden_records = [{"tokens": tokens} for tokens in hidden_dataset]
    public_baseline = float(tools.evaluate(public_records)["accuracy"])
    public_drops = [tools.ablate_head(tokens, int(component.get("layer", -1)), int(component.get("head", -1)))["drop"] for tokens in public_dataset] if _valid_component(component) else []
    replayed_public_drop = sum(public_drops) / max(len(public_drops), 1)
    hidden_baseline = float(tools.evaluate(hidden_records)["accuracy"])
    hidden_ablation = [tools.ablate_head(tokens, int(component.get("layer", -1)), int(component.get("head", -1))) for tokens in hidden_dataset] if _valid_component(component) else []
    hidden_drop = sum(item["drop"] for item in hidden_ablation) / max(len(hidden_ablation), 1)
    causal = recorded_ablation and replayed_public_drop >= 0.5
    generalizes = hidden_baseline >= 0.95 and hidden_drop >= 0.5

    dimensions = {
        "localization": 40 if identified else 0,
        "causal_evidence": 35 if causal else 0,
        "hidden_generalization": 25 if generalizes else 0,
    }
    score = sum(dimensions.values())
    return {
        "score": score,
        "dimensions": dimensions,
        "checks": {
            "identified_component": identified,
            "recorded_ablation": causal,
            "replayed_public_ablation_drop": replayed_public_drop,
            "public_baseline_accuracy": public_baseline,
            "hidden_baseline_accuracy": hidden_baseline,
            "hidden_ablation_drop": hidden_drop,
            "generalizes": generalizes,
        },
        "feedback": _feedback(identified, causal, generalizes),
    }


def grade_ablation_control(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str = "/models/previous-token.pt",
) -> dict[str, Any]:
    base = grade_previous_token(claim, run_result, public_dataset, hidden_dataset, checkpoint_path)
    events = [event for event in run_result.get("events", []) if isinstance(event, dict) and event.get("type") == "ablation"]
    target = next((event for event in events if (event.get("layer"), event.get("head")) == (2, 1)), None)
    decoy = next((event for event in events if (event.get("layer"), event.get("head")) == (2, 0)), None)
    replay_model = load_checkpoint(checkpoint_path)
    replay_tools = ExperimentTools(replay_model)
    target_drop = [replay_tools.ablate_head(tokens, 2, 1)["drop"] for tokens in public_dataset]
    decoy_drop = [replay_tools.ablate_head(tokens, 2, 0)["drop"] for tokens in public_dataset]
    verified = bool(target and target.get("drop", 0) >= 0.9 and decoy and abs(decoy.get("drop", 1)) < 0.01
                    and sum(target_drop) / max(len(target_drop), 1) >= 0.9
                    and sum(abs(drop) for drop in decoy_drop) / max(len(decoy_drop), 1) < 0.01)
    dimensions = {"localization": 35 if base["checks"]["identified_component"] else 0, "causal_evidence": 40 if verified else 0,
                  "hidden_generalization": base["dimensions"]["hidden_generalization"]}
    return {**base, "score": sum(dimensions.values()), "dimensions": dimensions,
            "checks": {**base["checks"], "decoy_control_verified": verified},
            "feedback": base["feedback"] + (["Your negative control tracks the token but has no causal output effect."] if verified else ["Compare the target head with the correlated decoy using ablation."])}


def grade_value_path(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str = "/models/previous-token.pt",
) -> dict[str, Any]:
    base = grade_previous_token(claim, run_result, public_dataset, hidden_dataset, checkpoint_path)
    events = [event for event in run_result.get("events", []) if isinstance(event, dict) and event.get("type") == "ov_analysis"]
    pathway = False
    replay_tools = ExperimentTools(load_checkpoint(checkpoint_path))
    for event in events:
        tokens = event.get("tokens")
        position = event.get("position")
        if not isinstance(tokens, list) or not isinstance(position, int) or not 1 <= position < len(tokens):
            continue
        replay = replay_tools.value_path(tokens, 2, 1, position)
        if (event.get("layer") == 2 and event.get("head") == 1
                and replay["source_position"] == position - 1 and replay["value_norm"] > 0
                and replay["output_norm"] > 0 and replay["target_logit_contribution"] > 1):
            pathway = True
            break
    dimensions = {"localization": 30 if base["checks"]["identified_component"] else 0,
                  "causal_evidence": 25 if base["checks"]["recorded_ablation"] else 0,
                  "ov_pathway": 30 if pathway else 0, "hidden_generalization": 15 if base["checks"]["generalizes"] else 0}
    return {**base, "score": sum(dimensions.values()), "dimensions": dimensions,
            "checks": {**base["checks"], "ov_pathway_verified": pathway},
            "feedback": base["feedback"] + (["The value/output path writes positive evidence for the expected token."] if pathway else ["Trace the attended token through V and W_O, then measure its expected-token logit contribution."])}


def grade_activation_patch(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str = "/models/previous-token.pt",
) -> dict[str, Any]:
    base = grade_previous_token(claim, run_result, public_dataset, hidden_dataset, checkpoint_path)
    events = [event for event in run_result.get("events", []) if isinstance(event, dict) and event.get("type") == "activation_patch"]
    patch = False
    public_tools = ExperimentTools(load_checkpoint(checkpoint_path))
    for event in events:
        clean = event.get("clean_tokens")
        corrupted = event.get("corrupted_tokens")
        position = event.get("position")
        if (event.get("layer") != 2 or event.get("head") != 1 or not isinstance(clean, list)
                or not isinstance(corrupted, list) or not isinstance(position, int)
                or len(clean) != len(corrupted) or not 1 <= position < len(clean)):
            continue
        replay = public_tools.patch_head_output(clean, corrupted, 2, 1, position)
        if (replay["recovery"] > 1 and replay["corrupted_logit_difference"] < 0
                and replay["patched_logit_difference"] > 0):
            patch = True
            break
    # Recreate clean/corrupted pairs from held-out sequences and measure the same intervention.
    fresh = load_checkpoint(checkpoint_path)
    tools = ExperimentTools(fresh)
    recoveries = []
    for tokens in hidden_dataset:
        if len(tokens) < 3:
            continue
        position = len(tokens) - 1
        corrupted = list(tokens)
        corrupted[position - 1] = tokens[0]
        recoveries.append(tools.patch_head_output(tokens, corrupted, 2, 1, position)["recovery"])
    hidden_recovery = sum(recoveries) / max(len(recoveries), 1)
    verified = patch and hidden_recovery > 1
    dimensions = {"localization": base["dimensions"]["localization"], "patch_evidence": 35 if patch else 0,
                  "hidden_generalization": 25 if verified else 0}
    return {**base, "score": sum(dimensions.values()), "dimensions": dimensions,
            "checks": {**base["checks"], "activation_patch_verified": patch, "hidden_patch_recovery": hidden_recovery,
                       "generalizes": verified},
            "feedback": base["feedback"] + (["Patching the clean head output restores the expected-token logit margin on hidden sequences."] if verified else ["Patch the clean head output into a corrupted run and show the target logit margin recovers."])}


def grade_induction(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str = "/models/induction-toy.pt",
) -> dict[str, Any]:
    expected = {(0, 0), (1, 1)}
    components = claim.get("components", []) if isinstance(claim, dict) else []
    identified = {(item.get("layer"), item.get("head")) for item in components
                  if isinstance(item, dict) and item.get("type") == "attention_head"} == expected
    events = [event for event in run_result.get("events", []) if isinstance(event, dict) and event.get("type") == "ablation"]
    measured = {(event.get("layer"), event.get("head")) for event in events if event.get("drop", 0) > 0.5}
    causal = expected.issubset(measured)
    model = load_induction_checkpoint(checkpoint_path)
    tools = ExperimentTools(model)
    hidden = [{"tokens": tokens} for tokens in hidden_dataset]
    baseline = tools.evaluate(hidden)["accuracy"]
    drops = [tools.ablate_head(tokens, layer, head)["drop"] for tokens in hidden_dataset for layer, head in sorted(expected)]
    hidden_drop = sum(drops) / max(len(drops), 1)
    generalizes = baseline >= 0.95 and hidden_drop >= 0.5
    dimensions = {"circuit_localization": 35 if identified else 0, "causal_evidence": 35 if causal else 0,
                  "hidden_generalization": 30 if generalizes else 0}
    feedback = [
        "Both the previous-token feature head and induction head are identified." if identified else "Identify both stages: the head that writes previous-token identity and the head that matches it.",
        "Ablating both circuit stages reduces induction accuracy." if causal else "Ablate each proposed circuit head and record its measured effect.",
        "The two-head mechanism reproduces on held-out sequences." if generalizes else "The circuit did not retain the expected behavior and causal effect on hidden sequences.",
    ]
    return {"score": sum(dimensions.values()), "dimensions": dimensions,
            "checks": {"identified_circuit": identified, "causal_ablation": causal,
                       "hidden_baseline_accuracy": baseline, "hidden_ablation_drop": hidden_drop,
                       "generalizes": generalizes}, "feedback": feedback}


def grade_ioi(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str = "/models/ioi-toy.pt",
) -> dict[str, Any]:
    expected = {(0, 0), (2, 1)}
    components = claim.get("components", []) if isinstance(claim, dict) else []
    identified = {(item.get("layer"), item.get("head")) for item in components
                  if isinstance(item, dict) and item.get("type") == "attention_head"} == expected
    events = [event for event in run_result.get("events", []) if isinstance(event, dict) and event.get("type") == "ablation"]
    measured = {(event.get("layer"), event.get("head")) for event in events if event.get("drop", 0) >= 0.25}
    causal = expected.issubset(measured)
    model = load_ioi_checkpoint(checkpoint_path)
    tools = ExperimentTools(model)
    hidden = [{"tokens": tokens} for tokens in hidden_dataset]
    baseline = tools.evaluate(hidden)["accuracy"]
    drops = [tools.ablate_head(tokens, layer, head)["drop"] for tokens in hidden_dataset for layer, head in sorted(expected)]
    hidden_drop = sum(drops) / max(len(drops), 1)
    generalizes = baseline == 1.0 and hidden_drop >= 0.25
    dimensions = {"circuit_localization": 35 if identified else 0, "causal_evidence": 35 if causal else 0,
                  "hidden_generalization": 30 if generalizes else 0}
    return {"score": sum(dimensions.values()), "dimensions": dimensions,
            "checks": {"identified_circuit": identified, "causal_ablation": causal,
                       "hidden_baseline_accuracy": baseline, "hidden_ablation_drop": hidden_drop,
                       "generalizes": generalizes},
            "feedback": [
                "The subject-finding and name-moving heads are both identified." if identified else "Identify the head that reads the subject role and the head that moves the other name to the output.",
                "Both proposed heads have measured causal effects." if causal else "Ablate each proposed head and record its effect on the IOI answer.",
                "The circuit generalizes to unseen name pairs." if generalizes else "The proposed circuit did not pass the held-out name-pair tests.",
            ]}


def grade_palindrome(
    claim: dict[str, Any], run_result: dict[str, Any], public_dataset: list[list[str]],
    hidden_dataset: list[list[str]], checkpoint_path: str,
) -> dict[str, Any]:
    """Score independently replayed causal effects; component IDs are never answer-key matched."""
    from .palindrome import load_checkpoint as load_palindrome_checkpoint
    network = load_palindrome_checkpoint(checkpoint_path)
    public = [{"tokens": tokens} for tokens in public_dataset]
    hidden = [{"tokens": tokens} for tokens in hidden_dataset]
    components = claim.get("components", []) if isinstance(claim, dict) else []
    selected: list[tuple[int, int]] = []
    for item in components if isinstance(components, list) else []:
        if not isinstance(item, dict) or item.get("type", "attention_head") != "attention_head":
            continue
        layer, head = item.get("layer"), item.get("head")
        if isinstance(layer, int) and isinstance(head, int) and 0 <= layer < network.n_layers and 0 <= head < network.n_heads:
            pair = (layer, head)
            if pair not in selected:
                selected.append(pair)
    selected = selected[:4]
    residual_locations: list[tuple[int, int]] = []
    raw_locations = claim.get("residual_locations", []) if isinstance(claim, dict) else []
    for item in raw_locations if isinstance(raw_locations, list) else []:
        if not isinstance(item, dict):
            continue
        layer, position = item.get("layer"), item.get("position")
        if isinstance(layer, int) and isinstance(position, int) and 0 <= layer < network.n_layers and position >= -1:
            location = (layer, position)
            if location not in residual_locations:
                residual_locations.append(location)
    residual_locations = residual_locations[:4]
    intervention_count = len(selected) + len(residual_locations)
    baseline_public = behavior_accuracy(network, public)
    baseline_hidden = behavior_accuracy(network, hidden)
    selected_set = set(selected)
    after_public = behavior_accuracy(network, public, ablate=selected_set) if selected else baseline_public
    after_hidden = behavior_accuracy(network, hidden, ablate=selected_set) if selected else baseline_hidden
    public_drop = baseline_public - after_public
    hidden_drop = baseline_hidden - after_hidden

    # Compare against same-sized random component sets, to estimate whether the
    # claimed intervention is selective rather than just a broad model knockout.
    all_components = [(layer, head) for layer in range(network.n_layers) for head in range(network.n_heads)]
    controls = [pair for pair in all_components if pair not in selected_set]
    control_drops = []
    for pair in controls:
        after = behavior_accuracy(network, hidden, ablate={pair})
        control_drops.append(max(0.0, baseline_hidden - after))
    control_mean = sum(control_drops) / max(1, len(control_drops))
    specificity = max(0.0, min(1.0, (max(0.0, hidden_drop) - control_mean) / max(max(0.0, hidden_drop), 0.05)))

    # Events are only evidence that the learner ran an intervention. Their
    # numeric values are discarded; the grader recomputes on fresh checkpoints/data.
    public_keys = {tuple(tokens) for tokens in public_dataset}
    events = [event for event in run_result.get("events", []) if isinstance(event, dict)]
    replayed_ablation = any(
        event.get("type") == "ablation"
        and (event.get("layer"), event.get("head")) in selected_set
        and isinstance(event.get("tokens"), list)
        and tuple(event["tokens"]) in public_keys
        for event in events
    )
    patch_events = [event for event in events if event.get("type") in {"activation_patch", "residual_patch"}]
    public_patch_recovery = 0.0
    patch_event_valid = False
    for event in patch_events:
        pair = (event.get("layer"), event.get("head"))
        location = (event.get("layer"), event.get("position"))
        event_type = event.get("type")
        clean, corrupted, position = event.get("clean_tokens"), event.get("corrupted_tokens"), event.get("position")
        supported = pair in selected_set if event_type == "activation_patch" else location in residual_locations
        if not supported or not isinstance(clean, list) or not isinstance(corrupted, list):
            continue
        valid_position = (
            isinstance(position, int)
            and (0 <= position < len(clean) if event_type == "activation_patch" else position == -1 or 0 <= position < len(clean) + 2)
        )
        if tuple(clean) not in public_keys or not valid_position:
            continue
        if len(clean) != len(corrupted) or clean == corrupted or network.target_id(clean) != 1 or network.target_id(corrupted) != 0:
            continue
        replay_tools = ExperimentTools(network)
        replay = (replay_tools.patch_head_output(clean, corrupted, pair[0], pair[1], position)
                  if event_type == "activation_patch"
                  else replay_tools.patch_residual(clean, corrupted, location[0], location[1]))
        clean_margin = replay["clean_logit_difference"]
        corrupt_margin = replay["corrupted_logit_difference"]
        patched_margin = replay["patched_logit_difference"]
        if clean_margin > 0 and corrupt_margin < 0:
            patch_event_valid = True
            public_patch_recovery = float(patched_margin > 0)
            break

    # Construct held-out near-miss pairs by changing the terminal symbol of
    # hidden positive examples. Replay the same candidate head/position patch.
    patch_rates = []
    for pair in selected:
        restored = attempted = 0
        for clean in hidden_dataset:
            if network.target_id(clean) != 1:
                continue
            corrupted = list(clean)
            corrupted[-1] = next(symbol for symbol in network.vocab[:8] if symbol != corrupted[-1])
            result = ExperimentTools(network).patch_head_output(clean, corrupted, pair[0], pair[1], len(clean) - 1)
            attempted += 1
            restored += int(result["clean_logit_difference"] > 0 and result["corrupted_logit_difference"] < 0 and result["patched_logit_difference"] > 0)
        if attempted:
            patch_rates.append(restored / attempted)
    for layer, position in residual_locations:
        restored = attempted = 0
        for clean in hidden_dataset:
            if network.target_id(clean) != 1:
                continue
            corrupted = list(clean)
            corrupted[-1] = next(symbol for symbol in network.vocab[:8] if symbol != corrupted[-1])
            try:
                result = ExperimentTools(network).patch_residual(clean, corrupted, layer, position)
            except ValueError:
                continue
            attempted += 1
            restored += int(result["clean_logit_difference"] > 0 and result["corrupted_logit_difference"] < 0 and result["patched_logit_difference"] > 0)
        if attempted:
            patch_rates.append(restored / attempted)
    hidden_patch_recovery = max(patch_rates, default=0.0)
    replayable_event = replayed_ablation or patch_event_valid
    causal_effect = max(max(0.0, hidden_drop / 0.60), hidden_patch_recovery)
    generalization = max(
        max(0.0, min(1.0, hidden_drop / max(public_drop, 0.05))),
        max(0.0, min(1.0, hidden_patch_recovery / max(public_patch_recovery, 0.25))),
    )
    size_score = max(0.0, 1.0 - max(0, intervention_count - 1) / 3)
    dimensions = {
        "causal_effect": round(35 * min(1.0, causal_effect)),
        "hidden_generalization": round(25 * generalization),
        "specificity_vs_other_heads": round(20 * specificity),
        "reproducible_intervention": 10 if replayable_event else 0,
        "intervention_size": round(10 * size_score) if selected else 0,
    }
    feedback = []
    if not selected:
        feedback.append("Name the component or small circuit you want the grader to test.")
    elif hidden_drop < 0.05 and hidden_patch_recovery < 0.05:
        feedback.append("The proposed intervention has little effect on the hidden task distribution. Revisit the hypothesis or test another intervention.")
    elif hidden_drop >= 0.05:
        feedback.append(f"Independent replay reduces hidden accuracy by {hidden_drop:.1%}; the effect is measured on fresh inputs, not from submitted numbers.")
    else:
        feedback.append(f"Independent clean-to-corrupted patching restores the target margin on {hidden_patch_recovery:.1%} of hidden near-misses.")
    if not replayable_event:
        feedback.append("Record a proposed ablation or clean-to-corrupted patch with the tools API; the grader ignores reported scores and replays the intervention.")
    if selected and specificity < 0.25:
        feedback.append("The effect is not clearly selective relative to other single-head ablations. Test a narrower component or compare matched controls.")
    if baseline_hidden < 0.9:
        feedback.append("The checkpoint is imperfect on these hidden inputs, so interpret intervention effects only over the model's supported behavior distribution.")
    feedback.append("This score validates behavioral intervention evidence, not the uniqueness or completeness of a prose mechanism.")
    score = sum(dimensions.values())
    if intervention_count > 2:
        score = min(score, 55)
    return {"score": score, "dimensions": dimensions,
            "checks": {"public_baseline_accuracy": baseline_public, "hidden_baseline_accuracy": baseline_hidden,
                       "public_intervened_accuracy": after_public, "hidden_intervened_accuracy": after_hidden,
                       "public_accuracy_drop": public_drop, "hidden_accuracy_drop": hidden_drop,
                       "mean_single_control_head_drop": control_mean, "replayable_public_intervention": replayable_event,
                       "public_patch_recovery": public_patch_recovery, "hidden_patch_recovery": hidden_patch_recovery,
                       "component_count": intervention_count, "residual_location_count": len(residual_locations)}, "feedback": feedback}


def _valid_component(component: Any) -> bool:
    return (
        isinstance(component, dict)
        and isinstance(component.get("layer"), int)
        and 0 <= component["layer"] < 4
        and isinstance(component.get("head"), int)
        and 0 <= component["head"] < 4
    )


def _feedback(identified: bool, causal: bool, generalizes: bool) -> list[str]:
    feedback = []
    if identified:
        feedback.append("Your component matches the mechanism in the checkpoint.")
    else:
        feedback.append("The submitted component does not match the head responsible for this behavior.")
    if causal:
        feedback.append("Your run recorded an ablation of the claimed head with a large measured effect.")
    else:
        feedback.append("Ablate your claimed head and record the behavior metric before and after.")
    if generalizes:
        feedback.append("The behavior and intervention reproduce on held-out sequences.")
    else:
        feedback.append("The mechanism did not meet the hidden-set behavior and intervention thresholds.")
    return feedback
