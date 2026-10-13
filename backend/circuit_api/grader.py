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
    hidden_dataset: list[list[str]], checkpoint_path: str, *, milestone_id: str = "behavior",
    source: str = "",
) -> dict[str, Any]:
    """Grade one research milestone; point weights sum to the mission's 100-point total."""
    from .palindrome import load_checkpoint as load_palindrome_checkpoint
    from .palindrome_research import _candidate, _make_patch_pairs, _patch_margin, evaluate_learner_research
    network = load_palindrome_checkpoint(checkpoint_path)
    weights = {"behavior": 10, "attention": 15, "causality": 15, "mechanism": 25,
               "information_flow": 20, "defend": 15}
    if milestone_id not in weights:
        return {"score": 0, "passed": False, "milestone_id": milestone_id,
                "feedback": ["Unknown research milestone."]}
    stage = evaluate_learner_research(source, milestone_id, network, hidden_dataset, claim)
    checks = stage.get("checks", {})
    candidate = _candidate(claim)
    feedback: list[str] = []
    earned = 0
    if milestone_id == "behavior":
        earned = weights[milestone_id] if checks.get("evaluate_by_length", {}).get("passed") else 0
        feedback.append("Your length-stratified evaluation matches private replay, including correct counts and the empty-input case." if earned else checks.get("evaluate_by_length", {}).get("error", "Implement evaluate_by_length(model, samples) and return correct, total, and accuracy for each length."))
    elif milestone_id == "attention":
        earned = weights[milestone_id] if checks.get("mirrored_attention_mass", {}).get("passed") else 0
        feedback.append("Your attention statistic matches raw cached tensors across private heads and sequences. This verifies the measurement, not a causal mechanism." if earned else checks.get("mirrored_attention_mass", {}).get("error", "Implement mirrored_attention_mass(attention, content_length) from the raw attention matrix."))
    elif milestone_id == "causality":
        components = claim.get("components", []) if isinstance(claim, dict) else []
        unique = []
        for item in components if isinstance(components, list) else []:
            if isinstance(item, dict) and _valid_palindrome_component(item) and item not in unique:
                unique.append(item)
        events = [event for event in run_result.get("events", []) if isinstance(event, dict)]
        public_keys = {tuple(row) for row in public_dataset}
        event_ok = any(_event_matches_component(event, component) and isinstance(event.get("tokens"), list)
                       and tuple(event["tokens"]) in public_keys for event in events for component in unique)
        baseline = behavior_accuracy(network, [{"tokens": row} for row in hidden_dataset])
        effect = 0.0
        if unique and len(unique) <= 2:
            head_set = {(item["layer"], item["head"]) for item in unique if item["type"] == "attention_head"}
            mlp_set = {item["layer"] for item in unique if item["type"] == "mlp"}
            after = behavior_accuracy(network, [{"tokens": row} for row in hidden_dataset],
                                      ablate=head_set, ablate_mlp=mlp_set)
            effect = baseline - after
        valid = bool(event_ok and len(unique) in (1, 2) and effect >= 0.08)
        earned = weights[milestone_id] if valid else (5 if event_ok else 0)
        feedback.append(f"Independent hidden replay measures a {effect:.1%} accuracy decrease from the proposed component set." if valid else "A recorded ablation is only candidate evidence. Keep the circuit to at most two components and establish a reproducible hidden effect before advancing.")
        checks.update(candidate_ablation_recorded=event_ok, hidden_accuracy_drop=effect,
                      hidden_baseline_accuracy=baseline, component_count=len(unique),
                      comparison_label="effect relative to matched component interventions; not task specificity")
    elif milestone_id == "mechanism":
        evidence = checks.get("tensor_reconstruction", {})
        earned = weights[milestone_id] if evidence.get("passed") else round(weights[milestone_id] * min(1, (evidence.get("qk_passed", 0) + evidence.get("ov_passed", 0)) / 50))
        feedback.append("Your QK and OV tensor calculations match independent replay on real and synthetic inputs." if earned == weights[milestone_id] else evidence.get("error", "Reconstruct both scaled QK logits and pattern @ V @ W_O using the provided raw tensors."))
    elif milestone_id == "information_flow":
        evidence = checks.get("patch_experiment", {})
        effect_gap = evidence.get("mean_absolute_target_change", 0) - evidence.get("mean_absolute_control_change", 0)
        localized = evidence.get("passed") and effect_gap > 1.5
        earned = weights[milestone_id] if localized else (8 if evidence.get("passed") else 0)
        feedback.append(f"Your patch experiment matches private replay; target-site margin movement exceeds the causal-position control by {effect_gap:.2f} logits. This localizes a contribution without asserting the full algorithm." if localized else evidence.get("error", "Compare a clean patch at the changed token with the same component patched at its causally earlier mirrored control position."))
        checks["target_vs_control_margin_gap"] = effect_gap
        checks["causal_position_control"] = localized
    elif milestone_id == "defend":
        evidence = checks.get("patch_experiment", {})
        prediction = claim.get("prediction", {}) if isinstance(claim, dict) else {}
        alternative = claim.get("alternative_component") if isinstance(claim, dict) else None
        pairs = _make_patch_pairs(network, hidden_dataset, count=32)
        alt_valid = _valid_palindrome_component(alternative) if alternative else False
        candidate = candidate or _first_component(claim)
        comparison = _compare_patch_hypotheses(network, pairs, candidate, alternative) if candidate and alt_valid else {}
        relation = prediction.get("candidate_vs_alternative") if isinstance(prediction, dict) else None
        restoration = prediction.get("candidate_restoration") if isinstance(prediction, dict) else None
        relation_ok = relation == comparison.get("observed_relation")
        restoration_ok = restoration == comparison.get("observed_restoration")
        text_ok = bool(isinstance(claim.get("hypothesis"), str) and len(claim["hypothesis"].strip()) >= 20
                       and isinstance(claim.get("alternative_hypothesis"), str)
                       and len(claim["alternative_hypothesis"].strip()) >= 20)
        complete = bool(evidence.get("passed") and alt_valid and relation_ok and restoration_ok and text_ok)
        earned = weights[milestone_id] if complete else (5 if evidence.get("passed") and comparison else 0)
        feedback.append("Your predicted target/control and candidate/alternative relations match fresh held-out patch replay." if complete else "State a competing mechanism and predict how the candidate compares with it on hidden matched patch/control pairs; free-text explanation is retained but not judged as scientific ground truth.")
        checks.update(candidate_alternative_patch_comparison=comparison,
                      predicted_candidate_relation=relation, relation_supported=relation_ok,
                      restoration_prediction=restoration, restoration_supported=restoration_ok,
                      qualitative_claim_recorded=text_ok)
    passed = earned == weights[milestone_id]
    feedback.append("The milestone verifies the stated measurement or intervention contract; it does not certify a unique or complete prose explanation.")
    return {"score": earned, "earned_points": earned, "possible_points": weights[milestone_id],
            "milestone_score": round(100 * earned / weights[milestone_id]),
            "milestone_id": milestone_id, "passed": passed, "checks": checks, "feedback": feedback}


def _valid_palindrome_component(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") == "attention_head":
        return isinstance(item.get("layer"), int) and 0 <= item["layer"] < 3 and isinstance(item.get("head"), int) and 0 <= item["head"] < 4
    return item.get("type") == "mlp" and isinstance(item.get("layer"), int) and 0 <= item["layer"] < 3


def _event_matches_component(event: dict[str, Any], component: dict[str, Any]) -> bool:
    if component["type"] == "attention_head":
        return event.get("type") == "ablation" and event.get("layer") == component["layer"] and event.get("head") == component["head"]
    return event.get("type") == "mlp_ablation" and event.get("layer") == component["layer"]


def _first_component(claim: dict[str, Any]) -> dict[str, Any] | None:
    values = claim.get("components", []) if isinstance(claim, dict) else []
    return values[0] if values and _valid_palindrome_component(values[0]) else None


def _compare_patch_hypotheses(network, pairs, candidate, alternative):
    from .palindrome_research import _patch_margin
    if not candidate or not alternative or not pairs:
        return {}
    effects, alt_effects, restored = [], [], []
    for clean, corrupted, changed, _ in pairs:
        baseline = float(network(corrupted)[-1, 1] - network(corrupted)[-1, 0])
        candidate_margin = _patch_margin(network, clean, corrupted, candidate, changed)
        alternative_margin = _patch_margin(network, clean, corrupted, alternative, changed)
        effects.append(abs(candidate_margin - baseline))
        alt_effects.append(abs(alternative_margin - baseline))
        restored.append(candidate_margin > 0)
    candidate_effect, alternative_effect = sum(effects)/len(effects), sum(alt_effects)/len(alt_effects)
    relation = "greater" if candidate_effect > alternative_effect + 1.5 else "less" if alternative_effect > candidate_effect + 1.5 else "similar"
    restoration_rate = sum(restored)/len(restored)
    restoration = "rare" if restoration_rate < 0.25 else "common" if restoration_rate > 0.70 else "mixed"
    return {"candidate_mean_absolute_margin_change": candidate_effect,
            "alternative_mean_absolute_margin_change": alternative_effect,
            "candidate_restoration_rate": restoration_rate,
            "observed_relation": relation, "observed_restoration": restoration,
            "n_matched_pairs": len(pairs)}


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
