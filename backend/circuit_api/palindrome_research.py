"""Private replay for executable research functions in the Palindrome mission.

Only imports and learner-defined functions/classes are evaluated. The user's top-level
exploration script is not rerun during grading. In hosted deployments this module must
run in the isolated evaluator container, with hidden inputs mounted read-only.
"""
from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import torch

from .model import ModelAdapter
from .palindrome import generate_examples
from .tools import ExperimentTools


FUNCTIONS = {
    "behavior": ("evaluate_by_length",),
    "attention": ("mirrored_attention_mass",),
    "mechanism": ("predict_qk_logits", "predict_ov_output"),
    "information_flow": ("patch_experiment",),
    "defend": ("patch_experiment",),
}


def _learner_namespace(source: str, model: Any, dataset: list[dict[str, Any]]) -> dict[str, Any]:
    tree = ast.parse(source, filename="research_submission.py")
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            try:
                ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                continue
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace: dict[str, Any] = {"torch": torch, "np": np, "math": math,
                                 "model": model, "dataset": dataset,
                                 "tools": ExperimentTools(model)}
    exec(compile(module, "research_submission.py", "exec"), namespace, namespace)
    return namespace


def _close(actual: Any, expected: Any, atol: float = 2e-5, rtol: float = 2e-5) -> bool:
    try:
        a, b = torch.as_tensor(actual).detach().cpu(), torch.as_tensor(expected).detach().cpu()
        return a.shape == b.shape and bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
    except (TypeError, ValueError, RuntimeError):
        return False


def _test_behavior(fn, model, hidden: list[list[str]]) -> dict[str, Any]:
    samples = [{"tokens": list(tokens)} for tokens in hidden]
    try:
        measured = fn(model, samples)
    except Exception as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}
    expected: dict[int, dict[str, int | float]] = {}
    for row in samples:
        tokens = row["tokens"]
        bucket = expected.setdefault(len(tokens), {"correct": 0, "total": 0, "accuracy": 0.0})
        bucket["correct"] += int(int(model(tokens)[-1].argmax()) == model.target_id(tokens))
        bucket["total"] += 1
    for bucket in expected.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"]
    if not isinstance(measured, dict) or set(map(str, measured)) != set(map(str, expected)):
        return {"passed": False, "error": "Return one metrics record per observed sequence length."}
    for length, target in expected.items():
        actual = measured.get(length, measured.get(str(length)))
        if not isinstance(actual, dict):
            return {"passed": False, "error": f"Length {length} must include correct, total, and accuracy."}
        if actual.get("correct") != target["correct"] or actual.get("total") != target["total"]:
            return {"passed": False, "error": f"Counts are incorrect for length {length}."}
        if not isinstance(actual.get("accuracy"), (int, float)) or abs(actual["accuracy"] - target["accuracy"]) > 1e-10:
            return {"passed": False, "error": f"Accuracy is incorrect for length {length}."}
    try:
        empty = fn(model, [])
        if empty != {}:
            return {"passed": False, "error": "The empty dataset should return an empty result."}
    except Exception as error:
        return {"passed": False, "error": f"Empty-input case failed: {type(error).__name__}: {error}"}
    return {"passed": True, "lengths_tested": len(expected), "examples_tested": len(samples)}


def _test_attention(fn, model, hidden: list[list[str]]) -> dict[str, Any]:
    tests = 0
    errors = []
    with torch.inference_mode():
        for index, tokens in enumerate(hidden[:12]):
            trace = model.run_with_trace(tokens)
            for layer, head in ((0, 0), (0, 1), (1, 2), (2, 3)):
                pattern = trace.attention[layer, head]
                values = [pattern[q, len(tokens) + 1 - q]
                          for q in range(1, len(tokens) + 1)
                          if len(tokens) + 1 - q <= q]
                expected = torch.stack(values).mean() if values else pattern.new_zeros(())
                try:
                    actual = fn(pattern, len(tokens))
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
                    continue
                tests += 1
                if not _close(actual, expected, atol=1e-6, rtol=1e-6):
                    errors.append(f"attention statistic mismatch on private example {index}, L{layer}H{head}")
    return {"passed": tests == 48 and not errors, "tests_passed": tests - len(errors),
            "tests_total": tests, "error": errors[0] if errors else None}


def _test_tensor_math(namespace, model, hidden: list[list[str]]) -> dict[str, Any]:
    qk_fn, ov_fn = namespace.get("predict_qk_logits"), namespace.get("predict_ov_output")
    qk_pass = ov_pass = 0
    failures = []
    rng = torch.Generator().manual_seed(717)
    with torch.inference_mode():
        for tokens in hidden[:8]:
            trace = model.run_with_trace(tokens)
            for layer, head in ((0, 0), (0, 1), (2, 3)):
                q, k = trace.query[layer, head], trace.key[layer, head]
                expected_qk = q @ k.T / math.sqrt(q.shape[-1])
                try:
                    actual = qk_fn(q, k)
                    if _close(actual, expected_qk):
                        qk_pass += 1
                    else:
                        failures.append("QK reconstruction did not match raw scaled dot products.")
                except Exception as error:
                    failures.append(f"QK function: {type(error).__name__}: {error}")
                pattern, values, wo = (trace.attention[layer, head], trace.head_values[layer, head],
                                       model.W_O[layer, head])
                expected_ov = pattern @ values @ wo
                try:
                    actual = ov_fn(pattern, values, wo)
                    if _close(actual, expected_ov):
                        ov_pass += 1
                    else:
                        failures.append("OV reconstruction did not match pattern @ V @ W_O.")
                except Exception as error:
                    failures.append(f"OV function: {type(error).__name__}: {error}")
        # Private synthetic tensors prevent passing by returning a cached activation or constant.
        q = torch.randn(5, 12, generator=rng); k = torch.randn(7, 12, generator=rng)
        try:
            qk_pass += int(_close(qk_fn(q, k), q @ k.T / math.sqrt(12)))
        except Exception as error:
            failures.append(f"Synthetic QK case: {type(error).__name__}: {error}")
        weights = torch.randn(12, 48, generator=rng)
        values = torch.randn(7, 12, generator=rng)
        pattern = torch.softmax(torch.randn(5, 7, generator=rng), dim=-1)
        try:
            ov_pass += int(_close(ov_fn(pattern, values, weights), pattern @ values @ weights))
        except Exception as error:
            failures.append(f"Synthetic OV case: {type(error).__name__}: {error}")
    return {"passed": qk_pass == 25 and ov_pass == 25, "qk_passed": qk_pass, "qk_total": 25,
            "ov_passed": ov_pass, "ov_total": 25, "error": failures[0] if failures else None}


def _make_patch_pairs(model, hidden: list[list[str]], count: int = 12):
    positives = [row for row in hidden if model.target_id(row) == 1]
    generated = [tokens for tokens, label in generate_examples(128, seed=8931) if label == 1]
    pool = positives + generated
    pairs = []
    for index, clean in enumerate(pool):
        if len(clean) < 5:
            continue
        offset = (index * 3 + 1) % (len(clean) // 2)
        changed = len(clean) - 1 - offset
        corrupted = list(clean)
        corrupted[changed] = next(symbol for symbol in "ABCDEFGH" if symbol != corrupted[changed])
        if (model.target_id(clean) == 1 and model.target_id(corrupted) == 0
                and int(model(clean)[-1].argmax()) == 1 and int(model(corrupted)[-1].argmax()) == 0):
            pairs.append((clean, corrupted, changed, offset))
        if len(pairs) == count:
            break
    return pairs


def _candidate(claim: dict[str, Any]) -> dict[str, Any] | None:
    prediction = claim.get("mechanism_prediction", {}) if isinstance(claim, dict) else {}
    item = prediction.get("component") if isinstance(prediction, dict) else None
    if not isinstance(item, dict):
        components = claim.get("components", []) if isinstance(claim, dict) else []
        item = components[0] if components else None
    if not isinstance(item, dict):
        return None
    if item.get("type") == "attention_head":
        layer, head = item.get("layer"), item.get("head")
        return item if isinstance(layer, int) and isinstance(head, int) and 0 <= layer < 3 and 0 <= head < 4 else None
    if item.get("type") == "mlp" and isinstance(item.get("layer"), int) and 0 <= item["layer"] < 3:
        return item
    return None


def _patch_margin(model, clean, corrupted, component, position):
    trace_clean, trace_corrupted = model.run_with_trace(clean), model.run_with_trace(corrupted)
    internal_position = position + 1
    if component["type"] == "attention_head":
        layer, head = component["layer"], component["head"]
        patched = model.run_with_trace(corrupted, patch=(layer, head, internal_position,
                                                          trace_clean.head_outputs[layer, head, internal_position]))
    else:
        layer = component["layer"]
        patched = model.run_with_trace(corrupted, patch_mlp=(layer, internal_position,
                                                              trace_clean.mlp_outputs[layer, internal_position]))
    return float(patched.logits[-1, 1] - patched.logits[-1, 0])


def _test_patch_experiment(fn, model, hidden, claim):
    component = _candidate(claim)
    if component is None:
        return {"passed": False, "error": "Choose one candidate attention head or MLP layer."}
    pairs = _make_patch_pairs(model, hidden)
    if len(pairs) < 8:
        return {"passed": False, "error": "Could not construct enough matched hidden corruption pairs."}
    errors, target_deltas, control_deltas, restorations = [], [], [], []
    for clean, corrupted, target_position, control_position in pairs:
        try:
            output = fn(model, clean, corrupted, component, target_position, control_position)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            continue
        if not isinstance(output, dict):
            errors.append("patch_experiment must return target_margin and control_margin.")
            continue
        expected_target = _patch_margin(model, clean, corrupted, component, target_position)
        expected_control = _patch_margin(model, clean, corrupted, component, control_position)
        if not _close(output.get("target_margin"), expected_target) or not _close(output.get("control_margin"), expected_control):
            errors.append("Patch prediction disagreed with independent target/control replay.")
            continue
        baseline = float(model(corrupted)[-1, 1] - model(corrupted)[-1, 0])
        target_deltas.append(expected_target - baseline)
        control_deltas.append(expected_control - baseline)
        restorations.append(expected_target > 0)
    passed = len(target_deltas) >= 8 and not errors
    return {"passed": passed, "pairs_tested": len(pairs), "pairs_matched": len(target_deltas),
            "mean_target_margin_change": float(np.mean(target_deltas)) if target_deltas else 0.0,
            "mean_absolute_target_change": float(np.mean(np.abs(target_deltas))) if target_deltas else 0.0,
            "mean_absolute_control_change": float(np.mean(np.abs(control_deltas))) if control_deltas else 0.0,
            "target_restoration_rate": float(np.mean(restorations)) if restorations else 0.0,
            "error": errors[0] if errors else None}


def evaluate_learner_research(source: str, milestone: str, network: Any,
                              hidden_dataset: list[list[str]], claim: dict[str, Any]) -> dict[str, Any]:
    model = ModelAdapter(network)
    rows = [{"tokens": list(tokens)} for tokens in hidden_dataset]
    try:
        namespace = _learner_namespace(source, model, rows)
    except Exception as error:
        return {"passed": False, "error": f"Could not load learner functions: {type(error).__name__}: {error}"}
    if milestone == "mechanism":
        result = _test_tensor_math(namespace, model, hidden_dataset)
        return {"passed": result["passed"], "checks": {"tensor_reconstruction": result}}
    results = {}
    for name in FUNCTIONS.get(milestone, ()):
        fn = namespace.get(name)
        if not callable(fn):
            results[name] = {"passed": False, "error": f"Define {name}(...) in your research script."}
        elif name == "evaluate_by_length":
            results[name] = _test_behavior(fn, model, hidden_dataset)
        elif name == "mirrored_attention_mass":
            results[name] = _test_attention(fn, model, hidden_dataset)
        elif name == "patch_experiment":
            results[name] = _test_patch_experiment(fn, model, hidden_dataset, claim)
    return {"passed": all(value.get("passed", False) for value in results.values()), "checks": results}
