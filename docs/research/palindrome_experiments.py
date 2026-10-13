"""Reproduce the measured Palindrome Machine findings and code-generated figures.

From the repository root:
    cd backend && uv run python ../docs/research/palindrome_experiments.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from circuit_api.challenge_store import get_challenge
from circuit_api.palindrome import generate_examples, load_checkpoint

CHECKPOINT = ROOT / "backend/circuit_api/checkpoints/palindrome-machine.pt"
OUTPUT = ROOT / "docs/research/generated"


def accuracy(model, rows):
    return sum(int(model(tokens)[-1].argmax()) == label for tokens, label in rows) / len(rows)


def build_controlled_sets(seed=601, count=1500):
    rng = Random(seed)
    datasets = {"iid": generate_examples(count, seed=801)}
    for kind in ("restricted_repeated", "single_mismatch", "two_mismatch", "three_mismatch"):
        rows = []
        for _ in range(count):
            n = rng.randint(3, 12)
            alphabet = "AB" if kind == "restricted_repeated" else "ABCDEFGH"
            half = [rng.choice(alphabet) for _ in range((n + 1) // 2)]
            tokens = half + (half[:-1] if n % 2 else half)[::-1]
            label = 1
            if kind != "restricted_repeated" or rng.random() < 0.5:
                mismatches = {"restricted_repeated": 1, "single_mismatch": 1,
                              "two_mismatch": 2, "three_mismatch": 3}[kind]
                offsets = rng.sample(range(n // 2), min(mismatches, n // 2))
                for offset in offsets:
                    tokens[n - 1 - offset] = next(symbol for symbol in "ABCDEFGH" if symbol != tokens[n - 1 - offset])
                label = 0
            rows.append((tokens, label))
        datasets[kind] = rows
    return datasets


def qk_pair_probe(model, seed=915, count=600):
    rng = Random(seed)
    stats = {}
    for layer, head in ((0, 0), (0, 1), (0, 2)):
        stats[(layer, head)] = defaultdict(list)
    for tokens, label in generate_examples(count, seed=seed):
        if label != 1:
            continue
        n = len(tokens)
        offset = rng.randrange(n // 2)
        right = n - 1 - offset
        corrupted = tokens.copy()
        corrupted[right] = next(symbol for symbol in "ABCDEFGH" if symbol != corrupted[right])
        clean_trace, corrupt_trace = model.run_with_trace(tokens), model.run_with_trace(corrupted)
        query, key = right + 1, offset + 1
        for layer, head in stats:
            bucket = stats[(layer, head)]
            for name, trace in (("clean", clean_trace), ("corrupted", corrupt_trace)):
                bucket[name + "_qk"].append(float(trace.attention_logits[layer, head, query, key]))
                bucket[name + "_attention"].append(float(trace.attention[layer, head, query, key]))
                alternatives = [float(trace.attention_logits[layer, head, query, pos])
                               for pos in range(1, query) if pos != key]
                bucket[name + "_mirror_margin"].append(
                    float(trace.attention_logits[layer, head, query, key]) - max(alternatives, default=-99.0))
    return {f"L{layer}H{head}": {name: statistics.mean(values) for name, values in bucket.items()}
            for (layer, head), bucket in stats.items()}


def ablation_probe(model, hidden):
    result = {}
    baseline = accuracy(model, [(tokens, model.target_id(tokens)) for tokens in hidden])
    for name, heads, mlps in (
        ("L0H0", {(0, 0)}, None), ("L0H1", {(0, 1)}, None),
        ("L0H0+L0H1", {(0, 0), (0, 1)}, None),
        ("MLP0", None, {0}), ("MLP1", None, {1}), ("MLP2", None, {2}),
        ("L0H0+MLP0", {(0, 0)}, {0}), ("L0H1+MLP0", {(0, 1)}, {0}),
        ("L0H0+L0H1+MLP0", {(0, 0), (0, 1)}, {0}),
    ):
        correct = 0
        for tokens in hidden:
            expected = model.target_id(tokens)
            correct += int(int(model(tokens, ablate=heads, ablate_mlp=mlps)[-1].argmax()) == expected)
        result[name] = {"accuracy": correct / len(hidden), "drop": baseline - correct / len(hidden)}
    result["baseline_accuracy"] = baseline
    return result


def interaction_probe(model, hidden):
    """Difference-in-differences in signed class margin, per hidden example."""
    probes = {
        "L0H0_x_L0H1": ( {(0, 0)}, {(0, 1)}, None, None),
        "L0H0_x_MLP0": ( {(0, 0)}, None, {0}, None),
        "L0H1_x_MLP0": ( {(0, 1)}, None, {0}, None),
    }
    result = {}
    for name, (left_heads, right_heads, left_mlp, right_mlp) in probes.items():
        interactions = []
        for tokens in hidden:
            def margin(heads=None, mlps=None):
                logits = model(tokens, ablate=heads, ablate_mlp=mlps)[-1]
                return float(logits[1] - logits[0])
            if right_heads is not None:
                both = margin(left_heads | right_heads)
                left = margin(left_heads)
                right = margin(right_heads)
                base = margin()
            else:
                layer = next(iter(left_mlp))
                component_heads = left_heads
                both = margin(component_heads, {layer})
                left = margin(component_heads)
                right = margin(None, {layer})
                base = margin()
            interactions.append(both - left - right + base)
        result[name] = {"mean_signed_interaction": statistics.mean(interactions),
                        "mean_absolute_interaction": statistics.mean(map(abs, interactions)),
                        "examples": len(interactions)}
    return result


def matched_patch_probe(model, hidden, seed=2008, count=80):
    rng = Random(seed)
    pairs = []
    for tokens, label in generate_examples(1000, seed=seed):
        if label != 1 or int(model(tokens)[-1].argmax()) != 1:
            continue
        n = len(tokens)
        offset = rng.randrange(n // 2)
        changed = n - 1 - offset
        corrupted = tokens.copy()
        corrupted[changed] = next(symbol for symbol in "ABCDEFGH" if symbol != corrupted[changed])
        if int(model(corrupted)[-1].argmax()) == 0:
            pairs.append((tokens, corrupted, changed, offset))
        if len(pairs) == count:
            break
    results = {}
    for name, kind, layer, head in (
        ("L0H0", "head", 0, 0), ("L0H1", "head", 0, 1),
        ("MLP0", "mlp", 0, -1), ("MLP2", "mlp", 2, -1),
    ):
        target_deltas, control_deltas, recoveries = [], [], []
        for clean, corrupted, changed, control in pairs:
            baseline = float(model(corrupted)[-1, 1] - model(corrupted)[-1, 0])
            trace = model.run_with_trace(clean)
            patched_margins = []
            for position in (changed, control):
                internal = position + 1
                if kind == "head":
                    patched = model.run_with_trace(corrupted, patch=(layer, head, internal,
                                                                       trace.head_outputs[layer, head, internal]))
                else:
                    patched = model.run_with_trace(corrupted, patch_mlp=(layer, internal,
                                                                          trace.mlp_outputs[layer, internal]))
                patched_margins.append(float(patched.logits[-1, 1] - patched.logits[-1, 0]))
            target, control_margin = patched_margins
            target_deltas.append(abs(target - baseline))
            control_deltas.append(abs(control_margin - baseline))
            recoveries.append(target > 0)
        results[name] = {"target_abs_margin_change": statistics.mean(target_deltas),
                         "control_abs_margin_change": statistics.mean(control_deltas),
                         "target_restoration_rate": statistics.mean(recoveries), "pairs": len(pairs)}
    residual = {}
    for layer in range(model.n_layers):
        restores = []
        for clean, corrupted, _, _ in pairs:
            trace = model.run_with_trace(clean)
            pos = len(clean) + 1
            patched = model.run_with_trace(corrupted, patch={"type": "residual", "layer": layer,
                                                              "position": pos,
                                                              "value": trace.residual_by_layer[layer + 1, pos]})
            restores.append(float(patched.logits[-1, 1] - patched.logits[-1, 0]) > 0)
        residual[str(layer)] = statistics.mean(restores)
    return {"component_patches": results, "final_eos_residual_restoration_rate": residual}


def attention_figure(model):
    values = np.zeros((2, 2))
    for layer, head in ((0, 0), (0, 1)):
        for label in (0, 1):
            mass = []
            for tokens, y in generate_examples(800, seed=3000 + label):
                if y != label:
                    continue
                trace = model.run_with_trace(tokens)
                n = len(tokens)
                mass.extend(float(trace.attention[layer, head, q, n + 1 - q])
                            for q in range(1, n + 1) if n + 1 - q <= q)
            values[head, label] = np.mean(mass)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    image = ax.imshow(values, vmin=0, vmax=0.55, cmap="viridis")
    ax.set_xticks([0, 1], ["near-miss", "palindrome"])
    ax.set_yticks([0, 1], ["L0H0", "L0H1"])
    ax.set_title("Layer 0 mirrored-position attention mass")
    fig.colorbar(image, ax=ax, label="Mean attention weight")
    fig.tight_layout()
    return fig


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(str(CHECKPOINT))
    challenge = get_challenge("palindrome-machine")
    validation = json.loads(CHECKPOINT.with_suffix(".provenance.json").read_text())
    reports = {
        "checkpoint_seed": validation["seed"],
        "validation_accuracy": validation["validation_accuracy"],
        "public_accuracy": accuracy(model, [(row, model.target_id(row)) for row in challenge["public_dataset"]]),
        "hidden_accuracy": accuracy(model, [(row, model.target_id(row)) for row in challenge["hidden_dataset"]]),
        "controlled_accuracy": {name: accuracy(model, rows) for name, rows in build_controlled_sets().items()},
        "qk_matched_pair_probe": qk_pair_probe(model),
        "hidden_ablation": ablation_probe(model, challenge["hidden_dataset"]),
        "hidden_margin_interactions": interaction_probe(model, challenge["hidden_dataset"]),
        "matched_patch_probe": matched_patch_probe(model, challenge["hidden_dataset"]),
    }
    (OUTPUT / "palindrome_measurements.json").write_text(json.dumps(reports, indent=2) + "\n")
    attention_figure(model).savefig(OUTPUT / "layer0_mirrored_attention.png", dpi=150)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
