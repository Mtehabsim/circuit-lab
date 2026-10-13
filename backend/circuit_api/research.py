"""Code-first analysis and plotting helpers for learner experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import SequenceTokenizer


def behavior_accuracy(model: Any, dataset: list[dict[str, Any]], *, ablate=None) -> float:
    """Measure the checkpoint's task metric without using the experiment recorder."""
    correct = total = 0
    with torch.inference_mode():
        for row in dataset:
            tokens = row["tokens"]
            if getattr(model, "behavior", None) in {"induction", "ioi"}:
                target = model.target_id(tokens)
                if target is None:
                    continue
                correct += int(model(tokens, ablate=ablate)[-1].argmax()) == target
                total += 1
            elif getattr(model, "behavior", None) == "palindrome":
                expected = model.target_id(tokens)
                correct += int(model(tokens, ablate=ablate)[-1].argmax()) == expected
                total += 1
            elif len(tokens) >= 2:
                predicted = model(tokens, ablate=ablate).argmax(dim=-1).tolist()
                ids = SequenceTokenizer(model.d_model).encode(tokens).tolist()
                correct += sum(predicted[pos] == ids[pos - 1] for pos in range(1, len(tokens)))
                total += len(tokens) - 1
    return correct / total if total else 0.0


def ablation_effect(model: Any, dataset: list[dict[str, Any]], layer: int, head: int,
                    metric=behavior_accuracy) -> dict[str, float]:
    before = metric(model, dataset)
    after = metric(model, dataset, ablate=(layer, head))
    return {"before": before, "after": after, "drop": before - after}


def plot_attention(attention: Any, tokens: Sequence[str], *, title: str = "Attention"):
    matrix = attention.detach().cpu().numpy() if isinstance(attention, torch.Tensor) else np.asarray(attention)
    fig, ax = plt.subplots(figsize=(max(4, len(tokens) * 0.55), max(3, len(tokens) * 0.45)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("Key / source token")
    ax.set_ylabel("Query / destination token")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def plot_head_heatmap(effects: Sequence[Any], n_layers: int, n_heads: int, *, title: str = "Head ablation effect"):
    values = np.zeros((n_layers, n_heads), dtype=float)
    for item in effects:
        if isinstance(item, dict):
            layer, head, effect = int(item["layer"]), int(item["head"]), float(item.get("drop", item.get("effect", 0)))
        else:
            layer, head, effect = int(item[0]), int(item[1]), float(item[2])
        values[layer, head] = effect
    fig, ax = plt.subplots(figsize=(max(4, n_heads * 0.8), max(3, n_layers * 0.55)))
    image = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    ax.set_yticks(range(n_layers), [f"L{i}" for i in range(n_layers)])
    ax.set_title(title)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    for layer in range(n_layers):
        for head in range(n_heads):
            ax.text(head, layer, f"{values[layer, head]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Accuracy drop")
    fig.tight_layout()
    return fig


def plot_patching_heatmap(values: Any, *, title: str = "Activation patch recovery"):
    matrix = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    fig, ax = plt.subplots(figsize=(max(4, matrix.shape[-1] * 0.45), max(3, matrix.shape[0] * 0.5)))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Logit-difference recovery")
    fig.tight_layout()
    return fig
