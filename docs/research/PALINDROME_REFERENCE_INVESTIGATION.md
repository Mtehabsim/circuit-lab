# Reference investigation: The Palindrome Machine

This is an original analysis of the checkpoint shipped at `backend/circuit_api/checkpoints/palindrome-machine.pt`. It is written from measurements on this checkpoint. It does not claim to reproduce the mechanism found in ARENA's separately trained model.

## Model and task

The model is a three-layer, four-head, width-48 causal transformer with learned absolute position embeddings, pre-attention/pre-MLP LayerNorm, a GELU MLP, and a binary classifier at the final position. It was optimized with AdamW on balanced generated palindrome/non-palindrome sequences over `A`–`H`, lengths 3–12, using seed 1729. No task-specific circuit weights are inserted. The tracked provenance file records the data seeds, optimization settings, architecture, and validation result.

The generated validation sample is 98.9% accurate. The mission's fixed public sample is classified perfectly; the hidden set is 94/96 (97.9%). A separate check at lengths 13–16 is about 50% accurate. Those longer lengths are out of distribution; the mission does not count them as hidden generalization. Near-miss accuracy varies by mirrored mismatch offset, so report it by offset instead of treating the aggregate as a solved algorithm.

## Investigation steps

Start with behavior, not a head label. Measure accuracy overall and by length, then inspect errors using near-miss pairs that differ at one mirrored location. This checks whether a proposed mechanism explains the algorithm rather than memorizing common strings.

Next, inspect the actual cache tensors. For a sequence of `n` symbols, the model internally prepends `<BOS>` and appends `<EOS>`, so attention and residual activations have `n + 2` positions. Compare attention rows across layers, heads, positive/negative labels, and matched clean/corrupted strings. On the sample `ABCCBA`, the layer-0/head-0 final-query attention is spread across several positions. It does not simply place all its mass on the mirrored character. Treating that heatmap as the explanation would be misleading.

A held-out ablation sweep identifies more than one substantial early contribution: on the fixed hidden set, ablating L0H0 drops accuracy by 36.5 percentage points and ablating L0H1 drops it by 29.2 points. Ablating both drops accuracy by 43.8 points, less than the sum of the separate drops, which is consistent with interaction or partial redundancy. This is evidence of causal contribution under this distribution, not proof that either head computes mirrored equality by itself.

A residual patch sweep provides localization evidence. Patching the final EOS residual after layer 2 restores the positive margin on the tested hidden palindrome-to-near-miss pairs, while earlier layer/position patches recover less consistently. This is expected for a downstream state containing the classifier's decision. It does not explain how mirrored equality was computed. Head-output patching is less reliable on this checkpoint; learners should report the actual recovery rate for the component, position, and corruption they test rather than assume a canonical patch result.

## Reproduce the measurements

From the `backend` directory:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

from circuit_api.model import ModelAdapter, load_model, make_tokenizer
from circuit_api.research import plot_attention, plot_head_heatmap
from circuit_api.tools import ExperimentTools

challenge = json.loads(Path("circuit_api/challenges/palindrome-machine.json").read_text())
checkpoint = "circuit_api/checkpoints/palindrome-machine.pt"
model = ModelAdapter(load_model("palindrome-trained", checkpoint))
tokenizer = make_tokenizer("palindrome-trained", model.d_model)
tools = ExperimentTools(model)
public = [{"tokens": row} for row in challenge["public_dataset"]]
hidden = [{"tokens": row} for row in challenge["hidden_dataset"]]

print("public", tools.evaluate(public))
print("hidden", tools.evaluate(hidden))

head_effects = []
for layer in range(model.n_layers):
    for head in range(model.n_heads):
        per_example = [tools.ablate_head(row["tokens"], layer, head)["drop"] for row in hidden]
        head_effects.append((layer, head, sum(per_example) / len(per_example)))

plot_head_heatmap(head_effects, model.n_layers, model.n_heads)
plt.show()

clean = list("ABCCBA")
corrupted = list("ABCCBB")
logits, cache = model.run_with_cache(clean)
print("cache shapes", {name: tuple(value.shape) for name, value in cache.items()})
pattern = cache["blocks.0.attn.pattern"][0]
plot_attention(pattern, tokenizer.display_tokens(clean), title="L0H0 · clean palindrome")
plt.show()

patch = tools.patch_residual(clean, corrupted, layer=2, position=-1)
print({key: patch[key] for key in ("clean_logit_difference", "corrupted_logit_difference", "patched_logit_difference", "recovery")})
```

The figures come from this Python code. The grader does not trust the printed effects: it independently recomputes a submitted candidate intervention and the hidden-set metrics.

## What the evidence does and does not say

The strongest current evidence supports causal contributions from multiple early heads and shows that a late residual state carries enough information to restore the decision in this particular corruption setup. It does not establish that either head is a universal “palindrome head,” that the late residual state explains the algorithm, or that the circuit is complete. The classifier includes MLP and residual transformations, and the final position can aggregate information through several paths.

A next investigation should decompose this checkpoint's actual QK and OV matrices, track the head's output through later residual/MLP computations, stratify ablation and patch recovery by length and mismatch position, and compare targeted activation patches with matched control positions. Those analyses may strengthen, revise, or falsify the current hypothesis. Automated scoring here deliberately reports behavioral intervention evidence rather than pretending to certify a unique natural-language circuit explanation.

## ARENA relationship and content provenance

The research question is inspired by the public ARENA Monthly Algorithmic Problems sequence-classification task. The model, architecture, training examples, checkpoint, challenge wording, tests, and analysis above were independently created for CircuitLab. No ARENA notebook, code, dataset, checkpoint, or solution is copied into this repository. See `docs/ARENA_SKILL_COVERAGE.md` and ARENA's [materials-use policy](https://www.arena.education/using-our-materials).
