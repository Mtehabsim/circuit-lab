from __future__ import annotations

from typing import Any, Sequence

import torch

from .model import SequenceTokenizer


class ExperimentTools:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.events: list[dict[str, Any]] = []
        self.traces: dict[str, Any] = {}

    @staticmethod
    def _accuracy(model: Any, dataset: list[dict[str, Any]], ablate=None) -> float:
        correct = total = 0
        with torch.inference_mode():
            for item in dataset:
                tokens = item["tokens"]
                if getattr(model, "behavior", None) == "induction":
                    expected = model.target_id(tokens)
                    if expected is not None:
                        predicted = int(model(tokens, ablate=ablate)[-1].argmax())
                        correct += predicted == expected
                        total += 1
                    continue
                if getattr(model, "behavior", None) == "ioi":
                    expected = model.target_id(tokens)
                    predicted = int(model(tokens, ablate=ablate)[-1].argmax())
                    correct += predicted == expected
                    total += 1
                    continue
                if getattr(model, "behavior", None) == "palindrome":
                    expected = model.target_id(tokens)
                    predicted = int(model(tokens, ablate=ablate)[-1].argmax())
                    correct += predicted == expected
                    total += 1
                    continue
                if len(tokens) < 2:
                    continue
                predicted = model(tokens, ablate=ablate).argmax(dim=-1).tolist()
                token_ids = SequenceTokenizer(model.d_model).encode(tokens).tolist()
                correct += sum(predicted[position] == token_ids[position - 1] for position in range(1, len(tokens)))
                total += len(tokens) - 1
        return correct / total if total else 0.0

    def evaluate(self, dataset: list[dict[str, Any]]) -> dict[str, float | int]:
        correct = total = 0
        with torch.inference_mode():
            for item in dataset:
                tokens = item["tokens"]
                if getattr(self.model, "behavior", None) == "induction":
                    expected = self.model.target_id(tokens)
                    if expected is not None:
                        predicted = int(self.model(tokens)[-1].argmax())
                        correct += predicted == expected
                        total += 1
                    continue
                if getattr(self.model, "behavior", None) == "ioi":
                    expected = self.model.target_id(tokens)
                    predicted = int(self.model(tokens)[-1].argmax())
                    correct += predicted == expected
                    total += 1
                    continue
                if getattr(self.model, "behavior", None) == "palindrome":
                    expected = self.model.target_id(tokens)
                    predicted = int(self.model(tokens)[-1].argmax())
                    correct += predicted == expected
                    total += 1
                    continue
                if len(tokens) < 2:
                    continue
                predicted = self.model(tokens).argmax(dim=-1).tolist()
                token_ids = SequenceTokenizer(self.model.d_model).encode(tokens).tolist()
                correct += sum(predicted[position] == token_ids[position - 1] for position in range(1, len(tokens)))
                total += len(tokens) - 1
        return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}

    def attention(self, tokens: Sequence[str], layer: int, head: int) -> list[list[float]]:
        self._validate_component(layer, head)
        with torch.inference_mode():
            pattern = self.model.run_with_trace(tokens).attention[layer, head]
        result = pattern.cpu().tolist()
        self.traces["attention"] = {"layer": layer, "head": head, "tokens": list(tokens), "weights": result}
        self._record({"type": "attention", "layer": layer, "head": head, "tokens": list(tokens)})
        return result

    def activation(self, tokens: Sequence[str], name: str) -> list[Any]:
        tensor = self.model.activation(tokens, name)
        result = torch.nan_to_num(tensor.detach().cpu(), nan=0.0, posinf=1e9, neginf=-1e9).tolist()
        self._record({"type": "activation_read", "name": name, "shape": list(tensor.shape)})
        return result

    def logit_lens(self, tokens: Sequence[str], layer: int | None = None) -> list[list[float]]:
        activation_name = "resid.final" if layer is None else f"blocks.{layer}.resid.post"
        residual = self.model.activation(tokens, activation_name)
        logits = residual @ self.model.unembed.T
        result = torch.nan_to_num(logits.detach().cpu(), nan=0.0, posinf=1e9, neginf=-1e9).tolist()
        self._record({"type": "logit_lens", "layer": layer, "positions": len(tokens)})
        return result

    def weight(self, name: str, layer: int | None = None, head: int | None = None) -> list[Any]:
        aliases = {"W_Q": "q_proj", "W_K": "k_proj", "W_V": "v_proj", "W_O": "o_proj",
                   "W_Q_induction": "w_q", "W_K_induction": "w_k", "W_U": "unembed", "W_E": "embedding.weight"}
        attribute = aliases.get(name, name)
        value: Any = self.model
        try:
            for part in attribute.split("."):
                value = getattr(value, part)
            tensor = value.detach()
        except AttributeError as error:
            raise ValueError(f"Unknown model weight: {name}") from error
        if tensor.ndim >= 4:
            if layer is None or head is None:
                raise ValueError("layer and head are required for per-head projection weights")
            self._validate_component(layer, head)
            tensor = tensor[layer, head]
        result = torch.nan_to_num(tensor.cpu(), nan=0.0, posinf=1e9, neginf=-1e9).tolist()
        self._record({"type": "weight_read", "name": name, "shape": list(tensor.shape), "layer": layer, "head": head})
        return result

    def ablate_head(self, tokens: Sequence[str], layer: int, head: int) -> dict[str, Any]:
        self._validate_component(layer, head)
        one = [{"tokens": list(tokens)}]
        before = self._accuracy(self.model, one)
        after = self._accuracy(self.model, one, ablate=(layer, head))
        event = {"type": "ablation", "layer": layer, "head": head, "tokens": list(tokens),
                 "before": before, "after": after, "drop": before - after}
        return self._record(event)

    def ablate_components(self, dataset: list[dict[str, Any]], components: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Ablate a proposed circuit together and measure its task effect."""
        pairs = {(int(item["layer"]), int(item["head"])) for item in components
                 if isinstance(item, dict) and item.get("type", "attention_head") == "attention_head"}
        if not pairs:
            raise ValueError("components must contain at least one attention head")
        for layer, head in pairs:
            self._validate_component(layer, head)
        before = self._accuracy(self.model, dataset)
        after = self._accuracy(self.model, dataset, ablate=pairs)
        event = {"type": "circuit_ablation", "components": [
            {"type": "attention_head", "layer": layer, "head": head} for layer, head in sorted(pairs)
        ], "before": before, "after": after, "drop": before - after, "n_examples": len(dataset)}
        return self._record(event)

    def value_path(self, tokens: Sequence[str], layer: int, head: int, position: int) -> dict[str, Any]:
        self._validate_component(layer, head)
        if not isinstance(position, int) or not 1 <= position < len(tokens):
            raise ValueError("position must be a prediction position from 1 to sequence length - 1")
        with torch.inference_mode():
            trace = self.model.run_with_trace(tokens)
            source_position = int(trace.attention[layer, head, position].argmax())
            value = trace.head_values[layer, head, source_position]
            output = trace.head_outputs[layer, head, position]
            expected_id = int(SequenceTokenizer(self.model.d_model).encode(tokens)[position - 1])
            logit_contribution = float(output @ self.model.unembed[expected_id])
        event = {
            "type": "ov_analysis", "layer": layer, "head": head, "position": position,
            "tokens": list(tokens),
            "source_position": source_position, "source_token": tokens[source_position],
            "expected_token": tokens[position - 1], "value_norm": float(value.norm()),
            "output_norm": float(output.norm()), "target_logit_contribution": logit_contribution,
        }
        self.traces["value_path"] = event
        event = self._record(event)
        return event

    def patch_head_output(
        self,
        clean_tokens: Sequence[str],
        corrupted_tokens: Sequence[str],
        layer: int,
        head: int,
        position: int,
    ) -> dict[str, Any]:
        self._validate_component(layer, head)
        if len(clean_tokens) != len(corrupted_tokens):
            raise ValueError("clean and corrupted sequences must have equal length")
        if getattr(self.model, "behavior", None) == "palindrome":
            if not 0 <= position < len(clean_tokens):
                raise ValueError("position must refer to a content token")
            internal_position = position + 1  # account for model's internal BOS marker
            with torch.inference_mode():
                clean_trace = self.model.run_with_trace(clean_tokens)
                corrupted_trace = self.model.run_with_trace(corrupted_tokens)
                replacement = clean_trace.head_outputs[layer, head, internal_position]
                patched_trace = self.model.run_with_trace(
                    corrupted_tokens, patch=(layer, head, internal_position, replacement)
                )
                clean_target = self.model.target_id(clean_tokens)
                corrupt_target = self.model.target_id(corrupted_tokens)
                clean_margin = float((clean_trace.logits[-1, clean_target] - clean_trace.logits[-1, 1-clean_target]))
                corrupted_margin = float((corrupted_trace.logits[-1, clean_target] - corrupted_trace.logits[-1, 1-clean_target]))
                patched_margin = float((patched_trace.logits[-1, clean_target] - patched_trace.logits[-1, 1-clean_target]))
            event = {"type": "activation_patch", "component": "attention_head_output", "layer": layer,
                     "head": head, "position": position, "clean_tokens": list(clean_tokens),
                     "corrupted_tokens": list(corrupted_tokens), "clean_target": clean_target,
                     "corrupted_target": corrupt_target, "clean_logit_difference": clean_margin,
                     "corrupted_logit_difference": corrupted_margin, "patched_logit_difference": patched_margin,
                     "recovery": patched_margin - corrupted_margin}
            self.traces["patching"] = event
            return self._record(event)
        if not 1 <= position < len(clean_tokens):
            raise ValueError("position must be a prediction position from 1 to sequence length - 1")
        with torch.inference_mode():
            clean_trace = self.model.run_with_trace(clean_tokens)
            corrupted_trace = self.model.run_with_trace(corrupted_tokens)
            replacement = clean_trace.head_outputs[layer, head, position]
            patched_trace = self.model.run_with_trace(
                corrupted_tokens, patch=(layer, head, position, replacement)
            )
            clean_expected = int(SequenceTokenizer(self.model.d_model).encode(clean_tokens)[position - 1])
            corrupted_expected = int(SequenceTokenizer(self.model.d_model).encode(corrupted_tokens)[position - 1])
            clean_score = float(clean_trace.logits[position, clean_expected] - clean_trace.logits[position, corrupted_expected])
            corrupted_score = float(corrupted_trace.logits[position, clean_expected] - corrupted_trace.logits[position, corrupted_expected])
            patched_score = float(patched_trace.logits[position, clean_expected] - patched_trace.logits[position, corrupted_expected])
        event = {
            "type": "activation_patch", "component": "attention_head_output", "layer": layer, "head": head,
            "position": position, "clean_expected": clean_tokens[position - 1],
            "clean_tokens": list(clean_tokens), "corrupted_tokens": list(corrupted_tokens),
            "corrupted_expected": corrupted_tokens[position - 1],
            "clean_logit_difference": clean_score, "corrupted_logit_difference": corrupted_score,
            "patched_logit_difference": patched_score, "recovery": patched_score - corrupted_score,
        }
        event = self._record(event)
        self.traces["patching"] = event
        return event

    def patch_residual(
        self, clean_tokens: Sequence[str], corrupted_tokens: Sequence[str], layer: int, position: int
    ) -> dict[str, Any]:
        """Patch one post-block residual vector; position=-1 refers to the final EOS marker."""
        if getattr(self.model, "behavior", None) != "palindrome":
            raise ValueError("Residual patching is currently enabled for the Palindrome Machine checkpoint")
        if len(clean_tokens) != len(corrupted_tokens):
            raise ValueError("clean and corrupted sequences must have equal length")
        if not 0 <= layer < self.model.n_layers:
            raise ValueError(f"layer must be an integer from 0 to {self.model.n_layers - 1}")
        internal_position = len(clean_tokens) + 1 if position == -1 else position
        if not isinstance(position, int) or not 0 <= internal_position < len(clean_tokens) + 2:
            raise ValueError("position must be an internal token index or -1 for final EOS")
        with torch.inference_mode():
            clean_trace = self.model.run_with_trace(clean_tokens)
            corrupted_trace = self.model.run_with_trace(corrupted_tokens)
            replacement = clean_trace.residual_by_layer[layer + 1, internal_position]
            patched_trace = self.model.run_with_trace(corrupted_tokens, patch={
                "type": "residual", "layer": layer, "position": internal_position, "value": replacement
            })
            clean_target = self.model.target_id(clean_tokens)
            corrupted_target = self.model.target_id(corrupted_tokens)
            clean_margin = float(clean_trace.logits[-1, clean_target] - clean_trace.logits[-1, 1-clean_target])
            corrupted_margin = float(corrupted_trace.logits[-1, clean_target] - corrupted_trace.logits[-1, 1-clean_target])
            patched_margin = float(patched_trace.logits[-1, clean_target] - patched_trace.logits[-1, 1-clean_target])
        event = {"type": "residual_patch", "layer": layer, "position": position,
                 "clean_tokens": list(clean_tokens), "corrupted_tokens": list(corrupted_tokens),
                 "clean_target": clean_target, "corrupted_target": corrupted_target,
                 "clean_logit_difference": clean_margin, "corrupted_logit_difference": corrupted_margin,
                 "patched_logit_difference": patched_margin, "recovery": patched_margin - corrupted_margin}
        self.traces["patching"] = event
        return self._record(event)

    def ablation_sweep(self, tokens: Sequence[str]) -> list[dict[str, Any]]:
        results = [self.ablate_head(tokens, layer, head) for layer in range(self.model.n_layers) for head in range(self.model.n_heads)]
        return sorted(results, key=lambda result: result["drop"], reverse=True)

    def _validate_component(self, layer: int, head: int) -> None:
        if not isinstance(layer, int) or not 0 <= layer < self.model.n_layers:
            raise ValueError(f"layer must be an integer from 0 to {self.model.n_layers - 1}")
        if not isinstance(head, int) or not 0 <= head < self.model.n_heads:
            raise ValueError(f"head must be an integer from 0 to {self.model.n_heads - 1}")

    def _record(self, event: dict[str, Any]) -> dict[str, Any]:
        versioned = {"event_version": 1, **event}
        self.events.append(versioned)
        return versioned
