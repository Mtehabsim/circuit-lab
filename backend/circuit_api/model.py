"""Small torch transformer checkpoint for the previous-token challenge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn


def _is_ablated(ablate: Any, layer: int, head: int) -> bool:
    if isinstance(ablate, (set, frozenset, list)):
        return (layer, head) in ablate
    return ablate == (layer, head)


@dataclass(frozen=True)
class RunTrace:
    logits: Tensor
    attention: Tensor
    residual: Tensor
    head_values: Tensor
    head_outputs: Tensor
    residual_by_layer: Tensor | None = None
    query: Tensor | None = None
    key: Tensor | None = None
    attention_logits: Tensor | None = None


class ModelAdapter(nn.Module):
    """Common, named-activation interface for the small challenge checkpoints."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            aliases = {"W_Q": "q_proj", "W_K": "k_proj", "W_V": "v_proj", "W_O": "o_proj"}
            return getattr(inner, aliases.get(name, name))

    def forward(self, tokens: Sequence[str], *, ablate=None) -> Tensor:
        return self.inner(tokens, ablate=ablate)

    def run_with_trace(self, tokens: Sequence[str], *, ablate=None, patch=None) -> RunTrace:
        return self.inner.run_with_trace(tokens, ablate=ablate, patch=patch)

    def run(self, prompt: Sequence[str] | str) -> Tensor:
        return self(prompt.split() if isinstance(prompt, str) else prompt)

    def run_with_cache(self, prompt: Sequence[str] | str, *, ablate=None) -> tuple[Tensor, dict[str, Tensor]]:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        trace = self.run_with_trace(tokens, ablate=ablate)
        return trace.logits, self.activation_cache(tokens, ablate=ablate)

    def get_logits(self, prompt: Sequence[str] | str) -> Tensor:
        return self.run(prompt)

    def get_attention(self, prompt: Sequence[str] | str, layer: int, head: int) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        return self.run_with_trace(tokens).attention[layer, head]

    def get_q(self, prompt: Sequence[str] | str, layer: int, head: int) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        trace = self.run_with_trace(tokens)
        if trace.query is None:
            raise ValueError("This model does not expose query activations")
        return trace.query[layer, head]

    def get_k(self, prompt: Sequence[str] | str, layer: int, head: int) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        trace = self.run_with_trace(tokens)
        if trace.key is None:
            raise ValueError("This model does not expose key activations")
        return trace.key[layer, head]

    def get_v(self, prompt: Sequence[str] | str, layer: int, head: int) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        return self.run_with_trace(tokens).head_values[layer, head]

    def get_head_output(self, prompt: Sequence[str] | str, layer: int, head: int) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        return self.run_with_trace(tokens).head_outputs[layer, head]

    def get_residual(self, prompt: Sequence[str] | str, layer: int | None = None) -> Tensor:
        tokens = prompt.split() if isinstance(prompt, str) else prompt
        name = "resid.final" if layer is None else f"blocks.{layer}.resid.post"
        return self.activation(tokens, name)

    def activation_cache(self, tokens: Sequence[str], *, ablate=None) -> dict[str, Tensor]:
        trace = self.run_with_trace(tokens, ablate=ablate)
        cache = {"logits": trace.logits, "resid.final": trace.residual}
        for layer in range(self.n_layers):
            cache[f"blocks.{layer}.attn.pattern"] = trace.attention[layer]
            cache[f"blocks.{layer}.attn.value"] = trace.head_values[layer]
            cache[f"blocks.{layer}.attn.output"] = trace.head_outputs[layer]
            cache[f"blocks.{layer}.hook_resid_pre"] = trace.residual_by_layer[layer] if trace.residual_by_layer is not None else trace.residual
            if trace.residual_by_layer is not None:
                cache[f"blocks.{layer}.resid.post"] = trace.residual_by_layer[layer + 1]
                cache[f"blocks.{layer}.hook_resid_post"] = trace.residual_by_layer[layer + 1]
            if getattr(trace, "mlp_activations", None) is not None:
                cache[f"blocks.{layer}.mlp.hook_post"] = trace.mlp_activations[layer]
                cache[f"blocks.{layer}.mlp.hook_out"] = trace.mlp_outputs[layer]
            if trace.query is not None and trace.key is not None:
                cache[f"blocks.{layer}.attn.q"] = trace.query[layer]
                cache[f"blocks.{layer}.attn.k"] = trace.key[layer]
            if trace.attention_logits is not None:
                cache[f"blocks.{layer}.attn.logits"] = trace.attention_logits[layer]
        if trace.residual_by_layer is not None:
            cache["resid.pre"] = trace.residual_by_layer[0]
        return cache

    def activation(self, tokens: Sequence[str], name: str, *, ablate=None) -> Tensor:
        cache = self.activation_cache(tokens, ablate=ablate)
        try:
            return cache[name]
        except KeyError as error:
            raise ValueError(f"Unknown activation name: {name}") from error


class SequenceTokenizer:
    """Maps symbols to per-sequence identity vectors, including unseen symbols."""

    def __init__(self, width: int = 64) -> None:
        self.width = width

    def encode(self, tokens: Sequence[str]) -> Tensor:
        ids: dict[str, int] = {}
        return torch.tensor(
            [ids.setdefault(token, len(ids)) for token in tokens], dtype=torch.long
        )

    def decode_id(self, token_id: int, tokens: Sequence[str]) -> str:
        symbols = list(dict.fromkeys(tokens))
        return symbols[token_id]


class PreviousTokenTransformer(nn.Module):
    """A causal attention model with an inspectable previous-position circuit.

    The checkpoint uses a one-hot sequence-local embedding and a relative-position
    attention bias. Its target head copies the attended value into the residual
    stream. Keeping the model tiny makes the causal mechanism directly inspectable.
    """

    n_layers = 4
    n_heads = 4

    def __init__(self, d_model: int = 64, max_seq_len: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(d_model, d_model)
        self.q_proj = nn.Parameter(torch.zeros(self.n_layers, self.n_heads, d_model, d_model))
        self.k_proj = nn.Parameter(torch.zeros(self.n_layers, self.n_heads, d_model, d_model))
        self.v_proj = nn.Parameter(torch.zeros(self.n_layers, self.n_heads, d_model, d_model))
        self.o_proj = nn.Parameter(torch.zeros(self.n_layers, self.n_heads, d_model, d_model))
        self.relative_bias = nn.Parameter(
            torch.full((self.n_layers, self.n_heads, max_seq_len, max_seq_len), -8.0)
        )
        self.unembed = nn.Parameter(torch.eye(d_model))
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.d_model))
            self.unembed.copy_(torch.eye(self.d_model))
            self.q_proj.zero_()
            self.k_proj.zero_()
            self.v_proj.zero_()
            self.o_proj.zero_()
            self.relative_bias.zero_()

    def run_with_trace(
        self,
        tokens: Sequence[str],
        *,
        ablate: Any = None,
        patch: tuple[int, int, int, Tensor] | None = None,
    ) -> RunTrace:
        ids = SequenceTokenizer(self.d_model).encode(tokens).to(self.embedding.weight.device)
        length = len(tokens)
        if length == 0:
            empty = torch.empty((0, self.d_model), device=ids.device)
            empty_heads = torch.empty((self.n_layers, self.n_heads, 0, self.d_model), device=ids.device)
            return RunTrace(torch.empty((0, self.d_model), device=ids.device), empty, empty, empty_heads, empty_heads)
        if length > self.max_seq_len or int(ids.max()) >= self.d_model:
            raise ValueError(f"sequence must contain at most {self.d_model} unique tokens")

        residual = self.embedding(ids)
        patterns = torch.zeros(self.n_layers, self.n_heads, length, length, device=residual.device)
        values = torch.zeros(self.n_layers, self.n_heads, length, self.d_model, device=residual.device)
        outputs = torch.zeros_like(values)
        queries = torch.zeros_like(values)
        keys = torch.zeros_like(values)
        attention_logits = torch.zeros(self.n_layers, self.n_heads, length, length, device=residual.device)
        residual_by_layer = [residual.clone()]
        for layer in range(self.n_layers):
            layer_output = torch.zeros_like(residual)
            for head in range(self.n_heads):
                q = residual @ self.q_proj[layer, head]
                k = residual @ self.k_proj[layer, head]
                v = residual @ self.v_proj[layer, head]
                queries[layer, head] = q
                keys[layer, head] = k
                values[layer, head] = v
                scores = q @ k.T / math.sqrt(self.d_model)
                scores = scores + self.relative_bias[layer, head, :length, :length]
                scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
                attention_logits[layer, head] = scores
                pattern = scores.softmax(dim=-1)
                patterns[layer, head] = pattern
                result = pattern @ v @ self.o_proj[layer, head]
                outputs[layer, head] = result
                if _is_ablated(ablate, layer, head):
                    result = torch.zeros_like(result)
                if patch is not None:
                    patch_layer, patch_head, position, replacement = patch
                    if (layer, head) == (patch_layer, patch_head) and 0 <= position < length:
                        result = result.clone()
                        result[position] = replacement
                layer_output = layer_output + result
            residual = residual + layer_output
            residual_by_layer.append(residual.clone())

        logits = residual @ self.unembed.T
        return RunTrace(logits=logits, attention=patterns, residual=residual, head_values=values, head_outputs=outputs,
                        residual_by_layer=torch.stack(residual_by_layer), query=queries, key=keys,
                        attention_logits=attention_logits)

    def forward(self, tokens: Sequence[str], *, ablate: tuple[int, int] | None = None) -> Tensor:
        return self.run_with_trace(tokens, ablate=ablate).logits


class InductionTransformer(nn.Module):
    """Tiny two-stage induction circuit with an inspectable previous-token feature."""

    n_layers = 4
    n_heads = 4
    d_model = 128
    max_seq_len = 64
    behavior = "induction"

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.d_model, self.d_model)
        self.w_q = nn.Parameter(torch.eye(64) * 8)
        self.w_k = nn.Parameter(torch.eye(64) * 8)
        self.unembed = nn.Parameter(torch.eye(self.d_model))
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), 1))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.d_model))

    def run_with_trace(self, tokens: Sequence[str], *, ablate=None, patch=None) -> RunTrace:
        ids = SequenceTokenizer(self.d_model).encode(tokens)
        length = len(tokens)
        if length == 0:
            empty = torch.empty((0, self.d_model))
            heads = torch.empty((self.n_layers, self.n_heads, 0, self.d_model))
            return RunTrace(empty, torch.empty((self.n_layers, self.n_heads, 0, 0)), empty, heads, heads)
        if length > self.max_seq_len or int(ids.max()) >= self.d_model:
            raise ValueError("sequence exceeds the model's context or vocabulary size")
        token_vectors = self.embedding(ids)
        residual = torch.zeros_like(token_vectors)
        residual[:, :64] = token_vectors[:, :64]
        residual_initial = residual.clone()
        attention = torch.zeros(self.n_layers, self.n_heads, length, length)
        values = torch.zeros(self.n_layers, self.n_heads, length, self.d_model)
        outputs = torch.zeros_like(values)

        # First head writes the previous token identity into a separate residual subspace.
        for position in range(1, length):
            if not _is_ablated(ablate, 0, 0):
                attention[0, 0, position, position - 1] = 1.0
                residual[position, 64 + int(ids[position - 1])] = 1.0
                values[0, 0, position - 1, :64] = token_vectors[position - 1, :64]
                outputs[0, 0, position, 64 + int(ids[position - 1])] = 1.0
        residual_after_layer0 = residual.clone()

        # QK compares the current token with previous-token features; the head's
        # values copy the token that followed the matching earlier occurrence.
        q = residual[:, :64] @ self.w_q
        k = residual[:, 64:] @ self.w_k
        scores = q @ k.T / math.sqrt(64)
        valid_keys = residual[:, 64:].abs().sum(dim=-1) > 0
        scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
        scores = scores.masked_fill(~valid_keys.unsqueeze(0), float("-inf"))
        values[1, 1, :, :64] = token_vectors[:, :64]
        queries = torch.zeros(self.n_layers, self.n_heads, length, self.d_model)
        keys = torch.zeros_like(queries)
        attention_logits = torch.zeros(self.n_layers, self.n_heads, length, length)
        queries[1, 1, :, :64] = q
        keys[1, 1, :, :64] = k
        attention_logits[1, 1] = scores
        for position in range(1, length):
            row = scores[position]
            if not torch.isfinite(row).any():
                continue
            attention[1, 1, position] = row.softmax(dim=-1)
            if not _is_ablated(ablate, 1, 1):
                output = attention[1, 1, position] @ values[1, 1]
                outputs[1, 1, position] = output
                residual[position] = output
        if patch is not None:
            layer, head, position, replacement = patch
            if (layer, head) == (1, 1) and 0 <= position < length:
                residual[position] = replacement
                outputs[layer, head, position] = replacement
        logits = residual @ self.unembed.T
        residual_by_layer = torch.stack([residual_initial, residual_after_layer0, residual, residual, residual])
        return RunTrace(logits, attention, residual, values, outputs, residual_by_layer, queries, keys, attention_logits)

    def forward(self, tokens: Sequence[str], *, ablate=None) -> Tensor:
        return self.run_with_trace(tokens, ablate=ablate).logits

    @staticmethod
    def target_id(tokens: Sequence[str]) -> int | None:
        ids = SequenceTokenizer(128).encode(tokens).tolist()
        if len(ids) < 3:
            return None
        query = ids[-1]
        for source in range(1, len(ids) - 1):
            if ids[source - 1] == query:
                return ids[source]
        return None


class IOIToyTransformer(nn.Module):
    """Role-coded name-mover toy with a subject-finding and name-moving head."""

    n_layers = 4
    n_heads = 4
    d_model = 8
    max_seq_len = 32
    behavior = "ioi"

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.d_model, self.d_model)
        self.unembed = nn.Parameter(torch.eye(self.d_model))
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), 1))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(self.d_model))

    @staticmethod
    def encode_tokens(tokens: Sequence[str]) -> Tensor:
        if len(tokens) < 5 or tokens[0] == tokens[1]:
            raise ValueError("IOI inputs need two distinct names followed by role, context, and query tokens")
        fixed = {"SUBJECT_A": 2, "SUBJECT_B": 3, "GAVE": 4, "BOOK": 5, "TO": 6, "QUERY": 7}
        ids = []
        for position, token in enumerate(tokens):
            if position == 0:
                ids.append(0)
            elif position == 1:
                ids.append(1)
            elif token in fixed:
                ids.append(fixed[token])
            else:
                raise ValueError(f"unknown IOI role token: {token}")
        if not any(token in {"SUBJECT_A", "SUBJECT_B"} for token in tokens):
            raise ValueError("IOI input must contain a SUBJECT_A or SUBJECT_B role token")
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def target_id(tokens: Sequence[str]) -> int:
        if "SUBJECT_A" in tokens:
            return 1
        if "SUBJECT_B" in tokens:
            return 0
        raise ValueError("IOI input is missing its subject role")

    def run_with_trace(self, tokens: Sequence[str], *, ablate=None, patch=None) -> RunTrace:
        ids = self.encode_tokens(tokens)
        length = len(tokens)
        residual = torch.zeros(length, self.d_model)
        attention = torch.zeros(self.n_layers, self.n_heads, length, length)
        values = torch.zeros(self.n_layers, self.n_heads, length, self.d_model)
        outputs = torch.zeros_like(values)
        query = length - 1
        subject_position = next((i for i, token in enumerate(tokens) if token in {"SUBJECT_A", "SUBJECT_B"}), None)
        subject = self.target_id(tokens) ^ 1
        if subject_position is not None and not _is_ablated(ablate, 0, 0):
            attention[0, 0, query, subject_position] = 1.0
            outputs[0, 0, query, subject] = 1.0
        chosen_name = 1 - subject if subject_position is not None and not _is_ablated(ablate, 0, 0) else 0
        name_position = chosen_name
        if not _is_ablated(ablate, 2, 1):
            attention[2, 1, query, name_position] = 1.0
            values[2, 1, name_position, chosen_name] = 1.0
            outputs[2, 1, query, chosen_name] = 1.0
            residual[query, chosen_name] = 1.0
        if patch is not None:
            layer, head, position, replacement = patch
            if (layer, head) == (2, 1) and position == query:
                residual[query] = replacement
                outputs[layer, head, query] = replacement
        return RunTrace(residual @ self.unembed.T, attention, residual, values, outputs)

    def forward(self, tokens: Sequence[str], *, ablate=None) -> Tensor:
        return self.run_with_trace(tokens, ablate=ablate).logits


class IOITokenizer:
    def encode(self, tokens: Sequence[str]) -> Tensor:
        return IOIToyTransformer.encode_tokens(tokens)

    def decode_id(self, token_id: int, tokens: Sequence[str]) -> str:
        return tokens[token_id] if token_id in (0, 1) else str(token_id)


def build_checkpoint(path: str) -> None:
    model = PreviousTokenTransformer()
    with torch.no_grad():
        model.relative_bias.fill_(-8.0)
        for position in range(model.max_seq_len):
            model.relative_bias[2, 1, position, position - 1 if position else 0] = 12.0
            # The decoy tracks the same token but has no output pathway.
            model.relative_bias[2, 0, position, position - 1 if position else 0] = 12.0
        model.v_proj[2, 1].copy_(torch.eye(model.d_model))
        model.v_proj[2, 0].copy_(torch.eye(model.d_model))
        model.o_proj[2, 1].copy_(2 * torch.eye(model.d_model))
    torch.save({"state_dict": model.state_dict(), "format": 1}, path)


def load_checkpoint(path: str) -> PreviousTokenTransformer:
    model = PreviousTokenTransformer()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_induction_checkpoint(path: str) -> None:
    model = InductionTransformer()
    torch.save({"state_dict": model.state_dict(), "format": 1}, path)


def load_induction_checkpoint(path: str) -> InductionTransformer:
    model = InductionTransformer()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_ioi_checkpoint(path: str) -> None:
    model = IOIToyTransformer()
    torch.save({"state_dict": model.state_dict(), "format": 1}, path)


def load_ioi_checkpoint(path: str) -> IOIToyTransformer:
    model = IOIToyTransformer()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_model(model_name: str, path: str) -> nn.Module:
    if model_name == "palindrome-trained":
        from .palindrome import load_checkpoint as load_palindrome_checkpoint
        return load_palindrome_checkpoint(path)
    if model_name == "induction-toy":
        return load_induction_checkpoint(path)
    if model_name in {"prev-token-toy", "decoy-head-toy", "ov-path-toy", "patching-toy"}:
        return load_checkpoint(path)
    if model_name == "ioi-toy":
        return load_ioi_checkpoint(path)
    raise ValueError(f"Unknown checkpoint model: {model_name}")


def make_tokenizer(model_name: str, width: int) -> SequenceTokenizer | IOITokenizer:
    if model_name == "palindrome-trained":
        from .palindrome import PalindromeTokenizer
        return PalindromeTokenizer()
    if model_name == "ioi-toy":
        return IOITokenizer()
    return SequenceTokenizer(width)
