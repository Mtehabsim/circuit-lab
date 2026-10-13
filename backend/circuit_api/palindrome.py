"""Independently trained small decoder transformer for palindrome research."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ALPHABET = tuple("ABCDEFGH")
BOS, EOS = "<BOS>", "<EOS>"


class PalindromeTokenizer:
    def __init__(self) -> None:
        self.vocab = ALPHABET

    def encode(self, text: str | Sequence[str]) -> Tensor:
        tokens = list(text) if isinstance(text, str) else list(text)
        try:
            return torch.tensor([ALPHABET.index(token) for token in tokens], dtype=torch.long)
        except ValueError as error:
            raise ValueError(f"Use symbols from {''.join(ALPHABET)}") from error

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(ALPHABET[int(index)] for index in ids)

    @staticmethod
    def display_tokens(tokens: Sequence[str]) -> list[str]:
        return [BOS, *tokens, EOS]


class PalindromeTransformer(nn.Module):
    n_layers = 3
    n_heads = 4
    d_model = 48
    max_content_length = 16
    behavior = "palindrome"
    vocab = ALPHABET + (BOS, EOS)

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(self.vocab), self.d_model)
        self.position = nn.Embedding(self.max_content_length + 2, self.d_model)
        self.q_proj = nn.Parameter(torch.empty(self.n_layers, self.n_heads, self.d_model, self.d_model // self.n_heads))
        self.k_proj = nn.Parameter(torch.empty_like(self.q_proj))
        self.v_proj = nn.Parameter(torch.empty_like(self.q_proj))
        self.o_proj = nn.Parameter(torch.empty(self.n_layers, self.n_heads, self.d_model // self.n_heads, self.d_model))
        self.ln_attn = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.n_layers))
        self.ln_mlp = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.n_layers))
        self.mlp_in = nn.Parameter(torch.empty(self.n_layers, self.d_model, 2 * self.d_model))
        self.mlp_out = nn.Parameter(torch.empty(self.n_layers, 2 * self.d_model, self.d_model))
        self.unembed = nn.Linear(self.d_model, 2)
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_content_length + 2, self.max_content_length + 2, dtype=torch.bool), diagonal=1))
        nn.init.normal_(self.embedding.weight, std=0.08)
        nn.init.normal_(self.position.weight, std=0.04)
        for tensor in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.mlp_in, self.mlp_out):
            nn.init.xavier_uniform_(tensor)

    def encode(self, tokens: Sequence[str]) -> Tensor:
        try:
            ids = [self.vocab.index(str(token)) for token in (BOS, *tokens, EOS)]
        except ValueError as error:
            raise ValueError(f"Tokens must be single symbols from {''.join(ALPHABET)}") from error
        if len(ids) > self.max_content_length + 2:
            raise ValueError(f"Sequences may contain at most {self.max_content_length} symbols")
        return torch.tensor(ids, dtype=torch.long, device=self.embedding.weight.device)

    def target_id(self, tokens: Sequence[str]) -> int:
        return int(list(tokens) == list(tokens)[::-1])

    def forward_batch(self, batch_tokens: Sequence[Sequence[str]], *, ablate=None) -> Tensor:
        """Batched forward pass used by training, with padding excluded from keys."""
        device = self.embedding.weight.device
        encoded = [self.encode(tokens) for tokens in batch_tokens]
        lengths = torch.tensor([len(row) for row in encoded], device=device)
        width = int(max(len(row) for row in encoded))
        ids = torch.full((len(encoded), width), self.vocab.index(EOS), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = row
        valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
        x = self.embedding(ids) + self.position(torch.arange(width, device=device))[None, :, :]
        causal = self.causal_mask[:width, :width][None, None]
        key_padding = ~valid[:, None, None, :]
        for layer in range(self.n_layers):
            z = self.ln_attn[layer](x)
            q = torch.einsum("btd,hdf->bhtf", z, self.q_proj[layer])
            k = torch.einsum("btd,hdf->bhtf", z, self.k_proj[layer])
            v = torch.einsum("btd,hdf->bhtf", z, self.v_proj[layer])
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(causal | key_padding, -torch.inf)
            pattern = scores.softmax(dim=-1)
            heads = torch.einsum("bhts,bhsd->bhtd", pattern, v)
            heads = torch.einsum("bhtd,hdf->bhtf", heads, self.o_proj[layer])
            if ablate is not None:
                pairs = {ablate} if isinstance(ablate, tuple) and len(ablate) == 2 and isinstance(ablate[0], int) else set(ablate)
                for layer_id, head_id in pairs:
                    if layer_id == layer:
                        heads[:, head_id] = 0
            x = x + heads.sum(dim=1)
            x = x + F.gelu(self.ln_mlp[layer](x) @ self.mlp_in[layer]) @ self.mlp_out[layer]
        final_positions = lengths - 1
        return self.unembed(x[torch.arange(len(encoded), device=device), final_positions])

    def run_with_trace(self, tokens: Sequence[str], *, ablate: Any = None, patch=None):
        ids = self.encode(tokens)
        x = self.embedding(ids) + self.position(torch.arange(len(ids), device=ids.device))
        residuals = [x]
        attention, q_all, k_all, v_all, outputs, attn_logits, mlp_acts, mlp_outputs = [], [], [], [], [], [], [], []
        for layer in range(self.n_layers):
            residual = x
            z = self.ln_attn[layer](x)
            q = torch.einsum("td,hdf->htf", z, self.q_proj[layer])
            k = torch.einsum("td,hdf->htf", z, self.k_proj[layer])
            v = torch.einsum("td,hdf->htf", z, self.v_proj[layer])
            scores = torch.einsum("htd,hsd->hts", q, k) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(self.causal_mask[:len(ids), :len(ids)], -torch.inf)
            pattern = scores.softmax(dim=-1)
            head = torch.einsum("hts,hsd->htd", pattern, v)
            head_out = torch.einsum("htd,hdf->htf", head, self.o_proj[layer])
            for h in range(self.n_heads):
                if (layer, h) == ablate or (isinstance(ablate, (set, list, tuple)) and (layer, h) in ablate):
                    head_out[h] = 0
            attn_update = head_out.sum(dim=0)
            if isinstance(patch, tuple) and len(patch) == 4 and patch[0] == layer:
                _, patch_head, pos, replacement = patch
                if 0 <= pos < len(ids):
                    head_out[patch_head, pos] = replacement
                    attn_update = head_out.sum(dim=0)
            x = residual + attn_update
            mlp_post = F.gelu(self.ln_mlp[layer](x) @ self.mlp_in[layer])
            mlp_out = mlp_post @ self.mlp_out[layer]
            x = x + mlp_out
            if isinstance(patch, dict) and patch.get("type") == "residual" and patch.get("layer") == layer:
                position = int(patch["position"])
                if 0 <= position < len(ids):
                    x = x.clone()
                    x[position] = patch["value"]
            attention.append(pattern); q_all.append(q); k_all.append(k); v_all.append(v)
            outputs.append(head_out); attn_logits.append(scores); mlp_acts.append(mlp_post); mlp_outputs.append(mlp_out); residuals.append(x)
        logits = self.unembed(x)
        return type("PalindromeTrace", (), {"logits": logits, "attention": torch.stack(attention),
            "residual": x, "residual_by_layer": torch.stack(residuals), "query": torch.stack(q_all),
            "key": torch.stack(k_all), "attention_logits": torch.stack(attn_logits),
            "head_values": torch.stack(v_all), "head_outputs": torch.stack(outputs),
            "mlp_activations": torch.stack(mlp_acts), "mlp_outputs": torch.stack(mlp_outputs)})()

    def forward(self, tokens: Sequence[str], *, ablate=None) -> Tensor:
        return self.run_with_trace(tokens, ablate=ablate).logits


def generate_examples(count: int, seed: int, lengths=range(3, 13)) -> list[tuple[list[str], int]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        length = rng.choice(list(lengths))
        if rng.random() < 0.5:
            half = [rng.choice(ALPHABET) for _ in range((length + 1) // 2)]
            seq = half + (half[:-1] if length % 2 else half)[::-1]
            label = 1
        elif rng.random() < 0.5:
            # Hard negatives differ at exactly one mirrored pair. Sample the
            # mismatch location uniformly so the classifier cannot win by
            # checking only the first/last symbol.
            half = [rng.choice(ALPHABET) for _ in range((length + 1) // 2)]
            seq = half + (half[:-1] if length % 2 else half)[::-1]
            offset = rng.randrange(length // 2)
            seq[length - 1 - offset] = next(symbol for symbol in ALPHABET if symbol != seq[length - 1 - offset])
            label = 0
        else:
            seq = [rng.choice(ALPHABET) for _ in range(length)]
            if seq == seq[::-1]:
                seq[0] = ALPHABET[(ALPHABET.index(seq[0]) + 1) % len(ALPHABET)]
            label = 0
        rows.append((seq, label))
    return rows


def train_checkpoint(path: str, *, seed: int = 1729, epochs: int = 45, examples: int = 5000) -> dict[str, Any]:
    """Train from generated examples; no task-specific weights are hard-coded."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = PalindromeTransformer()
    data = generate_examples(examples, seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    model.train()
    rng = random.Random(seed + 2)
    for _ in range(epochs):
        rng.shuffle(data)
        for start in range(0, len(data), 256):
            batch = data[start:start + 256]
            logits = model.forward_batch([tokens for tokens, _ in batch])
            labels = torch.tensor([label for _, label in batch])
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval()
    validation = generate_examples(1000, seed + 3)
    with torch.inference_mode():
        correct = sum(int(model.forward_batch([tokens])[0].argmax()) == label for tokens, label in validation)
        train_correct = sum(int(model.forward_batch([tokens])[0].argmax()) == label for tokens, label in data[:1000])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "seed": seed, "training_data_seed": seed + 1, "validation_data_seed": seed + 3,
        "epochs": epochs, "training_examples": examples, "batch_size": 256,
        "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.01,
        "gradient_clip_norm": 1.0, "validation_examples": len(validation),
        "validation_accuracy": correct / len(validation),
        "train_accuracy_sample": train_correct / min(len(data), 1000),
        "alphabet": list(ALPHABET), "max_content_length": model.max_content_length,
        "training_content_lengths": [3, 12],
        "architecture": {"layers": model.n_layers, "heads": model.n_heads, "d_model": model.d_model,
            "head_dim": model.d_model // model.n_heads, "mlp_width": 2 * model.d_model,
            "position_embedding": "learned absolute", "attention": "causal scaled dot-product",
            "normalization": "pre-attention and pre-MLP LayerNorm", "mlp_activation": "GELU",
            "classifier": "final-position linear binary head"},
        "negative_sampling": {"hard_single_mirrored_mismatch_fraction": 0.5, "random_negative_fraction": 0.5,
            "hard_mismatch_offset": "uniform over mirrored pairs"},
        "asset_source": "CircuitLab-generated synthetic training data and random initialization; no external checkpoint or dataset",
    }
    torch.save({"state_dict": model.state_dict(), "config": provenance}, target)
    target.with_name(f"{target.stem}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"validation_accuracy": correct / len(validation), "train_accuracy_sample": train_correct / min(len(data), 1000),
            "examples": examples, "epochs": epochs, "seed": seed}


def load_checkpoint(path: str) -> PalindromeTransformer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = PalindromeTransformer()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
