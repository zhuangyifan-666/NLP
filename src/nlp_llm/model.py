from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = True
    pad_id: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_embd = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.transformer["drop"](self.transformer["wte"](input_ids) + self.transformer["wpe"](pos))
        for block in self.transformer["h"]:
            x = block(x)
        hidden = self.transformer["ln_f"](x)
        logits = self.lm_head(hidden)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            if loss_mask is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            else:
                flat_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                mask = loss_mask.reshape(-1).to(flat_loss.dtype)
                valid = (labels.reshape(-1) != -100).to(flat_loss.dtype)
                denom = (mask * valid).sum().clamp_min(1.0)
                loss = (flat_loss * mask).sum() / denom
            out["loss"] = loss
        if return_hidden:
            out["hidden_states"] = hidden
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.block_size :]
            logits = self(idx_cond)["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                keep = torch.cumsum(probs, dim=-1) <= top_p
                keep[..., 0] = True
                filtered = sorted_logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
                logits = torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(1, sorted_idx, filtered)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_id is not None and torch.all(next_id.squeeze(-1) == eos_id):
                break
        return input_ids

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and not name.endswith("bias"):
                decay.append(param)
            else:
                no_decay.append(param)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate,
            betas=betas,
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    best_val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    config = raw_model.config.to_dict() if hasattr(raw_model, "config") else None
    payload: dict[str, Any] = {
        "model": raw_model.state_dict(),
        "config": config,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    payload.update(extra or {})
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_gpt_from_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> GPT:
    ckpt = load_checkpoint(path, map_location=map_location)
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"Checkpoint missing model config: {path}")
    model = GPT(GPTConfig.from_dict(ckpt["config"]))
    lora_cfg = ckpt.get("lora")
    if isinstance(lora_cfg, dict) and lora_cfg.get("enabled", False):
        try:
            from advanced.lora import LoRAConfig, inject_lora
        except ImportError as exc:
            raise RuntimeError("Checkpoint uses LoRA, but advanced/lora.py is not importable.") from exc
        inject_lora(model, LoRAConfig.from_dict(lora_cfg))
    model.load_state_dict(strip_module_prefix(ckpt["model"]))
    return model
